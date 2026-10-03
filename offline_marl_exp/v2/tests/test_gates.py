"""
Sanity gates (review §5 step 1). Run:  python -m pytest tests -x -q     (or: python tests/test_gates.py)

G0   planner objective == env return on replay; BC target never appears among the inputs;
     dihedral augmentation really is a symmetry of the physics
GM   masks work: perturbing an entity OUTSIDE radius k leaves the masked actor/critic output unchanged and
     changes the unmasked critic
G1   every actor head (position mean, log-std, channel logits) and the critic receive a non-zero gradient
G1b  PPO ratio is exactly 1 on the first pass and moves afterwards (so the surrogate is not constant)
G2   actor weights move after fine-tuning updates
G3   (needs a trained student: results/bc_seed0.pt) distilled student beats stay-put AND random, paired CI > 0
G4   masked / unmasked critics are the same architecture with identical init but different functions;
     fine-tuned arms that differ in the mask end up as different networks
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import pytest

from commenv import EnvConfig, CommEnv, StayPut, RandomPolicy, link_rates, gen_scenario, Obs
from expert import plan, replay_returns
from bc import replay_observations, augment
from models import Actor, Critic, NetPolicy
from mappo import PPOConfig, collect, ppo_update
from evalproto import returns, paired_diff

torch.set_num_threads(1)
CFG = EnvConfig().with_k(6.0)
ROOT = os.path.join(os.path.dirname(__file__), "..")


def _small_batch(masked, seed=0, n_envs=8):
    torch.manual_seed(seed)
    actor, critic = Actor(CFG), Critic(CFG, d=32)
    ppo = PPOConfig(n_envs=n_envs, epochs=2, minibatches=2)
    batch = collect(CFG, list(range(500, 500 + n_envs)), actor, critic, masked, ppo)
    return actor, critic, batch, ppo


def test_G0_planner_objective_equals_env_return():
    seeds = list(range(900, 906))
    out = plan(seeds, CFG, restarts=2, rounds=1, adam_iters=15)
    env_ret = replay_returns(seeds, CFG, out["actions"], out["channels"])
    assert np.abs(env_ret - out["J"]).max() < 1e-3


def test_G0b_bc_target_is_not_an_input_feature():
    seeds = list(range(900, 904))
    out = plan(seeds, CFG, restarts=1, rounds=1, adam_iters=10)
    o, a, c = replay_observations(seeds, CFG, out["actions"], out["channels"])
    cols = torch.cat([o.sf, o.uf.flatten(1), o.af.flatten(1)], 1)
    for j in range(2):
        tgt = a[:, j:j + 1]
        assert not torch.isclose(cols, tgt, atol=1e-6).all(0).any(), "a target coordinate is copied in the input"
        # nor a trivial linear copy: max |corr| with any single input column must be far from 1
        x = (cols - cols.mean(0)) / (cols.std(0) + 1e-9); y = (tgt - tgt.mean()) / (tgt.std() + 1e-9)
        assert (x * y).mean(0).abs().max() < 0.95


def test_G0c_dihedral_augmentation_is_a_symmetry():
    from bc import _D8
    sc = gen_scenario(7, CFG)
    users, uav, ch = torch.tensor(sc[0][1], dtype=torch.float32), torch.tensor(sc[2], dtype=torch.float32), torch.tensor(sc[3])
    base, _ = link_rates(uav, users, ch, CFG)
    for M in _D8:
        f = lambda p: (p - CFG.world / 2) @ M.T + CFG.world / 2
        r, _ = link_rates(f(uav), f(users), ch, CFG)
        assert torch.allclose(base, r, atol=1e-5)


def test_GM_masks_hide_entities_outside_k():
    torch.manual_seed(0)
    env = CommEnv(CFG, range(4)); o = env.reset().flat()
    actor, critic = Actor(CFG), Critic(CFG, d=32)
    # make the heads non-trivial so a leak would be visible
    for p in list(actor.parameters()) + list(critic.parameters()):
        p.data += 0.05 * torch.randn_like(p)
    o2 = Obs(*[x.clone() for x in o])
    hidden_u = ~o.um                                    # users outside radius k
    o2.uf[hidden_u] += 3.0
    hidden_a = (~o.am) & ~torch.eye(CFG.n_agents, dtype=torch.bool).repeat(4, 1)   # UAVs outside radius k
    o2.af[..., :2][hidden_a] += 3.0
    assert hidden_u.any() and hidden_a.any()
    with torch.no_grad():
        assert torch.allclose(actor(o)[0], actor(o2)[0], atol=1e-6) and torch.allclose(actor(o)[1], actor(o2)[1], atol=1e-6)
        assert torch.allclose(critic(o, True), critic(o2, True), atol=1e-6), "masked critic sees hidden entities"
        assert (critic(o, False) - critic(o2, False)).abs().max() > 1e-4, "unmasked critic ignores the far field"


def test_G1_every_head_gets_gradient():
    actor, critic, batch, ppo = _small_batch(masked=True)
    a_opt = torch.optim.Adam(actor.parameters(), 1e-3); c_opt = torch.optim.Adam(critic.parameters(), 1e-3)
    # recompute grads on one minibatch with the same loss the update uses
    o = batch["o"].index(torch.arange(256))
    lp, ent = actor.log_prob_entropy(o, batch["a"][:256], batch["c"][:256])
    adv = (batch["adv"][:256] - batch["adv"][:256].mean()) / (batch["adv"][:256].std() + 1e-8)
    ratio = (lp - batch["lp"][:256]).exp()
    (-(ratio * adv).mean() - 1e-3 * ent.mean()).backward()
    for name in ("pos_head", "ch_head", "enc"):
        g = sum(float(p.grad.norm()) for n, p in actor.named_parameters() if n.startswith(name) and p.grad is not None)
        assert g > 1e-9, f"G1 FAILED: no gradient in actor.{name}"
    assert actor.log_std.grad is not None and float(actor.log_std.grad.abs().sum()) > 1e-9, "G1 FAILED: log_std"
    ((critic(o, True) - batch["ret"][:256]) ** 2).mean().backward()
    assert sum(float(p.grad.norm()) for p in critic.parameters() if p.grad is not None) > 1e-9


def test_G1b_ratio_is_one_then_moves():
    actor, critic, batch, ppo = _small_batch(masked=True)
    a_opt = torch.optim.Adam(actor.parameters(), 3e-3); c_opt = torch.optim.Adam(critic.parameters(), 1e-3)
    st = ppo_update(actor, critic, True, batch, a_opt, c_opt, ppo)
    assert st["first_ratio_dev"] < 1e-5, "on-policy first-pass ratio must be exactly 1"
    assert st["ratio_max"] > 1 + 1e-5 or st["ratio_min"] < 1 - 1e-5, "ratio never moved: surrogate is constant"
    assert st["gnorm_actor"] > 1e-9


def test_G2_weights_move_after_updates():
    actor, critic, batch, ppo = _small_batch(masked=True)
    before = torch.cat([p.flatten() for p in actor.parameters()]).detach().clone()
    a_opt = torch.optim.Adam(actor.parameters(), 1e-3); c_opt = torch.optim.Adam(critic.parameters(), 1e-3)
    for _ in range(3):
        ppo_update(actor, critic, True, batch, a_opt, c_opt, ppo)
    after = torch.cat([p.flatten() for p in actor.parameters()]).detach()
    assert float((after - before).abs().max()) > 1e-6


def test_G4_masked_and_unmasked_critics_differ_only_by_mask():
    torch.manual_seed(5); c1 = Critic(CFG, d=32)
    torch.manual_seed(5); c2 = Critic(CFG, d=32)
    assert all(torch.equal(a, b) for a, b in zip(c1.state_dict().values(), c2.state_dict().values()))
    env = CommEnv(CFG, range(4)); o = env.reset().flat()
    with torch.no_grad():
        assert (c1(o, True) - c2(o, False)).abs().max() > 1e-6          # same weights, different function: the mask
    # arms that differ by the mask must not stay the same network after training
    arms = []
    for masked in (True, False):
        torch.manual_seed(3); actor = Actor(CFG)
        torch.manual_seed(1003); critic = Critic(CFG, d=32)
        torch.manual_seed(3)
        ppo = PPOConfig(n_envs=8, epochs=2, minibatches=2, critic_warmup=0)
        batch = collect(CFG, list(range(600, 608)), actor, critic, masked, ppo)
        a_opt = torch.optim.Adam(actor.parameters(), 3e-3); c_opt = torch.optim.Adam(critic.parameters(), 1e-3)
        for _ in range(3):
            ppo_update(actor, critic, masked, batch, a_opt, c_opt, ppo)
        arms.append(actor)
    d = max(float((p - q).abs().max()) for p, q in zip(arms[0].parameters(), arms[1].parameters()))
    assert d > 1e-9, "G4 FAILED: the two arms are the same network"


def test_madt_has_no_label_leakage():
    from madt import MADT, action_tokens
    torch.manual_seed(0)
    m = MADT(CFG, d=32, layers=2).eval()
    env = CommEnv(CFG, range(2)); o1 = env.reset()
    L = 5
    s = torch.randn(4, L, 32); a = torch.randn(4, L, 2 + CFG.n_channels); r = torch.randn(4, L)
    with torch.no_grad():
        p0, l0 = m.forward_emb(s, a, r)
        for j in range(L):
            a2 = a.clone(); a2[:, j] += 5.0                         # change the action taken at step j
            p1, l1 = m.forward_emb(s, a2, r)
            assert torch.allclose(p0[:, :j + 1], p1[:, :j + 1], atol=1e-6), "prediction at step <= j depends on a_j (leak)"
            assert torch.allclose(l0[:, :j + 1], l1[:, :j + 1], atol=1e-6)
            if j < L - 1:
                assert (p0[:, j + 1:] - p1[:, j + 1:]).abs().max() > 1e-6, "history of actions is ignored"


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "results", "bc_seed0.pt")), reason="needs trained BC student")
def test_G3_student_beats_trivial_floors():
    actor = Actor(CFG); actor.load_state_dict(torch.load(os.path.join(ROOT, "results", "bc_seed0.pt")))
    seeds = list(range(300000, 300128))
    r, _ = returns(NetPolicy(actor), seeds, CFG)
    for pol in (StayPut(), RandomPolicy(0)):
        f, _ = returns(pol, seeds, CFG)
        d = paired_diff(r, f)
        assert d["ci95"][0] > 0, f"G3 FAILED vs {pol.name}: paired delta {d['mean']:.1f} CI {d['ci95']}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q"]))
