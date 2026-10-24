"""
Tests for the new BC-side code (bc2.py, planner_state.py, StudentActor, obs_extra). Run:  python -m pytest tests -x -q
Written without being executed by their author: if one fails, that is information about the NEW code, not about v2.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from dataclasses import replace

from commenv import EnvConfig, CommEnv, StayPut, link_rates
from models import StudentActor, Actor, NetPolicy, dims
from planner_state import plan_from_state
from expert import replay_returns
from bc2 import expert_rows, augment2, dagger_collect, train_student, Rows, make_student, student_opts
from mappo import PPOConfig, collect, ppo_update
from models import Critic

torch.set_num_threads(1)
SMALL = EnvConfig(n_agents=4, n_users=12, T=6).with_k(6.0)
SEEDS = [11, 12, 13]


def test_default_cfg_unchanged():
    env = CommEnv(EnvConfig().with_k(6.0), [1, 2]); o = env.reset()
    assert o.sf.shape[-1] == 6 and o.uf.shape[-1] == 4 and dims(EnvConfig()) == (6, 4, 5)


def test_obs_extra_shapes_and_prev_action():
    cfg = replace(SMALL, obs_extra=True)
    env = CommEnv(cfg, SEEDS); o = env.reset()
    assert o.sf.shape[-1] == dims(cfg)[0] and o.uf.shape[-1] == dims(cfg)[1] == 6
    assert torch.all(o.sf[..., -2:] == 0)                                   # no previous action at t=0
    d = torch.randn(env.B, cfg.n_agents, 2)
    o2, _, _, _ = env.step(d, env.ch)
    expect = d / d.norm(dim=-1, keepdim=True).clamp(min=1.0)
    assert torch.allclose(o2.sf[..., -2:], expect, atol=1e-6)
    # extrapolated rel. position = rel_u + 4*vel
    t = env.t
    rel = env.users_traj[:, t].unsqueeze(1) - env.uav.unsqueeze(2)
    ex = (rel + 4.0 * env.user_vel[:, t].unsqueeze(1)) / (cfg.world / 2)
    assert torch.allclose(o2.uf[..., 4:6], ex, atol=1e-5)


def test_plan_from_state_matches_env_replay_midepisode():
    cfg = SMALL
    env = CommEnv(cfg, SEEDS); env.reset()
    for _ in range(2):
        env.step(torch.zeros(env.B, cfg.n_agents, 2), env.ch)
    out = plan_from_state(env.users_traj[:, env.t + 1:], env.uav, env.ch, cfg, restarts=2, rounds=1, adam_iters=10)
    h = cfg.T - env.t
    assert out["actions"].shape == (env.B, h, cfg.n_agents, 2)
    ret = torch.zeros(env.B)
    for j in range(h):
        _, r, _, _ = env.step(out["actions"][:, j], out["channels"][:, j])
        ret += r
    assert np.abs(ret.numpy() - out["J"]).max() < 1e-3


def test_student_actor_mdn_chunk_grads_and_ppo_compat():
    cfg = SMALL
    st = StudentActor(cfg, d=32, mdn_k=3, chunk=3)
    env = CommEnv(cfg, SEEDS); o = env.reset().flat()
    n = o.sf.shape[0]
    a = torch.randn(n, 3, 2) * 0.3; m = torch.ones(n, 3, dtype=torch.bool); c = torch.randint(0, cfg.n_channels, (n,))
    loss, stats = st.bc_loss(o, a, m, c)
    loss.backward()
    for nme in ["mdn", "chunk_head", "ch_head"]:
        g = getattr(st, nme).weight.grad
        assert g is not None and g.abs().sum() > 0, nme
    mu, lg = st(o)
    assert mu.shape == (n, 2) and lg.shape == (n, cfg.n_channels)
    # Gaussian PPO code path works unchanged
    lp, ent = st.log_prob_entropy(o, mu.detach(), c)
    assert lp.shape == (n,) and torch.isfinite(lp).all()
    ppo = PPOConfig(n_envs=4, iters=1, epochs=1, minibatches=1)
    critic = Critic(cfg, d=16)
    batch = collect(cfg, [1, 2, 3, 4], st, critic, True, ppo)
    ppo_update(st, critic, True, batch, torch.optim.Adam(st.parameters(), 1e-3), torch.optim.Adam(critic.parameters(), 1e-3), ppo)


def test_expert_rows_alignment():
    cfg = SMALL
    S, N, T, H = len(SEEDS), cfg.n_agents, cfg.T, 3
    rng = np.random.default_rng(0)
    A = rng.normal(size=(S, T, N, 2)).astype(np.float32) * 0.3
    C = rng.integers(0, cfg.n_channels, (S, T, N))
    rows = expert_rows(SEEDS, cfg, A, C, H=H)
    assert len(rows) == S * N * T
    for t, b, n in [(0, 0, 0), (2, 1, 3), (T - 1, 2, 1), (T - 2, 0, 2)]:
        r = t * S * N + b * N + n
        assert torch.allclose(rows.a[r, 0], torch.tensor(A[b, t, n]))
        assert int(rows.c[r]) == C[b, t, n]
        for j in range(1, H):
            if t + j < T:
                assert rows.m[r, j] and torch.allclose(rows.a[r, j], torch.tensor(A[b, t + j, n]))
            else:
                assert not rows.m[r, j]


def test_augment2_symmetry_extra():
    cfg = replace(SMALL, obs_extra=True)
    env = CommEnv(cfg, SEEDS); env.reset()
    env.step(torch.randn(env.B, cfg.n_agents, 2), env.ch)
    o = env.obs(cfg.k).flat(); n = o.sf.shape[0]
    a = torch.randn(n, 2, 2)
    g = torch.Generator().manual_seed(3)
    o2, a2 = augment2(o, a, g, True)
    assert torch.allclose(a.norm(dim=-1), a2.norm(dim=-1), atol=1e-5)                     # isometry
    assert torch.allclose(o.uf[..., 4:6].norm(dim=-1), o2.uf[..., 4:6].norm(dim=-1), atol=1e-5)
    assert torch.allclose(o.sf[:, -2:].norm(dim=-1), o2.sf[:, -2:].norm(dim=-1), atol=1e-5)
    assert torch.equal(o.um, o2.um) and torch.equal(o.am, o2.am)                          # masks (distance-based) invariant


def test_train_and_dagger_smoke():
    cfg = replace(SMALL, obs_extra=True)
    opts = dict(k=6.0, d=16, mdn_k=2, chunk=2, obs_extra=True)
    model, ema = make_student(opts), make_student(opts)
    ema.load_state_dict(model.state_dict())
    rng = np.random.default_rng(1)
    S, N, T = len(SEEDS), cfg.n_agents, cfg.T
    A = rng.normal(size=(S, T, N, 2)).astype(np.float32) * 0.3; C = rng.integers(0, cfg.n_channels, (S, T, N))
    rows = expert_rows(SEEDS, cfg, A, C, H=2)
    train_student(model, ema, rows, rows, 1, torch.device("cpu"), bs=64, extra=True, log=lambda *_: None)
    new, plan_s, ret = dagger_collect(ema.cpu().eval(), [21, 22], cfg, cfg, 0.5,
                                      dict(restarts=2, rounds=1, adam_iters=5), H=2)
    assert len(new) == 2 * N * T and new.a.shape[1:] == (2, 2) and np.isfinite(ret)
    assert not new.m[:, 1][-2 * N:].any()                      # last step has no step t+1 in the plan
    both = Rows.cat([rows, new]); assert len(both) == len(rows) + len(new)


def test_split_radius_masks():
    cfg = replace(SMALL, k=4.0, k_a=1000.0)
    env = CommEnv(cfg, SEEDS); o = env.reset()
    N = cfg.n_agents
    d_u = (env.users_traj[:, 0].unsqueeze(1) - env.uav.unsqueeze(2)).norm(dim=-1)
    assert torch.equal(o.um, d_u <= 4.0)                                             # users: radius k
    assert torch.equal(o.am, ~torch.eye(N, dtype=torch.bool).expand(env.B, N, N))    # UAVs: everyone but self
    o2 = CommEnv(replace(SMALL, k=4.0), SEEDS).reset()                               # k_a unset => same radius k
    d_a = (env.uav.unsqueeze(1) - env.uav.unsqueeze(2)).norm(dim=-1)
    assert torch.equal(o2.am, (d_a <= 4.0) & ~torch.eye(N, dtype=torch.bool))


def test_bcast_features_and_dims():
    N, C = SMALL.n_agents, SMALL.n_channels
    for bc, extra in [(1, False), (2, False), (2, True)]:
        cfg = replace(SMALL, bcast=bc, obs_extra=extra)
        env = CommEnv(cfg, SEEDS); o = env.reset()
        assert o.sf.shape[-1] == dims(cfg)[0]
        b0 = 2 + C + 1                                                   # first broadcast column
        cnt = o.sf[..., b0:b0 + C]
        assert torch.allclose(cnt.sum(-1), torch.full((env.B, N), (N - 1) / N))      # all other UAVs, once each
        own = torch.nn.functional.one_hot(env.ch, C).float()
        tot = torch.nn.functional.one_hot(env.ch, C).float().sum(1, keepdim=True)    # [B,1,C]
        assert torch.allclose(cnt * N + own, tot.expand(-1, N, -1))                  # counts + self == channel totals
        if bc >= 2:
            rings = o.sf[..., b0 + C:b0 + C + 4]
            assert torch.allclose(rings.sum(-1), torch.ones(env.B, N), atol=1e-5)
        if extra:
            assert torch.all(o.sf[..., -2:] == 0)                        # prev displacement still the last two columns


def test_bcast_augment_invariance():
    cfg = replace(SMALL, bcast=2, obs_extra=True)
    env = CommEnv(cfg, SEEDS); env.reset()
    env.step(torch.randn(env.B, cfg.n_agents, 2), env.ch)
    o = env.obs(cfg.k).flat(); a = torch.randn(o.sf.shape[0], 2, 2)
    o2, _ = augment2(o, a, torch.Generator().manual_seed(5), True)
    b0, C = 2 + cfg.n_channels + 1, cfg.n_channels
    assert torch.equal(o.sf[:, b0:b0 + C + 4], o2.sf[:, b0:b0 + C + 4])              # broadcast columns untouched


def test_sensed_dims_and_equal_width_with_bcast1():
    """bcast 1 and 3/4/5 must be the same width (C), so the broadcast-vs-sensed comparison is equal-dimension."""
    for b in (1, 3, 4, 5):
        cfg = replace(SMALL, bcast=b)
        env = CommEnv(cfg, SEEDS); o = env.reset()
        assert o.sf.shape[-1] == dims(cfg)[0], (b, o.sf.shape, dims(cfg))
        assert dims(cfg)[0] == dims(replace(SMALL, bcast=1))[0], b
    assert dims(replace(SMALL, bcast=2))[0] == dims(replace(SMALL, bcast=1))[0] + 4


def test_sensed_matches_the_interference_in_link_rates():
    """bcast=3 must be exactly log1p(co-channel power / noise)/SENSE_SCALE for the UAV's own position."""
    from commenv import SENSE_SCALE
    cfg = replace(SMALL, bcast=3)
    env = CommEnv(cfg, SEEDS); o = env.reset()
    N, C = cfg.n_agents, cfg.n_channels
    feat = o.sf.view(len(SEEDS), N, -1)[..., 2 + C + 1:2 + C + 1 + C]
    d2 = ((env.uav.unsqueeze(2) - env.uav.unsqueeze(1)) ** 2).sum(-1)
    g = ((d2 + cfg.H ** 2) ** (-cfg.alpha / 2.0)).masked_fill(torch.eye(N, dtype=torch.bool), 0.0)
    oh = torch.nn.functional.one_hot(env.ch, C).float()
    want = torch.log1p(torch.einsum("bij,bjc->bic", g, oh) / cfg.noise) / SENSE_SCALE
    assert torch.allclose(feat, want, atol=1e-5)
    own = feat.gather(-1, env.ch.unsqueeze(-1)).squeeze(-1)
    assert (own >= 0).all() and torch.isfinite(feat).all()


def test_sensed_k_is_a_subset_and_placebo_keeps_marginals():
    cfg3, cfg4, cfg5 = (replace(SMALL, bcast=b, k_a=3.0) for b in (3, 4, 5))
    N, C = SMALL.n_agents, SMALL.n_channels
    cut = slice(2 + C + 1, 2 + C + 1 + C)
    f = lambda c: CommEnv(c, SEEDS).reset().sf.view(len(SEEDS), N, -1)[..., cut]
    f3, f4, f5 = f(cfg3), f(cfg4), f(cfg5)
    assert (f4 <= f3 + 1e-6).all(), "radius-restricted sensing cannot exceed sensing over all UAVs"
    assert f4.sum() < f3.sum(), "with k_a=3 the restricted feature must drop some interference"
    assert torch.allclose(f5.sort(-1).values, f3.sort(-1).values, atol=1e-6), "placebo must keep the per-agent multiset"
    assert not torch.allclose(f5, f3), "placebo must not be the identity permutation everywhere"


def test_sensed_is_rotation_invariant_so_augmentation_still_applies():
    """The feature depends only on inter-UAV distances, so the D8 augmentation stays valid."""
    cfg = replace(SMALL, bcast=3)
    N, C = cfg.n_agents, cfg.n_channels
    cut = slice(2 + C + 1, 2 + C + 1 + C)
    env = CommEnv(cfg, SEEDS); o = env.reset()
    base = o.sf.view(len(SEEDS), N, -1)[..., cut].clone()
    env2 = CommEnv(cfg, SEEDS); env2.reset()
    W = cfg.world
    env2.uav = torch.stack([W - env2.uav[..., 1], env2.uav[..., 0]], -1)              # rotate the whole swarm 90 deg
    rot = env2.obs(cfg.k).sf.view(len(SEEDS), N, -1)[..., cut]
    assert torch.allclose(base, rot, atol=1e-5)


def test_bcast6_is_count_plus_sensed():
    """bcast=6 must be 2C wide and carry exactly the bcast=1 block followed by the bcast=3 block."""
    N, C = SMALL.n_agents, SMALL.n_channels
    cfg1, cfg3, cfg6 = (replace(SMALL, bcast=b) for b in (1, 3, 6))
    assert dims(cfg6)[0] == dims(cfg1)[0] + C
    f = lambda c: CommEnv(c, SEEDS).reset().sf.view(len(SEEDS), N, -1)
    s1, s3, s6 = f(cfg1), f(cfg3), f(cfg6)
    base = 2 + C + 1
    assert torch.allclose(s6[..., base:base + C], s1[..., base:base + C], atol=1e-6)      # the count block
    assert torch.allclose(s6[..., base + C:base + 2 * C], s3[..., base:base + C], atol=1e-6)  # the sensed block
    assert s6.shape[-1] == dims(cfg6)[0]


def test_shortfall_is_graded_and_outage_threshold_is_fixed():
    """info['shortfall'] must respond to cfg.rate_floor; info['outage'] must NOT (it is the reported statistic)."""
    import torch as _t
    base = replace(SMALL, rate_floor=1.0)
    hi = replace(SMALL, rate_floor=4.0)
    o_b, o_h = [], []
    for c in (base, hi):
        env = CommEnv(c, SEEDS); env.reset()
        _, _, _, info = env.step(_t.zeros(len(SEEDS), c.n_agents, 2), env.ch)
        o_b.append(info)
    i1, i2 = o_b
    assert _t.allclose(i1["outage"], i2["outage"]), "outage must not move with rate_floor"
    assert (i2["shortfall"] >= i1["shortfall"]).all() and i2["shortfall"].sum() > i1["shortfall"].sum()
    assert (i1["shortfall"] >= 0).all()


def test_coverage_penalty_only_changes_the_training_reward():
    """PPOConfig.outage_coef must lower the penalised reward used by PPO while ep_return stays the true objective."""
    import torch as _t
    from mappo import PPOConfig, collect
    from models import Critic
    cfg = replace(SMALL, rate_floor=2.0)
    seeds = [41, 42]
    mk = lambda: (_t.manual_seed(0), make_student(dict(k=cfg.k, d=32, mdn_k=0, chunk=1, obs_extra=False,
                                                       k_a=cfg.k_a, bcast=cfg.bcast)))[1]
    out = {}
    for coef in (0.0, 5.0):
        _t.manual_seed(0); actor = mk()
        _t.manual_seed(1); critic = Critic(cfg, d=16)
        _t.manual_seed(2)
        b = collect(cfg, seeds, actor, critic, True, PPOConfig(outage_coef=coef, shortfall_coef=coef))
        out[coef] = b
    assert abs(out[0.0]["ep_return"] - out[5.0]["ep_return"]) < 1e-5, "ep_return must be the UNPENALISED objective"
    assert out[0.0]["ep_penalty"] == 0.0 and out[5.0]["ep_penalty"] > 0.0
