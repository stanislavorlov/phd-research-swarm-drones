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
