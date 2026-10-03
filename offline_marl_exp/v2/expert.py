"""
expert.py -- centralised, full-information, future-aware trajectory planner (the "teacher").

Not a MINLP solver and it does not claim a global optimality certificate: the problem
(joint trajectories + channels under SINR coupling) is non-convex mixed-integer. What it is:

  * block-coordinate descent (BCD) over  (a) continuous trajectories given channels  -- Adam on the
    exact differentiable objective, speed constraint enforced by construction (ball projection),
    and (b) channels given trajectories -- exact coordinate-wise best response (the objective is
    separable over time given positions, so each (UAV, step) is an exhaustive C-way search);
  * R random restarts per scenario, best kept; objective trace logged per BCD round;
  * the planner objective is *identical* to the env return (checked by replay: test_gates G0);
  * an interference-free relaxation (I:=0, channels irrelevant) is solved the same way and reported
    as an UPPER-BOUND PROXY. It is itself a heuristic optimum, so (UB-J)/UB is a gap *estimate*, not a certificate.

`strong` = many restarts / rounds; `weak` = single start, one round (for the expert-quality ablation).
"""
from __future__ import annotations
import time
import numpy as np
import torch

from commenv import EnvConfig, CommEnv, team_reward


def _positions(u, uav0, cfg):
    d = u / u.norm(dim=-1, keepdim=True).clamp(min=1.0)                 # unit-ball projection
    p, out = uav0, []
    for t in range(u.shape[1]):
        p = (p + cfg.vmax * d[:, t]).clamp(0.0, cfg.world)
        out.append(p)
    return torch.stack(out, 1), d


def _objective(u, uav0, users, ch, cfg, interference_free=False):
    pos, d = _positions(u, uav0, cfg)
    r = team_reward(pos, users, ch, cfg, interference_free)             # [B,T]
    en = (d ** 2).sum(-1).mean(-1)                                      # [B,T] energy, same as env
    return (r - cfg.energy * en).sum(-1), d


@torch.no_grad()
def _best_response_channels(pos, users, ch, cfg, sweeps=2):
    C, N = cfg.n_channels, cfg.n_agents
    for _ in range(sweeps):
        for i in range(N):
            vals = []
            for c in range(C):
                ch2 = ch.clone(); ch2[..., i] = c
                vals.append(team_reward(pos, users, ch2, cfg))
            ch[..., i] = torch.stack(vals, -1).argmax(-1)
    return ch


def canonicalise_channels(ch, ch0, C):
    """
    Reward depends only on WHICH UAVs share a channel, never on the label. So at every step we may
    permute labels for free. Pick, step by step, the permutation that agrees most with the previous
    step's labels (starting from the env's initial channels). This removes pure labelling noise from the
    target (the student can see its own and its neighbours' current channels, so 'stay' is learnable) and
    leaves the objective bit-for-bit unchanged.
    """
    import itertools
    perms = torch.tensor(list(itertools.permutations(range(C))))            # [P,C]
    out = ch.clone()
    prev = ch0.clone()                                                       # [S,N]
    for t in range(ch.shape[1]):
        cur = ch[:, t]                                                       # [S,N]
        cand = perms[:, cur]                                                 # [P,S,N] relabelled
        agree = (cand == prev.unsqueeze(0)).sum(-1)                          # [P,S]
        best = agree.argmax(0)                                               # [S]
        cur = cand[best, torch.arange(cur.shape[0])]
        out[:, t] = cur; prev = cur
    return out


def heuristic_init(env: CommEnv, cfg: EnvConfig):
    """
    Deterministic warm start: k-means (N centroids, warm-started from the previous step) of the users at
    every step, UAVs matched to centroids by min-cost assignment on the current position, each UAV heads for
    its centroid. Gives plans a consistent, state-determined structure (nearest-centroid assignment) instead of an
    arbitrary random-restart local optimum, which is what makes the target learnable from local observations.
    """
    from scipy.optimize import linear_sum_assignment
    S, N, T = env.B, cfg.n_agents, cfg.T
    U = env.users_traj.numpy(); P0 = env.uav0.numpy()
    out = np.zeros((S, T, N, 2), dtype=np.float32)
    for s in range(S):
        p = P0[s].copy(); cen = None
        for t in range(1, T + 1):
            X = U[s, t]
            if cen is None:                                       # farthest-point init, deterministic
                cen = [X[0]]
                for _ in range(N - 1):
                    dd = np.min([((X - c) ** 2).sum(1) for c in cen], 0)
                    cen.append(X[dd.argmax()])
                cen = np.array(cen)
            for _ in range(6):
                a = ((X[:, None] - cen[None]) ** 2).sum(-1).argmin(1)
                for m in range(N):
                    if (a == m).any():
                        cen[m] = X[a == m].mean(0)
            cost = ((p[:, None] - cen[None]) ** 2).sum(-1)
            r, c = linear_sum_assignment(cost)
            d = cen[c] - p[r]
            d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9) * np.minimum(np.linalg.norm(d, axis=1, keepdims=True), cfg.vmax)
            out[s, t - 1, r] = d / cfg.vmax
            p[r] = np.clip(p[r] + d, 0, cfg.world)
    return torch.tensor(out)


def plan(seeds, cfg: EnvConfig, restarts=4, rounds=4, adam_iters=80, lr=0.08,
         interference_free=False, rng_seed=0, verbose=False):
    """returns dict(actions [S,T,N,2], channels [S,T,N], J [S], trace [rounds,S], solve_s)"""
    t0 = time.time()
    env = CommEnv(cfg, seeds)
    S, T, N, C = env.B, cfg.T, cfg.n_agents, cfg.n_channels
    R = restarts
    rep = lambda x: x.repeat_interleave(R, 0)
    users, uav0 = rep(env.users_traj[:, 1:]), rep(env.uav0)
    g = torch.Generator().manual_seed(rng_seed)
    u = (torch.randn(S * R, T, N, 2, generator=g) * 0.3)
    # restart 0 = deterministic k-means/assignment warm start; restart 1 = same + noise; rest random drift
    h = heuristic_init(env, cfg)
    u.view(S, R, T, N, 2)[:, 0] = h
    if R > 1:
        u.view(S, R, T, N, 2)[:, 1] = h + 0.15 * torch.randn_like(h)
    ch = torch.randint(0, C, (S * R, T, N), generator=g)
    ch.view(S, R, T, N)[:, 0] = (torch.arange(N) % C).view(1, 1, N)
    trace = []
    for rd in range(rounds):
        u = u.detach().requires_grad_(True)
        opt = torch.optim.Adam([u], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, adam_iters)
        with torch.no_grad():
            best_J, _ = _objective(u, uav0, users, ch, cfg, interference_free)
        best_u = u.detach().clone()
        for _ in range(adam_iters):
            opt.zero_grad()
            J, _ = _objective(u, uav0, users, ch, cfg, interference_free)
            with torch.no_grad():                      # keep the best iterate per problem -> monotone BCD
                better = J.detach() > best_J
                best_J = torch.where(better, J.detach(), best_J)
                best_u[better] = u.detach()[better]
            (-J.sum()).backward()
            opt.step(); sched.step()
        with torch.no_grad():
            J, _ = _objective(u, uav0, users, ch, cfg, interference_free)
            better = J > best_J
            best_u[better] = u.detach()[better]
        u = best_u
        if not interference_free:
            pos, _ = _positions(u, uav0, cfg)
            ch = _best_response_channels(pos, users, ch, cfg)
        with torch.no_grad():
            J, _ = _objective(u, uav0, users, ch, cfg, interference_free)
        trace.append(J.view(S, R).max(1).values.numpy().copy())
        if verbose:
            print(f"  round {rd}: mean best J = {trace[-1].mean():.3f}")
    with torch.no_grad():
        J, d = _objective(u, uav0, users, ch, cfg, interference_free)
    Jv = J.view(S, R)
    best = Jv.argmax(1)
    # prefer the deterministic warm start unless another restart beats it by >1%: keeps plans state-determined
    best = torch.where(Jv[:, 0] >= 0.99 * Jv.max(1).values, torch.zeros_like(best), best)
    pick = lambda x: x.view(S, R, *x.shape[1:])[torch.arange(S), best]
    ch_best = pick(ch)
    if not interference_free:
        ch_best = canonicalise_channels(ch_best, env.ch0, C)
    return dict(actions=pick(d).numpy(), channels=ch_best.numpy(), J=Jv[torch.arange(S), best].numpy(),
                J_all=J.view(S, R).numpy(), trace=np.stack(trace), solve_s=time.time() - t0)


def replay_returns(seeds, cfg, actions, channels):
    """Replay planned actions through the env: returns per-scenario return."""
    env = CommEnv(cfg, seeds)
    env.reset()
    ret = torch.zeros(env.B)
    for t in range(cfg.T):
        _, r, _, _ = env.step(torch.tensor(actions[:, t]), torch.tensor(channels[:, t]).long())
        ret += r
    return ret.numpy()


def generate_dataset(seeds, cfg, path, **plan_kw):
    out = plan(seeds, cfg, **plan_kw)
    env_ret = replay_returns(seeds, cfg, out["actions"], out["channels"])
    err = np.abs(env_ret - out["J"]).max()
    assert err < 1e-3, f"planner objective != env return (max abs diff {err:.2e})"
    torch.save(dict(seeds=list(map(int, seeds)), actions=out["actions"], channels=out["channels"],
                    expert_return=env_ret, solve_s=out["solve_s"], cfg=cfg.__dict__), path)
    return out, env_ret


if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--start", type=int, default=100000)
    ap.add_argument("--out", default="data/expert_train.pt")
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=100)
    a = ap.parse_args()
    cfg = EnvConfig()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    seeds = list(range(a.start, a.start + a.n))
    parts = []
    for i in range(0, a.n, a.chunk):
        o = plan(seeds[i:i + a.chunk], cfg, restarts=a.restarts, rounds=a.rounds)
        parts.append(o); print(f"chunk {i//a.chunk}: {o['solve_s']:.1f}s mean J={o['J'].mean():.2f}", flush=True)
    cat = lambda k: np.concatenate([p[k] for p in parts], 0)
    actions, channels = cat("actions"), cat("channels")
    env_ret = replay_returns(seeds, cfg, actions, channels)
    assert np.abs(env_ret - cat("J")).max() < 1e-3
    torch.save(dict(seeds=seeds, actions=actions, channels=channels, expert_return=env_ret,
                    solve_s=sum(p["solve_s"] for p in parts), cfg=cfg.__dict__), a.out)
    print("saved", a.out, "mean expert return", env_ret.mean())
