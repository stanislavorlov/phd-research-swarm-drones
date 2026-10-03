"""
planner_state.py -- the expert planner, restartable from ANY state (receding-horizon / MPC form).

expert.plan() always starts from env.reset() and plans the whole episode. DAgger needs the expert's answer at the
states the STUDENT visits, so this version takes the current UAV positions, current channels and the remaining
user trajectory as arrays. Same algorithm as expert.py (BCD: Adam on trajectories <-> exact channel best response,
restarts, deterministic k-means warm start preferred unless beaten by >1 %), same objective code (it imports it).

Two differences, both deliberate:
  * restart 0 starts from the CURRENT channels, so "keep the channel" is the default and labels are persistent;
  * `cfg` is the PLANNING config. Pass replace(env_cfg, energy=lam) to plan with a different energy price than the
    environment charges (the "smoother expert" experiment). The env return is always computed by the env.

Works on CPU or GPU (`device`). The Python loop over horizon steps makes it launch-bound on GPU, so give it a big
batch (many scenarios x restarts) rather than many small calls.
"""
from __future__ import annotations
import time
import numpy as np
import torch

from commenv import EnvConfig
from expert import _positions, _objective, _best_response_channels, canonicalise_channels


def heuristic_init_arrays(users, uav0, cfg: EnvConfig):
    """users [S,h,U,2] = user positions at steps 1..h ; uav0 [S,N,2]  ->  unit-ball displacements [S,h,N,2]."""
    from scipy.optimize import linear_sum_assignment
    U = users.detach().cpu().numpy(); P0 = uav0.detach().cpu().numpy()
    S, h, N = U.shape[0], U.shape[1], cfg.n_agents
    out = np.zeros((S, h, N, 2), dtype=np.float32)
    for s in range(S):
        p = P0[s].copy(); cen = None
        for t in range(h):
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
            nrm = np.linalg.norm(d, axis=1, keepdims=True)
            d = d / np.maximum(nrm, 1e-9) * np.minimum(nrm, cfg.vmax)
            out[s, t, r] = d / cfg.vmax
            p[r] = np.clip(p[r] + d, 0, cfg.world)
    return torch.tensor(out)


def plan_from_state(users, uav0, ch0, cfg: EnvConfig, restarts=4, rounds=4, adam_iters=80, lr=0.08,
                    rng_seed=0, device="cpu"):
    """
    users [S,h,U,2]  user positions at steps 1..h (h = remaining horizon)     uav0 [S,N,2]    ch0 [S,N] long
    returns dict(actions [S,h,N,2] (unit-ball), channels [S,h,N] (canonicalised vs ch0), J [S], solve_s)
    J is the planner objective for the remaining horizon (== env return of replaying the plan, when cfg is the env cfg).
    """
    t0 = time.time()
    dev = torch.device(device)
    users, uav0, ch0 = users.detach().cpu().float(), uav0.detach().cpu().float(), ch0.detach().cpu().long()
    S, h = users.shape[0], users.shape[1]
    N, C, R = cfg.n_agents, cfg.n_channels, restarts
    g = torch.Generator().manual_seed(rng_seed)
    heur = heuristic_init_arrays(users, uav0, cfg)
    u = torch.randn(S * R, h, N, 2, generator=g) * 0.3
    u.view(S, R, h, N, 2)[:, 0] = heur
    if R > 1:
        u.view(S, R, h, N, 2)[:, 1] = heur + 0.15 * torch.randn(heur.shape, generator=g)
    ch = torch.randint(0, C, (S * R, h, N), generator=g)
    ch.view(S, R, h, N)[:, 0] = ch0.unsqueeze(1).expand(S, h, N)
    rep = lambda x: x.repeat_interleave(R, 0).to(dev)
    users_d, uav0_d = rep(users), rep(uav0)
    u, ch = u.to(dev), ch.to(dev)
    for _ in range(rounds):
        if adam_iters > 0:
            u = u.detach().requires_grad_(True)
            opt = torch.optim.Adam([u], lr=lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, adam_iters)
            with torch.no_grad():
                best_J, _ = _objective(u, uav0_d, users_d, ch, cfg)
            best_u = u.detach().clone()
            for _ in range(adam_iters):
                opt.zero_grad()
                J, _ = _objective(u, uav0_d, users_d, ch, cfg)
                with torch.no_grad():                         # keep the best iterate per problem -> monotone BCD
                    better = J.detach() > best_J
                    best_J = torch.where(better, J.detach(), best_J)
                    best_u[better] = u.detach()[better]
                (-J.sum()).backward()
                opt.step(); sched.step()
            with torch.no_grad():
                J, _ = _objective(u, uav0_d, users_d, ch, cfg)
                better = J > best_J
                best_u[better] = u.detach()[better]
            u = best_u
        pos, _ = _positions(u, uav0_d, cfg)
        ch = _best_response_channels(pos, users_d, ch, cfg)
    with torch.no_grad():
        J, d = _objective(u, uav0_d, users_d, ch, cfg)
    Jv = J.view(S, R)
    best = Jv.argmax(1)
    best = torch.where(Jv[:, 0] >= 0.99 * Jv.max(1).values, torch.zeros_like(best), best)   # prefer warm start
    idx = torch.arange(S, device=Jv.device)
    pick = lambda x: x.view(S, R, *x.shape[1:])[idx, best]
    ch_best = canonicalise_channels(pick(ch).cpu(), ch0, C)
    return dict(actions=pick(d).cpu(), channels=ch_best, J=Jv[idx, best].cpu().numpy(), solve_s=time.time() - t0)
