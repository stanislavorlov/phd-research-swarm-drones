"""
screen_regime.py -- training-free screen: is an environment-size or alpha sweep worth running?

    python3 screen_regime.py                                  # alpha 2.0/2.5/3.5 at the paper's size
    python3 screen_regime.py --sizes 8,3,20 16,6,28 12,3,24.5 # N,C,W triples at alpha 2.5
    python3 screen_regime.py --headroom --n 64                # add the joint/unilateral channel headroom (slow)

Measures the OPPORTUNITY for an information gap, with no student and no planner:

  1. hidden interference fraction -- of the co-channel interference a user actually suffers, what share comes
     from peers its serving UAV cannot see?  This is "informationally global" made quantitative.  If it does
     not move across alpha, an alpha sweep will return a null.

  2. best-response flip rate -- how often does a UAV choosing its own channel pick differently when it can only
     account for interference from peers inside its radius?  Closer to the decision the policy makes.

  3. (--headroom) joint vs unilateral channel headroom at these states, the same decomposition as Section 4.1.

CAVEAT, and it is the whole point: these bound what is AVAILABLE to be lost, not what a trained policy captures.
This project has already seen a probe-based prediction fail (greedy best-response accuracy said sensing would
beat the broadcast count; the trained students said otherwise).  Use this to rule an experiment OUT, or to
predict a direction -- never as a result.

NOTE: written without being executed by its author.
"""
from __future__ import annotations
import argparse, itertools, math
from dataclasses import replace
import numpy as np

from commenv import EnvConfig, gen_scenario


def lloyd_positions(users, n, iters=8):
    """Deterministic farthest-point init + Lloyd -- the planner's warm start, with no planner."""
    cen = [users[0]]
    for _ in range(n - 1):
        d = np.min([((users - c) ** 2).sum(1) for c in cen], 0)
        cen.append(users[d.argmax()])
    cen = np.array(cen)
    for _ in range(iters):
        a = ((users[:, None] - cen[None]) ** 2).sum(-1).argmin(1)
        for m in range(n):
            if (a == m).any():
                cen[m] = users[a == m].mean(0)
    return cen


def gains(users, uav, cfg):
    d2 = ((users[:, None, :] - uav[None, :, :]) ** 2).sum(-1)           # [U,N]
    return (d2 + cfg.H ** 2) ** (-cfg.alpha / 2.0)


def metrics_one_state(users, uav, ch, cfg, ka, tol=1e-9):
    N, C = cfg.n_agents, cfg.n_channels
    g = gains(users, uav, cfg)
    serv = g.argmax(1)                                                  # [U]
    S = g[np.arange(len(users)), serv]
    pd = np.sqrt(((uav[:, None, :] - uav[None, :, :]) ** 2).sum(-1))    # [N,N]
    vis = (pd <= ka) & ~np.eye(N, dtype=bool)                           # [N,N]

    # --- 1. hidden interference fraction, per user
    same = (ch[None, :] == ch[serv][:, None])                           # [U,N]
    I_all = (g * same).sum(1) - S
    vis_u = vis[serv]                                                   # [U,N] peers the SERVING uav can see
    I_vis = (g * same * vis_u).sum(1)
    ok = I_all > tol
    hidden = float(np.mean(1.0 - I_vis[ok] / I_all[ok])) if ok.any() else float("nan")

    # --- 2. best-response flip rate, over UAVs that serve at least one user
    flips, n_srv = 0, 0
    for i in range(N):
        mine = serv == i
        if not mine.any():
            continue
        n_srv += 1
        gi, Si = g[mine], S[mine]
        J_full, J_vis = np.empty(C), np.empty(C)
        for c in range(C):
            co = (ch == c)
            co_other = co.copy(); co_other[i] = False
            I_f = (gi * co_other[None, :]).sum(1)
            I_v = (gi * (co_other & vis[i])[None, :]).sum(1)
            J_full[c] = np.mean(np.log2(1.0 + Si / (cfg.noise + I_f)))
            J_vis[c] = np.mean(np.log2(1.0 + Si / (cfg.noise + I_v)))
        best_f = set(np.flatnonzero(J_full >= J_full.max() - 1e-9))
        if int(J_vis.argmax()) not in best_f:
            flips += 1
    flip = flips / n_srv if n_srv else float("nan")
    return hidden, flip, (g, serv, S, same)


def _nanmean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if x.size and not np.all(np.isnan(x)) else float("nan")


def _team_all_c(g, serv, S, ch, cfg, i):
    """Team rate for every choice of agent i's channel, vectorised. Serving assignment does not depend on ch."""
    C = cfg.n_channels
    chm = np.tile(ch, (C, 1))
    chm[:, i] = np.arange(C)                                            # [C,N]
    same = chm[:, None, :] == chm[np.arange(C)[:, None], serv][:, :, None]
    I = (g[None] * same).sum(-1) - S[None]
    return np.log2(1.0 + S[None] / (cfg.noise + I)).mean(-1)            # [C]


def ibr_joint(g, serv, S, ch, cfg, starts=6, sweeps=3, rng=None):
    """Multi-start iterated best response: a LOWER BOUND on the joint optimum, tractable at any N.
    Start 0 is the current assignment, so the bound is never worse than doing nothing."""
    rng = rng or np.random.default_rng(0)
    N, C = cfg.n_agents, cfg.n_channels
    best = -np.inf
    for s_ in range(starts):
        cur = ch.copy() if s_ == 0 else rng.integers(0, C, N)
        for _ in range(sweeps):
            changed = False
            for i in range(N):
                J = _team_all_c(g, serv, S, cur, cfg, i)
                c = int(J.argmax())
                if c != cur[i]:
                    cur[i] = c; changed = True
            if not changed:
                break
        best = max(best, float(_team_all_c(g, serv, S, cur, cfg, 0).max()))
    return best


def headroom_one_state(g, serv, S, ch, cfg, cap=500_000, ibr_starts=6, ibr_sweeps=3, rng=None):
    """Joint and unilateral channel headroom at one state (Section 4.1's decomposition)."""
    N, C = cfg.n_agents, cfg.n_channels
    def team(chv):
        same = (chv[None, :] == chv[serv][:, None])
        I = (g * same).sum(1) - S
        return float(np.mean(np.log2(1.0 + S / (cfg.noise + I))))
    now = team(ch)
    uni = now
    for i in range(N):
        for c in range(C):
            c2 = ch.copy(); c2[i] = c
            uni = max(uni, team(c2))
    total = C ** N
    opt = float("nan")
    if total <= cap:
        opt = max(team(np.array(a)) for a in itertools.product(range(C), repeat=N))
    opt_ibr = ibr_joint(g, serv, S, ch, cfg, ibr_starts, ibr_sweeps, rng)
    return now, uni, opt, opt_ibr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="*", default=[2.0, 2.5, 3.5])
    ap.add_argument("--sizes", nargs="*", default=["8,3,20"], help="N,C,W triples; users scale as 4N")
    ap.add_argument("--k", type=float, default=6.0, help="observation radius (peers)")
    ap.add_argument("--n", type=int, default=256, help="scenarios")
    ap.add_argument("--seed0", type=int, default=800_000)
    ap.add_argument("--steps", type=int, nargs="*", default=[0, 5, 10, 15, 20])
    ap.add_argument("--placement", nargs="*", default=["stay", "lloyd"])
    ap.add_argument("--headroom", action="store_true")
    ap.add_argument("--ibr_starts", type=int, default=6)
    ap.add_argument("--ibr_sweeps", type=int, default=3)
    a = ap.parse_args()

    print(f"{a.n} scenarios x {len(a.steps)} steps, k = {a.k}\n")
    hdr = f"{'size (N,C,W)':14s} {'alpha':>6s} {'place':6s} {'hidden I':>9s} {'flip rate':>10s}"
    if a.headroom:
        hdr += (f" {'uni':>7s} {'joint-ex':>8s} {'joint-ibr':>8s} {'coord-ex':>8s} {'coord-ibr':>8s} "
                f"{'ibr/ex':>7s}")
    print(hdr); print("-" * len(hdr))

    for size in a.sizes:
        N, C, W = size.split(",")
        N, C, W = int(N), int(C), float(W)
        for alpha in a.alphas:
            cfg = replace(EnvConfig(), n_agents=N, n_channels=C, world=W, n_users=4 * N, alpha=alpha)
            for place in a.placement:
                H, F, U_, J_, O_, Ji, Oi = [], [], [], [], [], [], []
                for s in range(a.seed0, a.seed0 + a.n):
                    traj, _, uav0, ch0 = gen_scenario(s, cfg)
                    for t in a.steps:
                        if t >= traj.shape[0]:
                            continue
                        users = traj[t]
                        uav = lloyd_positions(users, N) if place == "lloyd" else uav0
                        h, f, (g, serv, S, _) = metrics_one_state(users, uav, ch0, cfg, a.k)
                        H.append(h); F.append(f)
                        if a.headroom:
                            now, uni, opt, opt_i = headroom_one_state(
                                g, serv, S, ch0, cfg, ibr_starts=a.ibr_starts,
                                ibr_sweeps=a.ibr_sweeps, rng=np.random.default_rng(s))
                            U_.append(uni - now); J_.append(opt - now); O_.append(opt - uni)
                            Ji.append(opt_i - now); Oi.append(opt_i - uni)
                row = (f"{size:14s} {alpha:6.2f} {place:6s} {_nanmean(H):9.3f} {_nanmean(F):10.3f}")
                if a.headroom:
                    jx, ji = _nanmean(J_), _nanmean(Ji)
                    ratio = ji / jx if (jx == jx and jx > 0) else float("nan")
                    row += (f" {_nanmean(U_):7.3f} {jx:8.3f} {ji:8.3f} "
                            f"{_nanmean(O_):8.3f} {_nanmean(Oi):8.3f} {ratio:7.3f}")
                print(row)

    print("\nHow to read this:")
    print("  hidden I   fraction of a user's co-channel interference coming from peers its server cannot see.")
    print("             The paper's mechanism requires this to be large; if it is flat across alpha, an alpha")
    print("             sweep has nothing to find.")
    print("  flip rate  fraction of serving UAVs whose own best channel changes when they can only account for")
    print("             visible interferers. A direct proxy for what the radius limit costs per decision.")
    if a.headroom:
        print("  uni/joint  per-STATE headroom (not per-episode: multiply by T for the Section 4.1 scale).")
        print("  joint-ex   exact max over all C^N assignments; nan when that exceeds the 500k cap.")
        print("  joint-ibr  multi-start iterated best response: a LOWER BOUND, tractable at any N.")
        print("  ibr/ex     validation. Where both exist this should be ~1.0; if it is, the ibr columns")
        print("             can be trusted at the sizes where the exact enumeration is out of reach.")
    print("\n  These bound what is AVAILABLE to be lost, not what a trained policy captures. Use to rule an")
    print("  experiment out or to predict a direction -- not as a result.")


if __name__ == "__main__":
    main()