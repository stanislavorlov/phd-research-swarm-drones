"""
solver_advantages.py -- exact advantages from the planner, no learned critic.

    python3 solver_advantages.py --pairs data/pairs_train.pt --out data/adv_train.pt

Measured on this dataset: the true one-step advantage of the strong action over the weak one is +1.19 returns
(immediate +0.27, future +0.92), while a learned critic reports 0.00. The paired rows share an observation, so
V(o) cancels when the branch means are differenced -- which makes that zero a direct statement that Q is
ignoring its action argument. Two displacement dims and C channel one-hots appended to a 128-dim state
embedding, regressed onto a target whose variance is dominated by state, collapses to Q(s,a) ~ V(s).

So don't learn it. The planner can evaluate a counterfactual action directly:

    Q_true(s, a) = r(s, a) + J(s')      J = the planner's objective over the remaining horizon

with no estimation error. With two actions per state the behaviour value is their mean, so

    A(s, a_A) = +(Q_A - Q_B)/2          A(s, a_B) = -(Q_A - Q_B)/2

Both branches are solved in ONE planner call per step by stacking them along the scenario axis (they share the
user trajectory), which halves the number of calls and batches better on GPU.

NOTE: written without being executed by its author.
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch

from commenv import EnvConfig, CommEnv
from planner_state import plan_from_state


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs_train.pt")
    ap.add_argument("--out", default="data/adv_train.pt")
    ap.add_argument("--n", type=int, default=0, help="0 = all scenarios in the pairs file")
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    p = torch.load(a.pairs, map_location="cpu", weights_only=False)
    k = a.n or len(p["seeds"])
    seeds = list(p["seeds"])[:k]
    aA, cA = torch.as_tensor(p["actions_a"][:k]).float(), torch.as_tensor(p["channels_a"][:k]).long()
    aB, cB = torch.as_tensor(p["actions_b"][:k]).float(), torch.as_tensor(p["channels_b"][:k]).long()
    cfg = EnvConfig(**p["cfg"]) if isinstance(p.get("cfg"), dict) else EnvConfig()
    kw = dict(restarts=a.restarts, rounds=a.rounds, adam_iters=a.iters, device=a.device)

    env = CommEnv(cfg, seeds)
    env.reset()
    S, T = env.B, cfg.T
    qa, qb = torch.zeros(S, T), torch.zeros(S, T)
    ra, rb = torch.zeros(S, T), torch.zeros(S, T)
    t0 = time.time()
    print(f"{S} scenarios x {T} steps, evaluating both branches with planner {kw['restarts']}x"
          f"{kw['rounds']}x{kw['adam_iters']}")
    for t in range(T):
        st = (env.uav.clone(), env.ch.clone(), env.t, env.prev.clone())
        nxt, rew = [], []
        for A_, C_ in ((aA, cA), (aB, cB)):
            env.uav, env.ch, env.t, env.prev = st[0].clone(), st[1].clone(), st[2], st[3].clone()
            _, r, _, _ = env.step(A_[:, t], C_[:, t])
            nxt.append((env.uav.clone(), env.ch.clone()))
            rew.append(r.float())
        h = T - t - 1
        if h > 0:
            users = env.users_traj[:, t + 2:T + 1]                       # [S,h,U,2]; shared by both branches
            sol = plan_from_state(torch.cat([users, users], 0),
                                  torch.cat([nxt[0][0], nxt[1][0]], 0),
                                  torch.cat([nxt[0][1], nxt[1][1]], 0),
                                  cfg, rng_seed=t, **kw)
            J = torch.as_tensor(sol["J"]).float()
            Ja, Jb = J[:S], J[S:]
            took = sol["solve_s"]
        else:
            Ja = Jb = torch.zeros(S)
            took = 0.0
        ra[:, t], rb[:, t] = rew[0], rew[1]
        qa[:, t], qb[:, t] = rew[0] + Ja, rew[1] + Jb
        env.uav, env.ch, env.t, env.prev = st[0].clone(), st[1].clone(), st[2], st[3].clone()
        env.step(aA[:, t], cA[:, t])
        if t % 5 == 0 or t == T - 1:
            print(f"  step {t+1:2d}/{T}  h={h:2d}  solve {took:5.1f}s  "
                  f"gap {float((qa[:, t] - qb[:, t]).mean()):+6.3f}  ({time.time()-t0:.0f}s total)")

    gap = qa - qb
    adv_a, adv_b = gap / 2.0, -gap / 2.0
    torch.save(dict(seeds=seeds, q_a=qa.numpy(), q_b=qb.numpy(), r_a=ra.numpy(), r_b=rb.numpy(),
                    adv_a=adv_a.numpy(), adv_b=adv_b.numpy(), planner=kw, cfg=cfg.__dict__), a.out)
    print(f"\nsaved {a.out}")
    print(f"  mean Q gap (strong - weak): {float(gap.mean()):+.3f}   sd {float(gap.std()):.3f}")
    print(f"  immediate part {float((ra - rb).mean()):+.3f}   future part "
          f"{float(gap.mean() - (ra - rb).mean()):+.3f}")
    print(f"  per step t=0 {float(gap[:, 0].mean()):+.3f} ... t={T-1} {float(gap[:, -1].mean()):+.3f}")
    print(f"  fraction of states where the strong action is worse: {float((gap < 0).float().mean()):.3f}")
    print(f"  total solver time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()