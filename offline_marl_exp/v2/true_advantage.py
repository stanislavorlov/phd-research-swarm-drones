"""
true_advantage.py -- is there any advantage to learn?

    python3 true_advantage.py --n 32 --steps 0 5 10 15

Every experiment so far has assumed a one-step deviation from the strong trajectory is worth something, and
blamed the critic when the measured advantage came out at zero. This measures the ground truth instead.

At selected states along the strong trajectory, for each branch (strong action vs weak action):
    step once        -> immediate reward r and next state s'
    solve from s'    -> J, the strong planner's objective over the remaining horizon
    Q_true = r + J   (take this action, then play well)

The reported `advantage` is Q_true(strong) - Q_true(weak), decomposed into its immediate and future parts.

  gap >> 0  the signal exists, the critics were failing to estimate it, keep pushing on offline RL.
  gap ~ 0   single-step deviations genuinely do not matter here: a receding-horizon controller recovers.
            Then IQL = BC is the correct answer, not a tuning failure, and no critic or dataset fixes it.

NOTE: written without being executed by its author.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch

from commenv import EnvConfig, CommEnv
from planner_state import plan_from_state


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/expert_train.pt")
    ap.add_argument("--pairs", default="data/pairs_train.pt")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, nargs="*", default=[0, 5, 10, 15])
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    p = torch.load(a.pairs, map_location="cpu", weights_only=False)
    k = a.n
    seeds = list(p["seeds"])[:k]
    aA, cA = torch.as_tensor(p["actions_a"][:k]).float(), torch.as_tensor(p["channels_a"][:k]).long()
    aB, cB = torch.as_tensor(p["actions_b"][:k]).float(), torch.as_tensor(p["channels_b"][:k]).long()
    cfg = EnvConfig(**p["cfg"]) if isinstance(p.get("cfg"), dict) else EnvConfig()
    kw = dict(restarts=a.restarts, rounds=a.rounds, adam_iters=a.iters, device=a.device)

    env = CommEnv(cfg, seeds)
    env.reset()
    T = cfg.T
    print(f"{k} scenarios, strong planner {kw['restarts']}x{kw['rounds']}x{kw['adam_iters']}\n")
    print(f"{'t':>3}  {'r_strong':>9} {'r_weak':>8} {'dr':>7}   {'J_strong':>9} {'J_weak':>8} {'dJ':>7}   "
          f"{'adv':>7} {'adv/|A|':>8}")
    rows = []
    for t in range(T):
        if t in a.steps:
            st = (env.uav.clone(), env.ch.clone(), env.t, env.prev.clone())
            out = {}
            for name, (A_, C_) in (("strong", (aA, cA)), ("weak", (aB, cB))):
                env.uav, env.ch, env.t, env.prev = st[0].clone(), st[1].clone(), st[2], st[3].clone()
                _, r, _, _ = env.step(A_[:, t], C_[:, t])
                h = T - t - 1
                if h > 0:
                    sol = plan_from_state(env.users_traj[:, t + 2:T + 1], env.uav, env.ch, cfg,
                                          rng_seed=t, **kw)
                    J = torch.as_tensor(sol["J"]).float()
                else:
                    J = torch.zeros(env.B)
                out[name] = (r.float(), J)
            env.uav, env.ch, env.t, env.prev = st[0].clone(), st[1].clone(), st[2], st[3].clone()
            rs, Js = out["strong"]; rw, Jw = out["weak"]
            dr, dJ = (rs - rw), (Js - Jw)
            adv = dr + dJ
            mag = (aA[:, t] - aB[:, t]).norm(dim=-1).mean()
            rows.append((t, float(rs.mean()), float(rw.mean()), float(dr.mean()),
                         float(Js.mean()), float(Jw.mean()), float(dJ.mean()),
                         float(adv.mean()), float(adv.mean() / mag.clamp(min=1e-6))))
            print(f"{t:3d}  {rows[-1][1]:9.3f} {rows[-1][2]:8.3f} {rows[-1][3]:+7.3f}   "
                  f"{rows[-1][4]:9.3f} {rows[-1][5]:8.3f} {rows[-1][6]:+7.3f}   "
                  f"{rows[-1][7]:+7.3f} {rows[-1][8]:+8.3f}")
        env.step(aA[:, t], cA[:, t])

    A = np.array([r[7] for r in rows])
    print(f"\nmean true one-step advantage of the strong action: {A.mean():+.3f} returns "
          f"(per-step reward scale is about {np.mean([r[1] for r in rows]):.1f})")
    print("decomposition: immediate "
          f"{np.mean([r[3] for r in rows]):+.3f}, future {np.mean([r[6] for r in rows]):+.3f}")
    print("\nRead it against the critic's measured gap of ~0.00. If this is also ~0, the critic was right:")
    print("a receding-horizon controller recovers from a single bad action, so there is no advantage to learn")
    print("and IQL = BC is the correct result rather than a tuning failure.")


if __name__ == "__main__":
    main()