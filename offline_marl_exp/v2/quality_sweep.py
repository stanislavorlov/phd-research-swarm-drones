"""
quality_sweep.py -- item 4: is a *better* expert a *better teacher*?  (learnability vs optimality)

For each variant a fresh open-loop dataset is planned (different energy price the planner optimises, and/or a weaker
solver), replayed through the env at its true energy price, and the SAME student recipe (BC only, same seed) is trained
on it. Reported: expert env-return, student closed-loop return, both normalised against the same strong reference.

    python quality_sweep.py --n_train 600 --epochs 150 --d 128 --seed 0
    python quality_sweep.py --variants base smooth2 --head mdn --chunk 4

Variants (planner energy price, restarts, BCD rounds, Adam iters):
    base 0.5/4/4/80   smooth1 1.0/4/4/80   smooth2 2.0/4/4/80   weak 0.5/1/1/20   weakest 0.5/1/1/5
Seeds: train 100000.. (same scenarios as the main data), val 200000.., monitor 700000.. . No final seeds are touched.
NOTE: written without being executed by its author.
"""
from __future__ import annotations
import argparse, json, os, time
from dataclasses import replace
import numpy as np
import torch

from commenv import EnvConfig, StayPut
from evalproto import returns, expert_normalised
from expert import replay_returns
from planner_state import plan_from_state
from commenv import CommEnv
import bc2
from bc2 import expert_rows, train_student, closed_loop, expert_returns_on, add_student_args, student_opts, make_student

VARIANTS = dict(base=(0.5, 4, 4, 80), smooth1=(1.0, 4, 4, 80), smooth2=(2.0, 4, 4, 80),
                weak=(0.5, 1, 1, 20), weakest=(0.5, 1, 1, 5))


def plan_dataset(seeds, cfg, lam, restarts, rounds, iters, device, chunk=100):
    cfg_plan = replace(cfg, energy=lam)
    A, C = [], []
    for i in range(0, len(seeds), chunk):
        env = CommEnv(cfg, seeds[i:i + chunk]); env.reset()
        p = plan_from_state(env.users_traj[:, 1:], env.uav0, env.ch0, cfg_plan, restarts=restarts, rounds=rounds,
                            adam_iters=iters, device=device)
        A.append(p["actions"]); C.append(p["channels"])
    A, C = torch.cat(A).numpy(), torch.cat(C).numpy()
    return A, C, replay_returns(seeds, cfg, A, C)          # env-return at the TRUE energy price


def main():
    ap = argparse.ArgumentParser()
    add_student_args(ap)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--n_train", type=int, default=600)
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--n_monitor", type=int, default=128)
    ap.add_argument("--out", default="results2/quality_sweep")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    cfg = EnvConfig(k=a.k, obs_extra=bool(a.obs_extra))
    opts = student_opts(a)
    mon = list(range(700_000, 700_000 + a.n_monitor))
    r_stay = returns(StayPut(), mon, cfg)[0]
    r_ref = expert_returns_on(mon, cfg, cfg, dict(restarts=4, rounds=4, adam_iters=80, device=a.device))
    print(f"monitor: stay {r_stay.mean():.2f}  strong expert {r_ref.mean():.2f}", flush=True)
    tr_seeds = list(range(100_000, 100_000 + a.n_train))
    va_seeds = list(range(200_000, 200_000 + a.n_val))
    rows = []
    for name in a.variants:
        lam, R, rd, it = VARIANTS[name]
        t0 = time.time()
        A, C, ret_tr = plan_dataset(tr_seeds, cfg, lam, R, rd, it, a.device)
        Av, Cv, _ = plan_dataset(va_seeds, cfg, lam, R, rd, it, a.device)
        t_plan = time.time() - t0
        torch.manual_seed(a.seed)
        tr = expert_rows(tr_seeds, cfg, A, C, H=a.chunk)
        va = expert_rows(va_seeds, cfg, Av, Cv, H=a.chunk)
        model, ema = make_student(opts), make_student(opts)
        ema.load_state_dict(model.state_dict())
        train_student(model, ema, tr, va, a.epochs, dev, bs=a.bs, lr=a.lr, wd=a.wd, aug=not a.no_aug, extra=a.obs_extra,
                      seed=a.seed, w_chunk=a.w_chunk, log_every=max(1, a.epochs // 5))
        ret = closed_loop(ema, mon, cfg)
        row = dict(variant=name, plan_energy=lam, restarts=R, rounds=rd, iters=it, plan_s=t_plan,
                   expert_train_return=float(ret_tr.mean()), student_monitor=float(ret.mean()),
                   student_norm=expert_normalised(ret, r_stay, r_ref),
                   student_over_own_expert=float(ret.mean() / ret_tr.mean()))
        rows.append(row)
        print(row, flush=True)
        json.dump(dict(args=vars(a), stay=float(r_stay.mean()), ref=float(r_ref.mean()), rows=rows),
                  open(os.path.join(a.out, f"sweep_seed{a.seed}.json"), "w"), indent=1)
    print("\nvariant    plan-energy  expert(train)  student(monitor)  student-norm(vs strong expert)")
    for r in rows:
        print(f"{r['variant']:9s}  {r['plan_energy']:8.1f}   {r['expert_train_return']:10.2f}   {r['student_monitor']:12.2f}   {r['student_norm']:8.3f}")


if __name__ == "__main__":
    main()
