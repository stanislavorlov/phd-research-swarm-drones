"""
student.py -- train the decentralised k-masked student: BC on the open-loop expert, then (optionally) DAgger rounds
with a receding-horizon expert. One run = one (config, seed). Writes results2/<name>/seed<seed>/.

    # budget-only baseline (A0): no DAgger
    python student.py --name A0 --rounds 0 --epochs0 300 --d 128 --seed 0
    # everything on
    python student.py --name A4 --head mdn --chunk 4 --obs_extra --rounds 5 --seed 0 --final

Seed sets (never reused for anything else):  DAgger collection 2_000_000+r*10_000+j | monitor 700000..  | final 800000..
`--final` evaluates the BEST round (picked on the monitor seeds) on the final seeds: do that once per reported config.

NOTE: this code was written without being executed by its author. Run `python -m pytest tests -q` and `--smoke` first.
"""
from __future__ import annotations
import argparse, copy, json, os, time
from dataclasses import replace
import numpy as np
import torch

from commenv import EnvConfig, StayPut
from evalproto import returns, paired_diff, expert_normalised
from models import NetPolicy
import bc2
from bc2 import (Rows, expert_rows, dagger_collect, train_student, closed_loop, expert_returns_on,
                 add_student_args, student_opts, make_student, save_student)

MONITOR0, FINAL0 = 700_000, 800_000


def main():
    ap = argparse.ArgumentParser()
    add_student_args(ap)
    ap.add_argument("--name", default="run")
    ap.add_argument("--out", default="results2")
    ap.add_argument("--rounds", type=int, default=0, help="DAgger rounds after the round-0 BC (0 = BC only)")
    ap.add_argument("--scen", type=int, default=256, help="scenarios collected per DAgger round")
    ap.add_argument("--epochs_round", type=int, default=30)
    ap.add_argument("--beta0", type=float, default=0.5, help="prob. of executing the EXPERT action in round r is beta0**r")
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--n_monitor", type=int, default=128)
    ap.add_argument("--n_final", type=int, default=256)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end check (seconds); results are meaningless")
    ap.add_argument("--force", action="store_true", help="ignore the run lock / existing summary.json")
    a = ap.parse_args()

    if a.smoke:
        a.epochs0, a.rounds, a.scen, a.epochs_round, a.n_monitor, a.n_final = 2, min(a.rounds, 1) or 1, 8, 1, 16, 16
        a.plan_restarts, a.plan_rounds, a.plan_iters, a.final = 2, 1, 10, False
    run_dir = os.path.join(a.out, a.name, f"seed{a.seed}")
    os.makedirs(run_dir, exist_ok=True)
    # one process per (name, seed): refuse to start if a live process holds the lock or the run already finished
    lock = os.path.join(run_dir, ".lock")
    if not a.force:
        if os.path.exists(os.path.join(run_dir, "summary.json")):
            raise SystemExit(f"{run_dir} already has summary.json -- refusing to run again (use --force to overwrite)")
        if os.path.exists(lock):
            try:
                pid = int(open(lock).read().strip()); os.kill(pid, 0); alive = True
            except PermissionError:
                alive = True                      # process exists, owned by someone else
            except (ValueError, ProcessLookupError, OSError):
                alive = False                     # stale or unreadable lock
            if alive:
                raise SystemExit(f"{run_dir} is already being run by pid {pid} -- refusing to start a second copy")
    with open(lock, "w") as f:
        f.write(str(os.getpid()))
    import atexit
    atexit.register(lambda: os.path.exists(lock) and os.remove(lock))
    logf = open(os.path.join(run_dir, "log.txt"), "a")

    def log(*s):
        msg = " ".join(str(x) for x in s)
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device)
    opts = student_opts(a)
    cfg = bc2.cfg_from_opts(opts)
    cfg_plan = replace(cfg, energy=a.plan_energy) if a.plan_energy is not None else cfg
    if a.plan_energy is not None and a.plan_energy != cfg.energy and a.rounds > 0:
        log("WARNING: --plan_energy != env energy: DAgger labels use the planner price but round-0 data (data/*.pt) were "
            "made at 0.5. For the energy-price experiment use quality_sweep.py, which regenerates round-0 data too.")
    plan_kw = dict(restarts=a.plan_restarts, rounds=a.plan_rounds, adam_iters=a.plan_iters, device=a.device)
    strong_kw = dict(restarts=4, rounds=4, adam_iters=80, device=a.device)         # reference expert (fixed)
    log(f"== {a.name} seed {a.seed}  opts={opts}  device={a.device}  args={vars(a)}")

    # ---- round-0 data: open-loop expert replayed through the env (so obs_extra etc. are the env's own)
    def load(path, n=None):
        d = torch.load(path, weights_only=False)
        s, A, C = d["seeds"], d["actions"], d["channels"]
        if n:
            s, A, C = s[:n], A[:n], C[:n]
        return expert_rows(s, cfg, A, C, H=a.chunk)
    t0 = time.time()
    train_rows = load(a.train_data, 60 if a.smoke else None)
    val_rows = load(a.val_data, 16 if a.smoke else None)
    log(f"round-0 rows: train {len(train_rows)}  val {len(val_rows)}  ({time.time()-t0:.0f}s)")

    # ---- fixed references on the monitor seeds
    mon = list(range(MONITOR0, MONITOR0 + a.n_monitor))
    t0 = time.time()
    r_stay = returns(StayPut(), mon, cfg)[0]
    r_exp = expert_returns_on(mon, cfg, cfg, strong_kw)
    log(f"monitor refs: stay-put {r_stay.mean():.2f}  expert {r_exp.mean():.2f}  ({time.time()-t0:.0f}s)")

    model, ema = make_student(opts), make_student(opts)
    ema.load_state_dict(model.state_dict())
    hist = []

    def monitor(r, extra=None):
        ret = closed_loop(ema, mon, cfg)
        sc = expert_normalised(ret, r_stay, r_exp)
        va = bc2.eval_rows(ema, val_rows.to(dev), dev)
        row = dict(round=r, monitor_return=float(ret.mean()), norm=sc, val_mse0=va["mse0"], val_acc=va["acc"],
                   n_rows=len(train_rows), **(extra or {}))
        hist.append(row)
        save_student(os.path.join(run_dir, f"round{r}.pt"), ema, opts)
        log(f"[round {r}] closed-loop {ret.mean():.2f}  expert-normalised {sc:.3f}  val mse0 {va['mse0']:.4f} acc {va['acc']:.3f}"
            + (f"  {extra}" if extra else ""))
        return sc

    # ---- round 0: BC
    log(f"-- round 0: BC {a.epochs0} epochs")
    train_student(model, ema, train_rows, val_rows, a.epochs0, dev, bs=a.bs, lr=a.lr, wd=a.wd, aug=not a.no_aug,
                  extra=a.obs_extra, seed=a.seed, w_chunk=a.w_chunk, log_every=max(1, a.epochs0 // 10), log=log)
    monitor(0)

    # ---- DAgger rounds
    for r in range(1, a.rounds + 1):
        beta = a.beta0 ** r
        seeds = [2_000_000 + r * 10_000 + j for j in range(a.scen)]
        actor_cpu = copy.deepcopy(ema).cpu().eval()
        t0 = time.time()
        new, plan_s, roll_ret = dagger_collect(actor_cpu, seeds, cfg, cfg_plan, beta, plan_kw, H=a.chunk,
                                               noise=a.noise, rng_seed=a.seed * 1000 + r)
        log(f"-- round {r}: collected {len(new)} rows  beta {beta:.3f}  mixed-rollout return {roll_ret:.2f}  "
            f"plan {plan_s:.1f}s/step  total {time.time()-t0:.0f}s")
        train_rows = Rows.cat([train_rows, new])
        train_student(model, ema, train_rows, val_rows, a.epochs_round, dev, bs=a.bs, lr=a.lr * 0.5, wd=a.wd,
                      aug=not a.no_aug, extra=a.obs_extra, seed=a.seed + r, w_chunk=a.w_chunk,
                      log_every=max(1, a.epochs_round // 3), log=log)
        monitor(r, dict(beta=beta, collect_return=roll_ret))

    best = max(hist, key=lambda h: h["norm"])
    log(f"best round by monitor: {best['round']}  norm {best['norm']:.3f}")
    summary = dict(name=a.name, seed=a.seed, opts=opts, args=vars(a), hist=hist, best_round=best["round"],
                   monitor_norm_best=best["norm"], monitor_return_best=best["monitor_return"],
                   monitor_expert=float(r_exp.mean()), monitor_stay=float(r_stay.mean()))
    bm = bc2.load_student(os.path.join(run_dir, f"round{best['round']}.pt"))[0]
    save_student(os.path.join(run_dir, "best.pt"), bm, opts)

    if a.final:
        fin = list(range(FINAL0, FINAL0 + a.n_final))
        r_s = returns(StayPut(), fin, cfg)[0]
        r_x = expert_returns_on(fin, cfg, cfg, strong_kw)
        r_m = closed_loop(bm, fin, cfg)
        summary["final"] = dict(seeds=[fin[0], fin[-1]], student=float(r_m.mean()), stay=float(r_s.mean()),
                                expert=float(r_x.mean()), norm=expert_normalised(r_m, r_s, r_x),
                                vs_stay=paired_diff(r_m, r_s), vs_expert=paired_diff(r_m, r_x))
        log(f"FINAL (best round {best['round']}): student {r_m.mean():.2f} stay {r_s.mean():.2f} expert {r_x.mean():.2f} "
            f"norm {summary['final']['norm']:.3f}  paired vs stay {summary['final']['vs_stay']['mean']:+.2f} "
            f"{summary['final']['vs_stay']['ci95']}")
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log(f"wrote {run_dir}/summary.json")


if __name__ == "__main__":
    main()
