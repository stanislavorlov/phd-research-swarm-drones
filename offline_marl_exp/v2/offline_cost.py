"""
offline_cost.py -- what the offline (planner + DAgger) stage costs, next to the online phase it saves.

    python offline_cost.py                                  # A7, seeds 0 1 2, reads results2/A7/seed<s>/log.txt and data/*.pt
    python offline_cost.py --name A7 --seeds 0 1 2 --expert_solve_s 150 --online_s_per_it 8.5

Reads only logs and the dataset files (no planner is run).
  * round-0 expert data : n scenarios of the open-loop planner, wall clock = `solve_s` stored in data/expert_train.pt
  * DAgger              : per round `scen` scenarios rolled out for T steps, the receding-horizon planner is re-solved at EVERY visited
                          state (T solves per scenario batch). Time per round is parsed from "-- round r: ... plan X s/step total Ys".
  * training            : BC (epochs0) + one fine-tune per DAgger round; elapsed seconds of each call parsed from the epoch log lines.
  * monitor / closed-loop evaluation per round is NOT counted (evaluation overhead, also paid by any online run).
"Team step" = one environment step of one scenario (the unit used for the online budget: n_envs x T = 640 per PPO iteration).
Wall-clock numbers depend on the device (printed from the log): compare them only roughly with the online CPU runs.
"""
from __future__ import annotations
import argparse, os, re
import numpy as np

HDR = re.compile(r"^== (\S+) seed (\d+).*?device=(\w+)")
ROUND = re.compile(r"^-- round (\d+): collected (\d+) rows.*?plan ([\d.]+)s/step\s+total (\d+)s")
EPLINE = re.compile(r"\|\s*(\d+)s\s*$")
MON = re.compile(r"^\[round (\d+)\] closed-loop ([\d.]+)\s+expert-normalised (-?[\d.]+)")
R0 = re.compile(r"^round-0 rows: train (\d+)\s+val (\d+)\s+\((\d+)s\)")


def parse(path):
    lines = open(path).read().splitlines()
    start = max(i for i, l in enumerate(lines) if l.startswith("== "))        # last (re)start of this run directory
    lines = lines[start:]
    out = dict(device="?", rounds={}, train={}, r0_rows=None)
    last_train = None
    for l in lines:
        m = HDR.match(l)
        if m:
            out["device"] = m.group(3)
        m = R0.match(l)
        if m:
            out["r0_rows"] = int(m.group(1))
        m = ROUND.match(l)
        if m:
            out["rounds"][int(m.group(1))] = dict(rows=int(m.group(2)), plan_s_step=float(m.group(3)), total_s=float(m.group(4)))
            continue
        m = EPLINE.search(l)
        if m and l.lstrip().startswith("ep "):
            last_train = int(m.group(1))
            continue
        m = MON.match(l)
        if m:
            out["train"][int(m.group(1))] = last_train
            out.setdefault("norm", {})[int(m.group(1))] = float(m.group(3))
            last_train = None
    return out


def fmt_h(s):
    return f"{s / 3600:.2f} h ({s / 60:.0f} min)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="A7")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--root", default="results2")
    ap.add_argument("--train_data", default="data/expert_train.pt")
    ap.add_argument("--val_data", default="data/expert_val.pt")
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--scen", type=int, default=256)
    ap.add_argument("--n_envs", type=int, default=32, help="PPO envs per iteration")
    ap.add_argument("--expert_solve_s", type=float, default=None, help="override: wall clock of expert.py for the train set")
    ap.add_argument("--online_s_per_it", type=float, default=8.5, help="measured single-thread CPU seconds per PPO iteration")
    ap.add_argument("--online_its", type=int, default=450, help="online iterations after distillation")
    ap.add_argument("--scratch_its", type=int, default=1770, help="scratch iterations to reach the same level")
    a = ap.parse_args()

    n_tr = n_va = None; solve_tr = a.expert_solve_s; solve_va = None
    try:
        import torch
        d = torch.load(a.train_data, weights_only=False); n_tr = len(d["seeds"]); solve_tr = a.expert_solve_s or float(d["solve_s"])
        d = torch.load(a.val_data, weights_only=False); n_va = len(d["seeds"]); solve_va = float(d["solve_s"])
    except Exception as e:
        print(f"[note] dataset files not read ({type(e).__name__}); pass --expert_solve_s, scenario counts then come from the logs")

    runs = {}
    for s in a.seeds:
        f = os.path.join(a.root, a.name, f"seed{s}", "log.txt")
        if os.path.exists(f):
            runs[s] = parse(f)
        else:
            print(f"[warn] missing {f}")
    if not runs:
        raise SystemExit("no logs found")
    if n_tr is None:
        n_tr = next(iter(runs.values()))["r0_rows"]
        n_tr = None if n_tr is None else n_tr // (a.T * 8)

    print(f"\n== offline cost of {a.name} (per student seed), devices: {sorted({r['device'] for r in runs.values()})}")
    print(f"{'seed':>4s} {'rounds':>6s} {'DAgger rows':>11s} {'collect+plan s':>14s} {'plan s/step (mean)':>18s} {'train s (BC+rounds)':>20s} {'collect+train':>14s}")
    tot = []
    for s, r in sorted(runs.items()):
        rd = r["rounds"]
        col = sum(v["total_s"] for v in rd.values()); rows = sum(v["rows"] for v in rd.values())
        pl = float(np.mean([v["plan_s_step"] for v in rd.values()])) if rd else float("nan")
        tr = sum(v for v in r["train"].values() if v is not None)
        tot.append((col, tr, rows, len(rd)))
        print(f"{s:4d} {len(rd):6d} {rows:11d} {col:14.0f} {pl:18.1f} {tr:20.0f} {col + tr:14.0f}")
    col_m, tr_m = np.mean([t[0] for t in tot]), np.mean([t[1] for t in tot])
    n_rounds = int(round(np.mean([t[3] for t in tot])))

    # ---- sample / team-step accounting
    dag_steps = n_rounds * a.scen * a.T
    r0_steps = (n_tr or 0) * a.T
    print("\n== team steps (one scenario-step; online PPO iteration = n_envs x T = %d)" % (a.n_envs * a.T))
    print(f"round-0 expert data : {n_tr} scenarios x T={a.T} = {r0_steps:,} (planner solves: {n_tr} full-horizon plans)")
    print(f"DAgger              : {n_rounds} rounds x {a.scen} scenarios x T={a.T} = {dag_steps:,} (planner solves: {n_rounds * a.scen * a.T:,} receding-horizon plans = {n_rounds * a.scen:,} scenario-episodes x {a.T} states)")
    off_steps = r0_steps + dag_steps
    print(f"offline total       : {off_steps:,} team steps")
    ppo_it = a.n_envs * a.T
    for lab, its in (("A7 online", a.online_its), ("scratch (matching level)", a.scratch_its), ("scratch 4500 it", 4500)):
        print(f"{lab:25s}: {its:5d} it x {ppo_it} = {its * ppo_it:,} team steps")
    print(f"distilled total     : offline {off_steps:,} + online {a.online_its * ppo_it:,} = {off_steps + a.online_its * ppo_it:,}  "
          f"vs scratch {a.scratch_its * ppo_it:,}  (ratio scratch / distilled = {a.scratch_its * ppo_it / (off_steps + a.online_its * ppo_it):.2f})")

    # ---- wall clock
    on_s = a.online_its * a.online_s_per_it
    sc_s = a.scratch_its * a.online_s_per_it
    print("\n== wall clock (rough; offline stage on the log's device, online on single-thread CPU at %.1f s/it)" % a.online_s_per_it)
    if solve_tr is not None:
        print(f"expert data (train, one-off, shared by all students): {fmt_h(solve_tr)}" + (f"   (val set {solve_va:.0f}s)" if solve_va else ""))
    print(f"DAgger collect + planner (per student)               : {fmt_h(col_m)}")
    print(f"BC + DAgger training (per student)                   : {fmt_h(tr_m)}")
    off_s = col_m + tr_m + (solve_tr or 0.0)
    print(f"offline per student incl. one-off expert data         : {fmt_h(off_s)}   (excl. expert data: {fmt_h(col_m + tr_m)})")
    print(f"online fine-tune {a.online_its} it                            : {fmt_h(on_s)}")
    print(f"distilled pipeline total                              : {fmt_h(off_s + on_s)}")
    print(f"scratch to the same level ({a.scratch_its} it)                : {fmt_h(sc_s)}")
    print("caveat: planner/DAgger device and CPU contention differ from the online runs; treat wall clock as an order of magnitude.")


if __name__ == "__main__":
    main()
