"""
scratch_sweep_table.py -- compare scratch PPO configs (works on running jobs: it parses the log files).

    python scratch_sweep_table.py                       # default tags
    python scratch_sweep_table.py --max_it 1350 --seeds 0 1

Baseline = default scratch config from results2/online/scratch_4500 truncated at --max_it (seeds in --seeds).
Columns: monitor eval (seeds 700000..) seed-mean at iterations 450 / 900 / 1350 (nearest logged point <= that), mean of the last
10 logged points, first iteration where the 3-point-averaged seed-mean reaches 80 / 90, number of seeds/progress.
Held-out final (256 scenarios) is printed for runs that finished with --final (not for the truncated baseline).
"""
from __future__ import annotations
import argparse, os, re
import numpy as np

LINE = re.compile(r"\bit\s+(\d+)\s+eval\s+(-?[\d.]+)")
STAY, EXPERT = 51.08, 118.89


def read_log(path):
    pts = {}
    if os.path.exists(path):
        for ln in open(path):
            m = LINE.search(ln)
            if m:
                pts[int(m.group(1))] = float(m.group(2))
    return pts


def at(pts, it):
    ks = [k for k in pts if k <= it]
    return pts[max(ks)] if ks and it - max(ks) <= 15 else np.nan


def first_cross(its, m, level, w=3):
    if len(m) < w:
        return None
    sm = np.convolve(m, np.ones(w) / w, mode="valid")
    h = np.where(sm >= level)[0]
    return int(its[h[0] + w - 1]) if len(h) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results2/online")
    ap.add_argument("--tags", nargs="*", default=["scratch_4500", "sw_lr1e-3", "sw_std0.1", "sw_std0.4", "sw_wu15", "sw_combo"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--max_it", type=int, default=1350)
    a = ap.parse_args()
    cp = [450, 900, a.max_it]
    print(f"monitor eval, mean over seeds {a.seeds}; stay-put {STAY}, A7+online@450 ~90.2, scratch default@{a.max_it} ~89.6 (3 seeds)")
    print(f"{'tag':14s} {'n':>2s} {'last_it':>7s}  " + " ".join(f"{'@' + str(c):>8s}" for c in cp) + f" {'last10':>7s} {'>=80':>6s} {'>=90':>6s}  held-out final")
    for tag in a.tags:
        logs = {}
        for s in a.seeds:
            name = f"log_masked_seed{s}.txt"
            pts = read_log(os.path.join(a.root, tag, name))
            pts = {k: v for k, v in pts.items() if k <= a.max_it}
            if pts:
                logs[s] = pts
        if not logs:
            print(f"{tag:14s}  no logs"); continue
        L = min(max(p) for p in logs.values())                        # common progress
        its = np.array(sorted(k for k in next(iter(logs.values())) if k <= L))
        m = np.mean([[p.get(int(i), np.nan) for i in its] for p in logs.values()], 0)
        vals = [np.nanmean([at({k: v for k, v in p.items() if k <= L}, c) for p in logs.values()]) if c <= L else np.nan for c in cp]
        last10 = float(np.nanmean(m[-10:]))
        c80, c90 = first_cross(its, m, 80), first_cross(its, m, 90)
        held = []
        if tag != "scratch_4500":
            for s in logs:
                f = os.path.join(a.root, tag, f"run_masked_seed{s}.npz")
                if os.path.exists(f):
                    d = np.load(f)
                    if "final_returns" in d.files:
                        held.append(float(np.mean(d["final_returns"])))
        hs = (f"{np.mean(held):.2f} ({' '.join(f'{x:.1f}' for x in held)})" if held else "-")
        print(f"{tag:14s} {len(logs):2d} {L:7d}  " + " ".join(f"{v:8.2f}" for v in vals) + f" {last10:7.2f} {str(c80):>6s} {str(c90):>6s}  {hs}")


if __name__ == "__main__":
    main()
