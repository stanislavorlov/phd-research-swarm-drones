"""
scratch_compare.py -- from-scratch MAPPO vs distilled start (A7) + online MAPPO.

    python scratch_compare.py
    python scratch_compare.py --a7_tag online_A7 --seeds 0 1 2

Reads  results2/online/<tag>/run_masked_seed<s>.npz  for tags scratch_450, scratch_4500 and online_A7 (masked arm only).
npz keys: iters [E], returns [E, n_eval_seeds] (monitor seeds 700000..), final_returns [256] (held-out 800000.., only with --final).
Prints (1) monitor eval at checkpoints, (2) first iteration where scratch reaches the A7+online(450) level,
(3) held-out final scores + normalised score where available. Paired by seed index only in the sense of a shared PPO data seed;
the actors start differently, so with n=3 treat the CIs as indicative.
"""
from __future__ import annotations
import argparse, os
import numpy as np

STAY, EXPERT = 51.08, 118.89
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def load(root, tag, s):
    f = os.path.join(root, tag, f"run_masked_seed{s}.npz")
    return dict(np.load(f)) if os.path.exists(f) else None


def curve(d):
    return np.asarray(d["iters"]), np.asarray(d["returns"]).mean(1)


def at(d, it):
    its, m = curve(d)
    return float(m[np.argmin(np.abs(its - it))]) if len(its) and abs(its[np.argmin(np.abs(its - it))] - it) <= 7 else np.nan


def norm(x):
    return (x - STAY) / (EXPERT - STAY)


def ci(v):
    v = np.asarray(v, float); n = len(v)
    if n < 2:
        return np.nan, np.nan
    h = T975.get(n - 1, 1.96) * v.std(ddof=1) / np.sqrt(n)
    return v.mean() - h, v.mean() + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results2/online")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--a7_tag", default="online_A7")
    ap.add_argument("--long_tag", default="scratch_4500")
    ap.add_argument("--short_tag", default="scratch_450")
    ap.add_argument("--smooth", type=int, default=3, help="moving-average window (in eval points) for the crossing test")
    a = ap.parse_args()
    S = a.seeds
    runs = {"A7+online": {s: load(a.root, a.a7_tag, s) for s in S},
            "scratch(450)": {s: load(a.root, a.short_tag, s) for s in S},
            "scratch(4500)": {s: load(a.root, a.long_tag, s) for s in S}}
    for k, v in runs.items():
        miss = [s for s, d in v.items() if d is None]
        if miss:
            print(f"[warn] {k}: missing seeds {miss}")
        runs[k] = {s: d for s, d in v.items() if d is not None}

    cps = [0, 150, 300, 450, 900, 1350, 1800, 2250, 2700, 3150, 3600, 4050, 4500]
    print("\nMonitor eval (seeds 700000..), mean over eval seeds; per-seed in brackets")
    print(f"{'it':>5s}  " + "  ".join(f"{k:>30s}" for k in runs))
    for it in cps:
        row = []
        for k, v in runs.items():
            vals = [at(d, it) for d in v.values()]
            vals = [x for x in vals if not np.isnan(x)]
            row.append(f"{np.mean(vals):6.2f} [{' '.join(f'{x:5.1f}' for x in vals)}]" if vals else "-")
        print(f"{it:5d}  " + "  ".join(f"{r:>30s}" for r in row))

    # crossing test: first iteration where the smoothed seed-mean of scratch reaches the A7+online level at 450
    tgt_vals = [at(d, 450) for d in runs["A7+online"].values()]
    tgt = float(np.nanmean(tgt_vals)) if tgt_vals else np.nan
    print(f"\nTarget = A7+online monitor eval at it 450: {tgt:.2f}")
    for k in ("scratch(450)", "scratch(4500)"):
        v = runs[k]
        if not v:
            continue
        L = min(len(curve(d)[0]) for d in v.values())
        its = curve(next(iter(v.values())))[0][:L]
        m = np.mean([curve(d)[1][:L] for d in v.values()], 0)
        w = a.smooth
        sm = np.convolve(m, np.ones(w) / w, mode="valid")
        hit = np.where(sm >= tgt)[0]
        if len(hit):
            print(f"  {k}: seed-mean ({w}-pt avg) first >= target at it {int(its[hit[0] + w - 1])}  "
                  f"({its[hit[0] + w - 1] / 450:.1f}x the online budget)")
        else:
            print(f"  {k}: never reaches the target within {int(its[-1])} iterations (max smoothed {sm.max():.2f})")

    # per-run last-eval and plateau (mean of the last 10 eval points)
    print("\nEnd-of-run monitor eval (last point | mean of last 10 points)")
    for k, v in runs.items():
        out = []
        for s, d in sorted(v.items()):
            m = curve(d)[1]
            out.append(f"s{s}: {m[-1]:.2f}|{m[-10:].mean():.2f}")
        print(f"  {k:14s} " + "  ".join(out))

    # held-out final
    print("\nHeld-out final (seeds 800000.., 256 scenarios); normalised = (x - stay)/(expert - stay)")
    fin = {}
    for k, v in runs.items():
        vals = {s: float(np.mean(d["final_returns"])) for s, d in v.items() if "final_returns" in d}
        if vals:
            fin[k] = vals
            x = np.array(list(vals.values()))
            lo, hi = ci(x)
            print(f"  {k:14s} " + " ".join(f"s{s}:{x_:.2f}" for s, x_ in sorted(vals.items()))
                  + f"  mean {x.mean():.2f} (norm {norm(x.mean()):.3f})  95% CI [{lo:.2f}, {hi:.2f}]")
        else:
            print(f"  {k:14s} no final_returns in npz")
    for k in ("scratch(450)", "scratch(4500)"):
        if k in fin and "A7+online" in fin:
            common = sorted(set(fin[k]) & set(fin["A7+online"]))
            if len(common) >= 2:
                d = np.array([fin["A7+online"][s] - fin[k][s] for s in common])
                lo, hi = ci(d)
                print(f"  A7+online - {k}: {d.mean():+.2f} returns ({norm(d.mean()) - norm(0):+.3f} norm)  "
                      f"95% CI [{lo:+.2f}, {hi:+.2f}]  (n={len(common)}, indicative)")


if __name__ == "__main__":
    main()
