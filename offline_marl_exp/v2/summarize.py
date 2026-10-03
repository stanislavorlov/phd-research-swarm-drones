"""
summarize.py -- one table over results2/*/seed*/summary.json  (CI = bootstrap over TRAINING seeds, the unit of analysis).

    python summarize.py                 # monitor-seed scores of the best round per run, plus --final numbers if present
    python summarize.py --root results2 --names A0 A1 A2 A3 A4
Columns: expert-normalised score on the monitor seeds (0 = stay-put, 1 = strong expert), round-0 (pure BC) vs best round,
and, if student.py was run with --final, the held-out final-seed score. Paired differences between configs are over seeds.
"""
import argparse, glob, json, os
import numpy as np
from evalproto import seed_level_ci, bootstrap_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results2")
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--baseline", default="A0")
    a = ap.parse_args()
    runs = {}
    for f in sorted(glob.glob(os.path.join(a.root, "*", "seed*", "summary.json"))):
        d = json.load(open(f)); runs.setdefault(d["name"], {})[d["seed"]] = d
    names = a.names or sorted(runs)
    print(f"{'config':12s} {'n':>2s}  {'round0 norm':>18s}  {'best-round norm':>18s}  {'final norm':>18s}  {'final return':>12s}")
    fmt = lambda v: "      -" if not v else "%6.3f [%5.3f,%5.3f]" % ((lambda m, ci: (m, ci[0], ci[1]))(*seed_level_ci(v)))
    per = {}
    for n in names:
        if n not in runs:
            continue
        R = runs[n]; seeds = sorted(R)
        r0 = [R[s]["hist"][0]["norm"] for s in seeds]
        rb = [R[s]["monitor_norm_best"] for s in seeds]
        fn = [R[s]["final"]["norm"] for s in seeds if "final" in R[s]]
        fr = [R[s]["final"]["student"] for s in seeds if "final" in R[s]]
        per[n] = dict(seeds=seeds, best=rb, final=fn)
        print(f"{n:12s} {len(seeds):2d}  {fmt(r0):>18s}  {fmt(rb):>18s}  {fmt(fn):>18s}  {np.mean(fr) if fr else float('nan'):12.2f}")
    if a.baseline in per:
        print(f"\npaired vs {a.baseline} (same seed index; best-round monitor norm), 95% bootstrap CI over seeds:")
        b = per[a.baseline]
        for n, p in per.items():
            if n == a.baseline:
                continue
            common = sorted(set(p["seeds"]) & set(b["seeds"]))
            if len(common) < 2:
                continue
            d = np.array([p["best"][p["seeds"].index(s)] - b["best"][b["seeds"].index(s)] for s in common])
            print(f"  {n:12s} {d.mean():+.3f}  [{bootstrap_ci(d)[0]:+.3f}, {bootstrap_ci(d)[1]:+.3f}]  (n={len(common)})")


if __name__ == "__main__":
    main()
