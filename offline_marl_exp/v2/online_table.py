"""
online_table.py -- table for the online MAPPO grid (A7 and C_b1 starts, masked vs unmasked critic, 3 seeds).
Run it where results2/final/final_eval.json and results2/online/online_*/run_*.npz both exist (your Mac after fetch.sh).

    python online_table.py            # prints, per start: held-out return of the start student, of the final actor of each
                                      # arm, the change, and masked-minus-unmasked per seed, plus monitor start/end for context.
Held-out = seeds 800000.. (final_returns saved by online2.py --final); start values come from final_eval.json (same seeds).
Norm columns are approximate: return differences / (expert - stay-put) of the held-out references.
NOTE: written without being executed by its author.
"""
import json, os, sys
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "results2"
fe = json.load(open(os.path.join(root, "final", "final_eval.json")))
stay, exp = fe["stay"], fe["expert"]
unit = 1.0 / (exp - stay)
for ck in ["A7", "C_b1"]:
    if ck not in fe["results"]:
        print(f"\n{ck}: not in final_eval.json (run final_eval.py --names {ck})"); continue
    base = fe["results"][ck]["returns"]
    print(f"\n== start {ck}  held-out return of the start student per seed: {np.round(base, 2)}  (mean {np.mean(base):.2f})")
    print(f"{'seed':>4} {'start':>8} {'masked':>8} {'unmasked':>9} {'m-start':>8} {'u-start':>8} {'m-u':>7}   monitor start -> end (masked | unmasked)")
    dm, du, dd = [], [], []
    for s in range(3):
        r = {}
        for arm in ["masked", "unmasked"]:
            f = os.path.join(root, "online", f"online_{ck}", f"run_{arm}_seed{s}.npz")
            if not os.path.exists(f):
                r[arm] = None; continue
            d = np.load(f)
            m = d["returns"].mean(1)
            r[arm] = (float(d["final_returns"].mean()), float(m[0]), float(m[-3:].mean()))
        if None in r.values():
            print(f"{s:>4}  missing run"); continue
        m, u = r["masked"][0], r["unmasked"][0]
        dm.append(m - base[s]); du.append(u - base[s]); dd.append(m - u)
        print(f"{s:>4} {base[s]:8.2f} {m:8.2f} {u:9.2f} {m-base[s]:+8.2f} {u-base[s]:+8.2f} {m-u:+7.2f}   "
              f"{r['masked'][1]:.1f}->{r['masked'][2]:.1f} | {r['unmasked'][1]:.1f}->{r['unmasked'][2]:.1f}")
    if dm:
        print(f"mean over seeds: masked {np.mean(dm):+.2f} (~{np.mean(dm)*unit:+.3f} norm)   unmasked {np.mean(du):+.2f} (~{np.mean(du)*unit:+.3f} norm)   "
              f"masked-unmasked {np.mean(dd):+.2f} (sd {np.std(dd, ddof=1) if len(dd) > 1 else float('nan'):.2f}, n={len(dd)})")
