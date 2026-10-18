"""
online_contrast.py -- masked vs unmasked critic on one start (default A7), pooling PPO replicates.

Design: 3 offline students (seeds 0-2) x PPO replicates (--ppo_offset 0, 3, 6 -> PPO run seeds s, s+3, s+6) x 2 critic arms.
Within a (student, PPO seed) pair both arms share actor init, critic init and training scenarios; ONLY the critic mask differs.
Score = held-out return (seeds 800000.., saved by online2.py --final) of the final actor.

    python online_contrast.py                  # start A7, tags online_A7, online_A7_p3, online_A7_p6
    python online_contrast.py --start C_b1 --offsets 0

Two units of analysis are printed: all (student, PPO seed) pairs (n up to 9; replicates of one student are NOT independent),
and per-student means (n = 3, the conservative one). The student-level interval is the one to quote.
NOTE: written without being executed by its author.
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="A7")
ap.add_argument("--offsets", type=int, nargs="*", default=[0, 3, 6])
ap.add_argument("--students", type=int, nargs="*", default=[0, 1, 2])
ap.add_argument("--root", default="results2")
a = ap.parse_args()

try:
    from scipy import stats
    tcrit = lambda df: float(stats.t.ppf(0.975, df))
except Exception:
    tcrit = lambda df: {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31}.get(df, 2.0)

fe = json.load(open(os.path.join(a.root, "final", "final_eval.json")))
base = fe["results"][a.start]["returns"]
unit = 1.0 / (fe["expert"] - fe["stay"])

rows = []                                   # (student, offset, masked, unmasked)
for off in a.offsets:
    tag = f"online_{a.start}" + (f"_p{off}" if off else "")
    for s in a.students:
        r = {}
        for arm in ["masked", "unmasked"]:
            f = os.path.join(a.root, "online", tag, f"run_{arm}_seed{s}.npz")
            r[arm] = float(np.load(f)["final_returns"].mean()) if os.path.exists(f) else None
        if None in r.values():
            print(f"missing: {tag} student {s}"); continue
        rows.append((s, off, r["masked"], r["unmasked"]))

print(f"start {a.start}: start-student held-out returns {np.round(base, 2)}\n")
print(f"{'student':>7} {'ppo seed':>8} {'masked':>8} {'unmasked':>9} {'m-start':>8} {'u-start':>8} {'m-u':>7}")
for s, off, m, u in rows:
    print(f"{s:>7} {s+off:>8} {m:8.2f} {u:9.2f} {m-base[s]:+8.2f} {u-base[s]:+8.2f} {m-u:+7.2f}")

def summarise(name, d):
    d = np.asarray(d, float); n = len(d)
    if n < 2:
        print(f"{name}: n={n}"); return
    sd = d.std(ddof=1); h = tcrit(n - 1) * sd / np.sqrt(n)
    print(f"{name}: mean {d.mean():+.2f}  (~{d.mean()*unit:+.3f} norm)  sd {sd:.2f}  95% CI [{d.mean()-h:+.2f}, {d.mean()+h:+.2f}]  "
          f"masked>unmasked in {int((d > 0).sum())}/{n}")

print()
summarise("masked - unmasked, all pairs      ", [m - u for _, _, m, u in rows])
per = {}
for s, off, m, u in rows:
    per.setdefault(s, []).append(m - u)
summarise("masked - unmasked, per-student mean", [np.mean(v) for v in per.values()])
print("per-student mean diff:", {s: round(float(np.mean(v)), 2) for s, v in per.items()})
for arm, idx in [("masked", 2), ("unmasked", 3)]:
    summarise(f"{arm} - start student, all pairs  ", [r[idx] - base[r[0]] for r in rows])
