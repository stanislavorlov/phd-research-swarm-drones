"""
probe_table.py -- tabulate v4_probes2 behaviour output and pair each tuned actor with its own offline student.

    python probe_table.py                          # reads results2/final/probe_*.json
    python probe_table.py --glob 'results2/final/probe_*.json' --start A7

Answers the section-5 question on more than one actor: does the online step really buy bigger moves plus a full
initial channel re-assignment, and does it really trade away the worst-served users (q10 down, outage up)?
Pairing is by seed: actor_<arm>_seed<s>.pt is compared with results2/<start>/seed<s>/best.pt.
NOTE: written without being executed by its author.
"""
from __future__ import annotations
import argparse, glob, json, re
import numpy as np

COLS = [("ret", "return", 2), ("move", "move", 3), ("switch_t1", "sw@t1", 3), ("switch_t2plus", "sw@t>=2", 4),
        ("rate", "rate", 3), ("rate_q10", "q10", 3), ("rate_q50", "q50", 3), ("outage", "outage", 4),
        ("serv_dist", "servdist", 3), ("cochan", "cochan", 2)]


def rows(files):
    out = {}
    for f in files:
        d = json.load(open(f))
        for k, v in d.items():
            if k == "stay_put" or k.endswith("::channels") or not isinstance(v, dict) or "move" not in v:
                continue
            out[k] = v
    return out


def seed_of(path):
    m = re.search(r"seed(\d+)", path)
    return int(m.group(1)) if m else None


def arm_of(path):
    m = re.search(r"actor_(\w+?)_seed", path)
    return m.group(1) if m else "offline"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results2/final/probe_*.json")
    ap.add_argument("--start", default="A7")
    a = ap.parse_args()
    r = rows(sorted(glob.glob(a.glob)))
    if not r:
        raise SystemExit(f"no probe json matching {a.glob}")
    base = {seed_of(k): v for k, v in r.items() if arm_of(k) == "offline" and f"/{a.start}/" in k}
    tuned = {k: v for k, v in r.items() if arm_of(k) != "offline"}

    hdr = f"{'policy':34s} " + " ".join(f"{lab:>8s}" for _, lab, _ in COLS)
    print(hdr); print("-" * len(hdr))
    for s in sorted(base):
        print(f"{a.start + ' seed' + str(s) + ' (offline)':34s} " +
              " ".join(f"{base[s][k]:8.{p}f}" for k, _, p in COLS))
    for k in sorted(tuned):
        print(f"{arm_of(k) + ' seed' + str(seed_of(k)) + ' (tuned)':34s} " +
              " ".join(f"{tuned[k][c]:8.{p}f}" for c, _, p in COLS))

    print("\npaired change, tuned - its own offline student (mean over the actors present, % in brackets):")
    for arm in sorted({arm_of(k) for k in tuned}):
        ks = [k for k in tuned if arm_of(k) == arm and seed_of(k) in base]
        if not ks:
            continue
        print(f"  [{arm}, n={len(ks)}]")
        for c, lab, p in COLS:
            d = np.array([tuned[k][c] - base[seed_of(k)][c] for k in ks])
            rel = np.array([tuned[k][c] / base[seed_of(k)][c] - 1 for k in ks if base[seed_of(k)][c] != 0])
            sign = "all +" if (d > 0).all() else ("all -" if (d < 0).all() else "mixed")
            print(f"    {lab:9s} {d.mean():+9.{p}f}  ({100 * rel.mean():+6.1f}%)  per seed {np.round(d, p)}  {sign}")

    print("\nsection 5 predictions to check: move UP, sw@t1 UP a lot, q10 DOWN, outage UP, rate/q50 UP.")
    print("If q10 and outage move the other way on other seeds, the fairness trade is a single-actor artefact.")


if __name__ == "__main__":
    main()
