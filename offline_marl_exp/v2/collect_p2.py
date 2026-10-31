"""Collect the paired-dataset arms into one table: per-seed monitor returns, mean +/- sd, and n_eff.

    python3 collect_p2.py                  # everything under results2/
    python3 collect_p2.py --only P0 P05 P_hard P_rand

n_eff is read from the last weights[...] line in each run's log.txt, since summary.json does not store it.
"""
import argparse, glob, json, os, re, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results2")
ap.add_argument("--only", nargs="*", default=None, help="substring filter on arm name")
a = ap.parse_args()

W = re.compile(r"weights\[(\w+)\].*n_eff\s+(\d+)/(\d+)\s+\(([\d.]+)%\)")
arms = {}
for s in sorted(glob.glob(os.path.join(a.out, "*", "seed*", "summary.json"))):
    try:
        d = json.load(open(s))
    except Exception as e:
        print(f"!! {s}: {e}"); continue
    name = d.get("name", os.path.basename(os.path.dirname(os.path.dirname(s))))
    if a.only and not any(k in name for k in a.only):
        continue
    w = d.get("weights") or {}
    mode, beta, neff = w.get("mode", "-"), w.get("beta"), None
    lg = os.path.join(os.path.dirname(s), "log.txt")
    if os.path.exists(lg):
        for ln in open(lg, errors="replace"):
            m = W.search(ln)
            if m:
                mode, neff = m.group(1), float(m.group(4))
    key = (name, w.get("source", "-"), mode, beta)
    arms.setdefault(key, {"ret": [], "seed": [], "neff": neff,
                          "ce": w.get("critic_epochs"), "cf": w.get("critic_full")})
    arms[key]["ret"].append(d.get("monitor_return_best"))
    arms[key]["seed"].append(d.get("seed"))
    if neff is not None:
        arms[key]["neff"] = neff

rows = []
for (name, src, mode, beta), v in arms.items():
    r = [x for x in v["ret"] if x is not None]
    if not r:
        continue
    rows.append((sum(r) / len(r), name, src, mode, beta, v["neff"], v["ce"], r, sorted(v["seed"])))
rows.sort(key=lambda x: -x[0])

print(f"{'arm':<14} {'src':<7} {'mode':<5} {'beta':>5} {'n_eff%':>7} {'crit':>5} "
      f"{'n':>2} {'mean':>7} {'sd':>6}   per-seed")
print("-" * 104)
for mean, name, src, mode, beta, neff, ce, r, seeds in rows:
    sd = st.stdev(r) if len(r) > 1 else float("nan")
    print(f"{name:<14} {src:<7} {mode:<5} "
          f"{('-' if beta is None else f'{beta:.2f}'):>5} "
          f"{('-' if neff is None else f'{neff:.1f}'):>7} "
          f"{('-' if ce is None else str(ce)):>5} {len(r):>2} {mean:7.2f} {sd:6.2f}   "
          + "  ".join(f"s{s}:{x:.2f}" for s, x in zip(seeds, r)))
print("\nmonitor set: 128 scenarios from seed 700000. Not the test set -- use final_eval.py for tables.")

miss = [p for p in sorted(glob.glob(os.path.join(a.out, "*", "seed*")))
        if os.path.isdir(p) and not os.path.exists(os.path.join(p, "summary.json"))]
if miss:
    print("\nno summary.json (died or still running):")
    for p in miss:
        print("  ", p)