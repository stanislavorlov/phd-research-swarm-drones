"""
planner_check.py -- does the expert reference move if the planner is given more budget?

    python planner_check.py --n 64                       # strong (4x4) vs strong+ (16x8), paired, dev seeds
    python planner_check.py --n 64 --seed0 900000        # same check on the test seeds
    python planner_check.py --n 32 --grid               # also 8x4, 4x8, 16x8 with more Adam iters

Every normalised score in v1-v4 divides by (expert - stay-put) with the expert = the strong planner
(restarts 4, rounds 4, adam_iters 80). If a bigger budget finds materially better trajectories, that
denominator is wrong and every normalised number has to be regenerated. This script measures the gap
on the SAME scenarios (paired), so the comparison is not limited by scenario-to-scenario spread.

Decision rule: if the paired mean improvement is below the per-scenario paired SEM (and below ~1 return,
i.e. ~0.015 normalised), the reference is stable and nothing needs regenerating. Report the number either way.
NOTE: written without being executed by its author.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch

from commenv import EnvConfig, StayPut
from evalproto import returns, paired_diff
from bc2 import expert_returns_on

CONFIGS = {
    "strong":   dict(restarts=4,  rounds=4,  adam_iters=80),    # the reference used everywhere (and the DAgger teacher)
    "strong+":  dict(restarts=16, rounds=8,  adam_iters=80),
    "strong++": dict(restarts=32, rounds=16, adam_iters=160),   # --ladder: is strong+ itself converged?
    "strong3":  dict(restarts=64, rounds=16, adam_iters=320),
    "restarts": dict(restarts=16, rounds=4,  adam_iters=80),    # --grid: which knob matters
    "rounds":   dict(restarts=4,  rounds=8,  adam_iters=80),
    "iters":    dict(restarts=4,  rounds=4,  adam_iters=240),
}
LADDER = ["strong", "strong+", "strong++", "strong3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed0", type=int, default=800_000)
    ap.add_argument("--grid", action="store_true", help="also run the single-knob variants")
    ap.add_argument("--ladder", action="store_true", help="budget ladder: is the STRONGER planner itself converged?")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results2/final/planner_check.json")
    a = ap.parse_args()
    seeds = list(range(a.seed0, a.seed0 + a.n))
    cfg = EnvConfig()
    names = (LADDER if a.ladder else ["strong", "strong+"]) + (["restarts", "rounds", "iters"] if a.grid else [])

    stay = returns(StayPut(), seeds, cfg)[0]
    res = {}
    for nm in names:
        kw = dict(CONFIGS[nm], device=a.device)
        t0 = time.time()
        r = expert_returns_on(seeds, cfg, cfg, kw)
        res[nm] = dict(per_scenario=np.asarray(r, float), wall_s=time.time() - t0, **CONFIGS[nm])
        print(f"{nm:9s} restarts {kw['restarts']:2d} rounds {kw['rounds']} iters {kw['adam_iters']:3d}  "
              f"mean {np.mean(r):8.3f}  denominator (expert - stay) {np.mean(r) - stay.mean():8.3f}  "
              f"[{time.time() - t0:.0f}s]")

    base = res["strong"]["per_scenario"]
    print(f"\nstay-put {stay.mean():.3f};  paired vs the current reference (n={a.n} scenarios):")
    for nm in names[1:]:
        d = res[nm]["per_scenario"] - base
        sem = d.std(ddof=1) / np.sqrt(len(d))
        pd = paired_diff(torch.as_tensor(res[nm]["per_scenario"]), torch.as_tensor(base))
        norm_shift = d.mean() / (base.mean() - stay.mean())
        print(f"  {nm:9s} {d.mean():+8.3f} returns  SEM {sem:.3f}  95% CI {np.round(pd['ci95'], 3)}  "
              f"wins {int((d > 1e-6).sum())}/{len(d)}  -> every normalised score would shift by about "
              f"{-norm_shift * 100:+.1f}% of its value")
    if a.ladder:
        print("\nconsecutive rungs (each vs the previous one): the reference is converged where this stops being positive")
        for lo, hi in zip(LADDER, LADDER[1:]):
            if lo in res and hi in res:
                dd = res[hi]["per_scenario"] - res[lo]["per_scenario"]
                print(f"  {hi:9s} - {lo:9s} {dd.mean():+8.3f}  SEM {dd.std(ddof=1) / np.sqrt(len(dd)):.3f}  "
                      f"wins {int((dd > 1e-6).sum())}/{len(dd)}  [{res[hi]['wall_s'] / res['strong']['wall_s']:.0f}x the cost of strong]")
    d = res["strong+"]["per_scenario"] - base
    verdict = ("reference is STABLE: regeneration not needed"
               if abs(d.mean()) < max(1.0, d.std(ddof=1) / np.sqrt(len(d)))
               else "reference MOVES: regenerate every normalised number with the stronger planner")
    print(f"\nverdict: {verdict}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(seed0=a.seed0, n=a.n, stay=float(stay.mean()), verdict=verdict,
                   configs={k: {kk: vv for kk, vv in v.items() if kk != "per_scenario"} for k, v in res.items()},
                   means={k: float(np.mean(v["per_scenario"])) for k, v in res.items()}),
              open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()