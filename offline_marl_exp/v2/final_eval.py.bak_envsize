"""
final_eval.py -- unbiased score of already-trained students on a held-out seed set, without retraining.

    python final_eval.py --names A7 A8_kinf A9_k3 A9_k10 A9_k15                     # DEV set  800000.. (seen often)
    python final_eval.py --names ... --seed0 900000 --n 1024 --tag test             # TEST set 900000.. (frozen)

Three seed sets, never mixed:
  monitor/selection 700000..   chooses the best DAgger round inside student.py
  dev               800000..   development and arm-vs-arm contrasts; reported many times, so mildly optimistic
  test              900000..   frozen, reported ONCE per headline number; keep every diagnostic off it
For every results2/<name>/seed*/best.pt: closed-loop return on the chosen seeds, expert-normalised against the stay-put
policy and the strong planner on the SAME seeds (normalised scores are NOT comparable across seed sets -- the references
differ). References are cached per seed set in results2/final/refs_<seed0>_<n>.npz.
NOTE: written without being executed by its author.
"""
import argparse, glob, json, os
import numpy as np
import torch

from commenv import EnvConfig, StayPut
from evalproto import returns, paired_diff, expert_normalised, seed_level_ci
from bc2 import load_student, closed_loop, expert_returns_on

FINAL0 = 800_000          # dev set (default); --seed0 900000 selects the frozen test set


def refs(seed0, n, device, path, plan_kw=None):
    if os.path.exists(path):
        d = np.load(path)
        if len(d["stay"]) == n:
            return d["stay"], d["expert"]
    seeds = list(range(seed0, seed0 + n)); cfg = EnvConfig()
    stay = returns(StayPut(), seeds, cfg)[0]
    exp = expert_returns_on(seeds, cfg, cfg, plan_kw or dict(restarts=4, rounds=4, adam_iters=80, device=device))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, stay=stay, expert=exp)
    return stay, exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--root", default="results2")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed0", type=int, default=FINAL0, help="800000 = dev set, 900000 = frozen test set")
    ap.add_argument("--tag", default=None, help="suffix for the output json (e.g. test); default: derived from --seed0")
    ap.add_argument("--planner", default="strong", help="reference planner budget, a key of planner_check.CONFIGS "
                                                        "(strong = the 4x4 used in v1-v4; the ladder does not converge, "
                                                        "so this choice is a reporting convention and must be stated)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    from planner_check import CONFIGS
    if a.planner not in CONFIGS:
        raise SystemExit(f"--planner must be one of {sorted(CONFIGS)}")
    plan_kw = dict(CONFIGS[a.planner], device=a.device)
    seeds = list(range(a.seed0, a.seed0 + a.n))
    tag = a.tag or {800_000: "dev", 900_000: "test"}.get(a.seed0, str(a.seed0))
    stay, exp = refs(a.seed0, a.n, a.device,
                     os.path.join(a.root, "final", f"refs_{a.seed0}_{a.n}_{a.planner}.npz"), plan_kw)
    print(f"reference planner '{a.planner}': {CONFIGS[a.planner]}")
    print(f"{tag} seeds {seeds[0]}..{seeds[-1]}: stay-put {stay.mean():.2f}  expert {exp.mean():.2f}  "
          f"(denominator {exp.mean() - stay.mean():.2f}; normalised scores are specific to THIS seed set)")
    out = {}
    print(f"\n{'config':10s} {'n':>2s}  {'norm (mean, 95% CI over seeds)':>34s}  {'return':>8s}  per-seed norm")
    for name in a.names:
        norms, rets = [], []
        for p in sorted(glob.glob(os.path.join(a.root, name, "seed*", a.ckpt))):
            model, cfg = load_student(p)
            r = closed_loop(model, seeds, cfg)
            norms.append(expert_normalised(r, stay, exp)); rets.append(float(r.mean()))
        if not norms:
            print(f"{name:10s}  no checkpoints found"); continue
        m, ci = seed_level_ci(norms)
        out[name] = dict(norms=norms, returns=rets, mean=m, ci95=ci)
        print(f"{name:10s} {len(norms):2d}  {m:8.3f} [{ci[0]:.3f}, {ci[1]:.3f}]{'':>12s}  {np.mean(rets):8.2f}  {np.round(norms, 3)}")
    os.makedirs(os.path.join(a.root, "final"), exist_ok=True)
    name = ("final_eval.json" if a.seed0 == FINAL0 and a.tag is None and a.planner == "strong"
            else f"final_eval_{tag}_{a.planner}.json")
    with open(os.path.join(a.root, "final", name), "w") as f:
        json.dump(dict(seed_set=tag, seeds=[seeds[0], seeds[-1]], n=a.n, planner=a.planner, planner_cfg=CONFIGS[a.planner],
                       stay=float(stay.mean()), expert=float(exp.mean()), results=out), f, indent=1)
    print(f"wrote {os.path.join(a.root, 'final', name)}")


if __name__ == "__main__":
    main()
