"""
final_eval.py -- unbiased score of already-trained students on the held-out FINAL seeds (800000..), without retraining.

    python final_eval.py --names A7 A8_kinf A9_k3 A9_k10 A9_k15

For every results2/<name>/seed*/best.pt: closed-loop return on seeds 800000..800000+n-1, expert-normalised against the stay-put
policy and the strong planner on the SAME seeds. The best ROUND was chosen on the monitor seeds (700000..), never on these.
Use it once per configuration you report; do not tune against it. Expert/stay-put returns are cached in results2/final/refs.npz.
NOTE: written without being executed by its author.
"""
import argparse, glob, json, os
import numpy as np
import torch

from commenv import EnvConfig, StayPut
from evalproto import returns, paired_diff, expert_normalised, seed_level_ci
from bc2 import load_student, closed_loop, expert_returns_on

FINAL0 = 800_000


def refs(n, device, path):
    if os.path.exists(path):
        d = np.load(path)
        if len(d["stay"]) == n:
            return d["stay"], d["expert"]
    seeds = list(range(FINAL0, FINAL0 + n)); cfg = EnvConfig()
    stay = returns(StayPut(), seeds, cfg)[0]
    exp = expert_returns_on(seeds, cfg, cfg, dict(restarts=4, rounds=4, adam_iters=80, device=device))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, stay=stay, expert=exp)
    return stay, exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--root", default="results2")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    seeds = list(range(FINAL0, FINAL0 + a.n))
    stay, exp = refs(a.n, a.device, os.path.join(a.root, "final", "refs.npz"))
    print(f"final seeds {seeds[0]}..{seeds[-1]}: stay-put {stay.mean():.2f}  expert {exp.mean():.2f}")
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
    with open(os.path.join(a.root, "final", "final_eval.json"), "w") as f:
        json.dump(dict(seeds=[seeds[0], seeds[-1]], stay=float(stay.mean()), expert=float(exp.mean()), results=out), f, indent=1)


if __name__ == "__main__":
    main()
