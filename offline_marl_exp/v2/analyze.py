"""
analyze.py -- paired analysis + figure for the online phase.

Unit of analysis = TRAINING SEED. For every (arm, seed, checkpoint) we have the deterministic return on the same 256
held-out scenarios, so arm differences are computed scenario-paired inside a seed, then averaged per seed, and the
bootstrap CI is over seeds (n = number of seeds), never over episodes.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evalproto import bootstrap_ci, min_detectable_effect

RES = "results"
ARMS = {"bc_masked": "BC → MAPPO, masked critic (Ours)", "bc_unmasked": "BC → MAPPO, unmasked critic",
        "scratch_masked": "scratch → MAPPO, masked critic", "scratch_unmasked": "scratch → MAPPO, unmasked critic"}


def load():
    runs = {}
    for f in sorted(glob.glob(f"{RES}/run_*_seed*.npz")):
        b = os.path.basename(f)[4:-4]
        arm, s = b.rsplit("_seed", 1)
        d = np.load(f)
        runs.setdefault(arm, {})[int(s)] = (d["iters"], d["returns"])          # returns [n_ckpt, n_eval_scen]
    return runs


def seed_means(runs, arm, ckpt):
    """per-seed mean over scenarios at checkpoint index ckpt"""
    return {s: v[1][ckpt].mean() for s, v in runs[arm].items()}


def main():
    refs = np.load(f"{RES}/refs.npz")
    runs = load()
    arms = [a for a in ARMS if a in runs]
    lines = []
    P = lambda s="": (print(s), lines.append(s))
    stay, rnd, exp_ = refs["stay_put"].mean(), refs["random"].mean(), refs["expert"].mean()
    ns = lambda r: (r - stay) / (exp_ - stay)
    P(f"Reference returns on the {len(refs['stay_put'])} held-out scenarios: stay-put {stay:.2f}, random {rnd:.2f}, "
      f"weak expert {refs['weak_expert'].mean():.2f}, strong expert {exp_:.2f}, interference-free upper-bound proxy {refs['ub_proxy'].mean():.2f}")
    P(f"Expert solve time: {float(refs['expert_solve_s_single']):.2f} s for ONE scenario (4 restarts x 4 BCD rounds, CPU); "
      f"{float(refs['expert_solve_s_batched_per_scenario']):.2f} s/scenario when batched\n")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.35, 1]))
    col = {"bc_masked": "#1f77b4", "bc_unmasked": "#d62728", "scratch_masked": "#2ca02c", "scratch_unmasked": "#9467bd"}
    final = {}
    for a in arms:
        seeds = sorted(runs[a]); its = runs[a][seeds[0]][0]
        M = np.stack([[runs[a][s][1][c].mean() for c in range(len(its))] for s in seeds])   # [seeds, ckpt]
        mean = M.mean(0)
        ci = np.array([bootstrap_ci(M[:, c], n_boot=4000) for c in range(len(its))])
        ax[0].plot(its, mean, color=col[a], label=f"{ARMS[a]} (n={len(seeds)})", lw=2)
        ax[0].fill_between(its, ci[:, 0], ci[:, 1], color=col[a], alpha=0.18)
        final[a] = M[:, -1]
        P(f"{ARMS[a]:40s} start {mean[0]:7.2f}  final {mean[-1]:7.2f}  95% CI over seeds [{ci[-1,0]:.2f}, {ci[-1,1]:.2f}]  "
          f"expert-normalised final {ns(mean[-1]):+.3f}")
    for v, name, ls in ((exp_, "expert (oracle)", "--"), (stay, "do nothing", ":"), (rnd, "random", "-.")):
        ax[0].axhline(v, color="k", ls=ls, lw=1, alpha=0.7); ax[0].text(ax[0].get_xlim()[1], v, " " + name, va="center", fontsize=8)
    ax[0].set_xlabel("MAPPO iteration (32 scenarios x 20 steps each)"); ax[0].set_ylabel("held-out return (256 paired scenarios)")
    ax[0].set_title("Deterministic evaluation at fixed checkpoints, mean ± 95% bootstrap CI over training seeds")
    ax[0].title.set_fontsize(9); ax[0].legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, 0.92)); ax[0].grid(alpha=0.3)

    P("\nPaired differences at the final checkpoint (per-seed, same training scenarios & noise; bootstrap CI over seeds):")
    pairs = [("bc_masked", "bc_unmasked", "mask effect (BC init)"), ("scratch_masked", "scratch_unmasked", "mask effect (scratch init)"),
             ("bc_masked", "scratch_masked", "initialisation effect (masked critic)")]
    rows = []
    for a, b, name in pairs:
        if a in final and b in final:
            common = sorted(set(runs[a]) & set(runs[b]))
            d = np.array([seed_means(runs, a, -1)[s] - seed_means(runs, b, -1)[s] for s in common])
            lo, hi = bootstrap_ci(d, n_boot=10000)
            P(f"  {name:38s} {d.mean():+8.2f}  CI [{lo:+.2f}, {hi:+.2f}]  ({'separated from 0' if lo > 0 or hi < 0 else 'NOT separated from 0'}, n={len(d)} seeds)")
            rows.append((name, d))
    if "bc_masked" in runs:
        d = np.array([seed_means(runs, "bc_masked", -1)[s] - seed_means(runs, "bc_masked", 0)[s] for s in sorted(runs["bc_masked"])])
        lo, hi = bootstrap_ci(d, n_boot=10000)
        P(f"  {'online gain over BC-only (masked)':38s} {d.mean():+8.2f}  CI [{lo:+.2f}, {hi:+.2f}]  (n={len(d)} seeds)")
    if "bc_unmasked" in runs:
        d = np.array([seed_means(runs, "bc_unmasked", -1)[s] - seed_means(runs, "bc_unmasked", 0)[s] for s in sorted(runs["bc_unmasked"])])
        lo, hi = bootstrap_ci(d, n_boot=10000)
        P(f"  {'online gain over BC-only (unmasked)':38s} {d.mean():+8.2f}  CI [{lo:+.2f}, {hi:+.2f}]  (n={len(d)} seeds)")

    # panel b: expert-normalised score at the end
    labels, vals = [], []
    if "bc_masked" in runs:
        labels.append("BC-only"); vals.append(np.array([ns(v) for v in seed_means(runs, "bc_masked", 0).values()]))
    for a in arms:
        labels.append(ARMS[a].replace(" → ", "→\n").replace(", ", ",\n")); vals.append(ns(final[a]))
    try:
        ax[1].boxplot(vals, tick_labels=labels, showmeans=True)
    except TypeError:
        ax[1].boxplot(vals, labels=labels, showmeans=True)
    for i, v in enumerate(vals):
        ax[1].scatter(np.full(len(v), i + 1) + np.random.default_rng(0).uniform(-.08, .08, len(v)), v, s=14, color="k", zorder=3)
    ax[1].axhline(1.0, color="k", ls="--", lw=1); ax[1].axhline(0.0, color="k", ls=":", lw=1)
    ax[1].set_ylabel("expert-normalised score  (return − stay-put)/(expert − stay-put)")
    ax[1].set_title("Final policy, one dot per training seed", fontsize=9); ax[1].tick_params(axis="x", labelsize=7); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{RES}/online_comparison.png", dpi=140)
    open(f"{RES}/online_summary.txt", "w").write("\n".join(lines))
    print(f"\nwrote {RES}/online_comparison.png and {RES}/online_summary.txt")


if __name__ == "__main__":
    main()
