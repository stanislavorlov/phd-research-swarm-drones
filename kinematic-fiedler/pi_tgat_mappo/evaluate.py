"""
Evaluate a trained checkpoint at fixed node-dropout rates, reproducing the
manuscript's Table 1 evaluation protocol: Network Connectivity Ratio (CR)
and Mission Completion Rate (MCR), measured at 0% / 15% / 30% simultaneous
node dropout, over many held-out episodes.

This is deliberately separate from train.py: training uses organic,
distance-based RF link dropout throughout (see env.py's logistic
P_drop(distance) model); evaluation additionally forces a FIXED fraction
of agents to go silent each step (config.node_dropout_rate), independent
of geometry, as a stress test on top of that.

Usage:
    python3 -m pi_tgat_mappo.evaluate --checkpoint pi_tgat_mappo/results/pi_tgat_mappo.pt

    # compare two checkpoints (e.g. PI-TGAT vs the memoryless-GAT baseline)
    python3 -m pi_tgat_mappo.evaluate --checkpoint results_pi_tgat/pi_tgat_mappo.pt --label "PI-TGAT (ours)"
    python3 -m pi_tgat_mappo.evaluate --checkpoint results_baseline/pi_tgat_mappo.pt --label "Memoryless GNN"

Each invocation evaluates ONE checkpoint and appends its row(s) to
pi_tgat_mappo/results/evaluation_table.json, so you can run it once per
checkpoint and then build the paper-style comparison table/plot from the
accumulated file with --summarize.
"""

from __future__ import annotations
import argparse
import dataclasses
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config
from .env import UAVSwarmEnv
from .mappo import MAPPOTrainer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TABLE_PATH = os.path.join(RESULTS_DIR, "evaluation_table.json")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a PI-TGAT/baseline checkpoint under forced node dropout")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default=None,
                    help="path to config.json saved alongside the checkpoint by train.py "
                         "(default: same directory as --checkpoint)")
    p.add_argument("--label", type=str, default=None,
                    help="row label for this checkpoint in the comparison table (default: checkpoint filename)")
    p.add_argument("--dropout-rates", type=str, default="0.0,0.15,0.30",
                    help="comma-separated node dropout rates to evaluate at (paper: 0%%, 15%%, 30%%)")
    p.add_argument("--episodes-per-rate", type=int, default=20)
    p.add_argument("--n-min", type=int, default=None, help="override the checkpoint's training n_min (e.g. for cross-scale/zero-shot transfer eval)")
    p.add_argument("--n-max", type=int, default=None, help="override the checkpoint's training n_max")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=123, help="different from the training seed -> genuinely held-out episodes")
    p.add_argument("--stochastic", action="store_true",
                    help="sample actions instead of using the deterministic policy mean")
    p.add_argument("--summarize", action="store_true",
                    help="skip evaluation; just rebuild the comparison table/plot from evaluation_table.json")
    return p.parse_args()


def load_config_for_checkpoint(args) -> Config:
    cfg_path = args.config or os.path.join(os.path.dirname(args.checkpoint), "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            raw = json.load(f)
        field_names = {f.name for f in dataclasses.fields(Config)}
        cfg = Config(**{k: v for k, v in raw.items() if k in field_names})
        print(f"loaded config from {cfg_path}")
    else:
        print(f"WARNING: no config.json found at {cfg_path}; using defaults. "
              f"If this checkpoint was trained with --no-kinematic-prior or non-default "
              f"architecture sizes, pass a matching --config explicitly or this will "
              f"error/mismatch when loading weights.")
        cfg = Config()
    if args.n_min is not None:
        cfg.n_min = args.n_min
    if args.n_max is not None:
        cfg.n_max = args.n_max
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    return cfg


def evaluate_checkpoint(args) -> dict:
    cfg = load_config_for_checkpoint(args)
    trainer = MAPPOTrainer(cfg)
    trainer.load(args.checkpoint)
    print(f"device = {trainer.device}, kinematic_prior = {cfg.use_kinematic_prior}, "
          f"N in [{cfg.n_min}, {cfg.n_max}], max_steps = {cfg.max_steps}")

    env = UAVSwarmEnv(cfg, seed=args.seed)
    dropout_rates = [float(x) for x in args.dropout_rates.split(",")]
    label = args.label or os.path.basename(args.checkpoint)

    rows = []
    for rate in dropout_rates:
        cfg.node_dropout_rate = rate
        crs, mcrs = [], []
        for _ in range(args.episodes_per_rate):
            _, stats = trainer.collect_episode(env, deterministic=not args.stochastic)
            crs.append(stats["connectivity_ratio"])
            mcrs.append(float(stats["mission_complete"]))
        row = dict(
            label=label, checkpoint=args.checkpoint, dropout_pct=rate * 100,
            n_episodes=args.episodes_per_rate,
            cr_mean=float(np.mean(crs)), cr_std=float(np.std(crs)),
            mcr_mean=float(np.mean(mcrs)) * 100, mcr_std=float(np.std(mcrs)) * 100,
        )
        rows.append(row)
        print(f"  {label} @ {rate*100:4.0f}% node dropout: "
              f"CR = {row['cr_mean']*100:5.1f}% +/- {row['cr_std']*100:4.1f}%, "
              f"MCR = {row['mcr_mean']:5.1f}% +/- {row['mcr_std']:4.1f}%  "
              f"(n={args.episodes_per_rate} episodes)")
    return rows


def append_to_table(rows: list[dict]):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    existing = []
    if os.path.exists(TABLE_PATH):
        with open(TABLE_PATH) as f:
            existing = json.load(f)
    # Replace any prior rows with the same (label, dropout_pct) rather than duplicating.
    keys = {(r["label"], r["dropout_pct"]) for r in rows}
    existing = [r for r in existing if (r["label"], r["dropout_pct"]) not in keys]
    existing.extend(rows)
    with open(TABLE_PATH, "w") as f:
        json.dump(existing, f, indent=2)
    return existing


def plot_comparison(all_rows: list[dict]):
    labels = sorted({r["label"] for r in all_rows})
    rates = sorted({r["dropout_pct"] for r in all_rows})

    fig, (ax_cr, ax_mcr) = plt.subplots(1, 2, figsize=(11, 4.5))
    width = 0.8 / max(len(labels), 1)
    x = np.arange(len(rates))

    for i, label in enumerate(labels):
        by_rate = {r["dropout_pct"]: r for r in all_rows if r["label"] == label}
        cr = [by_rate[r]["cr_mean"] * 100 if r in by_rate else np.nan for r in rates]
        cr_err = [by_rate[r]["cr_std"] * 100 if r in by_rate else 0 for r in rates]
        mcr = [by_rate[r]["mcr_mean"] if r in by_rate else np.nan for r in rates]
        mcr_err = [by_rate[r]["mcr_std"] if r in by_rate else 0 for r in rates]
        offset = (i - (len(labels) - 1) / 2) * width
        ax_cr.bar(x + offset, cr, width, yerr=cr_err, capsize=3, label=label)
        ax_mcr.bar(x + offset, mcr, width, yerr=mcr_err, capsize=3, label=label)

    for ax, title in ((ax_cr, "Network Connectivity Ratio (CR)"), (ax_mcr, "Mission Completion Rate (MCR)")):
        ax.set_xticks(x); ax.set_xticklabels([f"{r:.0f}%" for r in rates])
        ax.set_xlabel("node dropout rate"); ax.set_ylabel("%"); ax.set_ylim(0, 105)
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "evaluation_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {out_path}")


def print_table(all_rows: list[dict]):
    labels = sorted({r["label"] for r in all_rows})
    rates = sorted({r["dropout_pct"] for r in all_rows})
    print("\n" + "=" * 78)
    print(f"{'Node Dropout':>14} | {'Architecture':<22} | {'CR':>16} | {'MCR':>16}")
    print("-" * 78)
    for rate in rates:
        for label in labels:
            match = [r for r in all_rows if r["label"] == label and r["dropout_pct"] == rate]
            if not match:
                continue
            r = match[0]
            print(f"{rate:>13.0f}% | {label:<22} | {r['cr_mean']*100:5.1f}% +/- {r['cr_std']*100:4.1f}% "
                  f"| {r['mcr_mean']:5.1f}% +/- {r['mcr_std']:4.1f}%")
    print("=" * 78)


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not args.summarize:
        rows = evaluate_checkpoint(args)
        all_rows = append_to_table(rows)
    else:
        if not os.path.exists(TABLE_PATH):
            raise SystemExit(f"--summarize but {TABLE_PATH} does not exist yet -- run without --summarize first.")
        with open(TABLE_PATH) as f:
            all_rows = json.load(f)

    print_table(all_rows)
    plot_comparison(all_rows)


if __name__ == "__main__":
    main()
