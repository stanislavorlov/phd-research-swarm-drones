"""
Train the PI-TGAT + MAPPO swarm policy.

Usage (run from the kinematic-fiedler/ directory, or anywhere with this
package on PYTHONPATH):

    python3 -m pi_tgat_mappo.train --smoke-test
    python3 -m pi_tgat_mappo.train
    python3 -m pi_tgat_mappo.train --n-min 20 --n-max 50 --iterations 300 --max-steps 6000

--smoke-test runs a tiny (5-8 agents, 20-step episodes, 3 iterations)
version of the whole pipeline in well under a minute, to confirm your
torch / torch_geometric install and the code are wired together correctly
before committing to a longer run. Always run this first.

See config.py for the full parameter list and README.md for the
paper-value -> used-value mapping table.
"""

from __future__ import annotations
import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config
from .env import UAVSwarmEnv
from .mappo import MAPPOTrainer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def parse_args():
    p = argparse.ArgumentParser(description="PI-TGAT + MAPPO drone-swarm experiment")
    p.add_argument("--smoke-test", action="store_true",
                    help="tiny config (5-8 agents, 20-step episodes, 3 iterations) to verify the install")
    p.add_argument("--n-min", type=int, default=None)
    p.add_argument("--n-max", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--episodes-per-iter", type=int, default=None)
    p.add_argument("--kappa", type=float, default=None,
                    help="channel decay steepness; Table 1 says 0.02, body text says 0.01667")
    p.add_argument("--no-kinematic-prior", action="store_true",
                    help="disable the kinematic-prior/temporal-decay mechanism -> memoryless-GAT baseline")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="'auto' (default), 'cpu', 'mps', or 'cuda'")
    p.add_argument("--log-every", type=int, default=5)
    return p.parse_args()


def build_config(args) -> Config:
    cfg = Config()
    if args.smoke_test:
        cfg.n_min, cfg.n_max = 5, 8
        cfg.max_steps = 20
        cfg.iterations = 3
        cfg.episodes_per_iter = 2
    for field, val in [
        ("n_min", args.n_min), ("n_max", args.n_max), ("max_steps", args.max_steps),
        ("iterations", args.iterations), ("episodes_per_iter", args.episodes_per_iter),
        ("kappa", args.kappa), ("seed", args.seed), ("device", args.device),
    ]:
        if val is not None:
            setattr(cfg, field, val)
    if args.no_kinematic_prior:
        cfg.use_kinematic_prior = False
    return cfg


def main():
    args = parse_args()
    cfg = build_config(args)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    np.random.seed(cfg.seed)
    env = UAVSwarmEnv(cfg, seed=cfg.seed)
    trainer = MAPPOTrainer(cfg)
    print(f"device = {trainer.device}, kinematic_prior = {cfg.use_kinematic_prior}, "
          f"N in [{cfg.n_min}, {cfg.n_max}], max_steps = {cfg.max_steps}")

    history = {k: [] for k in [
        "iteration", "episode_return", "connectivity_ratio", "mission_complete",
        "mission_progress", "mean_fiedler", "policy_loss", "value_loss", "entropy",
        "w_align", "w_cohesion", "w_separation", "decay_rate",
    ]}
    t0 = time.time()

    for it in range(1, cfg.iterations + 1):
        episodes, ep_stats = [], []
        for _ in range(cfg.episodes_per_iter):
            buf, stats = trainer.collect_episode(env)
            episodes.append(buf)
            ep_stats.append(stats)

        update_stats = trainer.update(episodes)

        history["iteration"].append(it)
        history["episode_return"].append(float(np.mean([s["episode_return"] for s in ep_stats])))
        history["connectivity_ratio"].append(float(np.mean([s["connectivity_ratio"] for s in ep_stats])))
        history["mission_complete"].append(float(np.mean([s["mission_complete"] for s in ep_stats])))
        history["mission_progress"].append(float(np.mean([s["mission_progress"] for s in ep_stats])))
        history["mean_fiedler"].append(float(np.mean([s["mean_fiedler"] for s in ep_stats])))
        for k in ("policy_loss", "value_loss", "entropy", "w_align", "w_cohesion", "w_separation", "decay_rate"):
            history[k].append(update_stats[k])

        if it % args.log_every == 0 or it == 1:
            elapsed = time.time() - t0
            print(
                f"iter {it:4d}/{cfg.iterations} | return {history['episode_return'][-1]:7.3f} | "
                f"CR {history['connectivity_ratio'][-1]:5.2f} | "
                f"MCR {history['mission_complete'][-1]:5.2f} | "
                f"progress {history['mission_progress'][-1]:5.2f} | "
                f"fiedler {history['mean_fiedler'][-1]:.4f} | "
                f"pi_loss {update_stats['policy_loss']:7.4f} | v_loss {update_stats['value_loss']:7.4f} | "
                f"elapsed {elapsed:6.1f}s"
            )

    ckpt_path = os.path.join(RESULTS_DIR, "pi_tgat_mappo.pt")
    trainer.save(ckpt_path)
    np.savez(os.path.join(RESULTS_DIR, "training_history.npz"), **{k: np.array(v) for k, v in history.items()})
    plot_learning_curve(history)
    print(f"\nDone. Results written to {RESULTS_DIR}")


def plot_learning_curve(history):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].plot(history["iteration"], history["episode_return"])
    axes[0, 0].set_title("Mean episode return"); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(history["iteration"], history["connectivity_ratio"], color="tab:green")
    axes[0, 1].set_title("Network Connectivity Ratio (CR)")
    axes[0, 1].set_ylim(0, 1.05); axes[0, 1].grid(alpha=0.3)

    axes[0, 2].plot(history["iteration"], history["mission_complete"], color="tab:purple")
    axes[0, 2].set_title("Mission Completion Rate (MCR)")
    axes[0, 2].set_ylim(0, 1.05); axes[0, 2].grid(alpha=0.3)

    axes[1, 0].plot(history["iteration"], history["mean_fiedler"], color="tab:red")
    axes[1, 0].axhline(0.2, ls="--", color="grey", lw=1, label="lambda_crit = 0.2")
    axes[1, 0].set_title("Mean Fiedler value per episode"); axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(history["iteration"], history["policy_loss"], label="policy loss")
    axes[1, 1].plot(history["iteration"], history["value_loss"], label="value loss")
    axes[1, 1].set_title("Losses"); axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    axes[1, 2].plot(history["iteration"], history["w_align"], label="w_align")
    axes[1, 2].plot(history["iteration"], history["w_cohesion"], label="w_cohesion")
    axes[1, 2].plot(history["iteration"], history["w_separation"], label="w_separation")
    axes[1, 2].plot(history["iteration"], history["decay_rate"], label="decay_rate", ls="--")
    axes[1, 2].set_title("Learned kinematic-prior parameters")
    axes[1, 2].legend(fontsize=7); axes[1, 2].grid(alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("iteration")
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "learning_curve.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved learning curve to {out_path}")


if __name__ == "__main__":
    main()
