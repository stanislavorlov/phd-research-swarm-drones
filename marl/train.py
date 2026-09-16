"""
Train a small drone swarm with MAPPO.

Usage:
    python3 -m marl.train
    python3 -m marl.train --n-agents 5 --iterations 300 --episodes-per-iter 12

Runs from the project root (the parent of the `marl/` package).
"""

from __future__ import annotations
import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from marl.env import DroneSwarmEnv
from marl.buffer import RolloutBuffer
from marl.mappo import MAPPOAgent

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def parse_args():
    p = argparse.ArgumentParser(description="MAPPO drone-swarm experiment")
    p.add_argument("--n-agents", type=int, default=4)
    p.add_argument("--world-size", type=float, default=10.0)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--iterations", type=int, default=200, help="number of PPO update rounds")
    p.add_argument("--episodes-per-iter", type=int, default=10, help="episodes collected per rollout")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def collect_rollout_with_global_states(env, agent, episodes_per_iter, gamma, lam):
    """Same as collect_rollout but also tracks the centralized-critic input
    (global state) per stored transition, needed for the critic update."""
    all_obs, all_actions, all_logprobs = [], [], []
    all_adv, all_ret, all_global = [], [], []
    ep_returns, ep_success = [], []

    for _ in range(episodes_per_iter):
        agent_bufs = [RolloutBuffer() for _ in range(env.n_agents)]
        global_states_per_step = []
        obs = env.reset()
        done = False
        ep_reward = 0.0
        last_info = {}
        while not done:
            gstate = env.global_state(obs)
            value = agent.value(gstate)
            actions, logprobs = [], []
            for i in range(env.n_agents):
                a, lp, _ = agent.act(obs[i])
                actions.append(a)
                logprobs.append(lp)
            actions = np.array(actions)
            next_obs, rewards, done, info = env.step(actions)
            last_info = info
            global_states_per_step.append(gstate)
            for i in range(env.n_agents):
                agent_bufs[i].add(obs[i], actions[i], logprobs[i], value, rewards[i], done)
            ep_reward += float(rewards.mean())
            obs = next_obs

        ep_returns.append(ep_reward)
        ep_success.append(last_info.get("success_rate", 0.0))
        global_states_per_step = np.array(global_states_per_step)

        for i in range(env.n_agents):
            b = agent_bufs[i]
            adv, ret = b.compute_gae(None, gamma=gamma, lam=lam)
            data = b.get()
            all_obs.append(data["obs"])
            all_actions.append(data["actions"])
            all_logprobs.append(data["logprobs"])
            all_adv.append(adv)
            all_ret.append(ret)
            all_global.append(global_states_per_step)

    return (
        np.concatenate(all_obs, axis=0),
        np.concatenate(all_actions, axis=0),
        np.concatenate(all_logprobs, axis=0),
        np.concatenate(all_adv, axis=0),
        np.concatenate(all_ret, axis=0),
        np.concatenate(all_global, axis=0),
        ep_returns,
        ep_success,
    )


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    env = DroneSwarmEnv(
        n_agents=args.n_agents,
        world_size=args.world_size,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    agent = MAPPOAgent(
        obs_dim=env.obs_dim,
        global_state_dim=env.global_state_dim,
        n_actions=env.n_actions,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        n_epochs=args.n_epochs,
        minibatch_size=args.minibatch_size,
        seed=args.seed,
    )

    history = {"iteration": [], "mean_return": [], "success_rate": [], "policy_loss": [], "value_loss": [], "entropy": []}
    t0 = time.time()

    for it in range(1, args.iterations + 1):
        obs, actions, logprobs, adv, ret, gstates, ep_returns, ep_success = collect_rollout_with_global_states(
            env, agent, args.episodes_per_iter, args.gamma, args.lam
        )
        stats = agent.update(obs, actions, logprobs, adv, ret, gstates)

        history["iteration"].append(it)
        history["mean_return"].append(float(np.mean(ep_returns)))
        history["success_rate"].append(float(np.mean(ep_success)))
        history["policy_loss"].append(stats["policy_loss"])
        history["value_loss"].append(stats["value_loss"])
        history["entropy"].append(stats["entropy"])

        if it % args.log_every == 0 or it == 1:
            elapsed = time.time() - t0
            print(
                f"iter {it:4d}/{args.iterations} | "
                f"mean_return {history['mean_return'][-1]:7.3f} | "
                f"success_rate {history['success_rate'][-1]:5.2f} | "
                f"policy_loss {stats['policy_loss']:7.4f} | "
                f"value_loss {stats['value_loss']:7.4f} | "
                f"entropy {stats['entropy']:6.3f} | "
                f"elapsed {elapsed:6.1f}s"
            )

    agent.save(os.path.join(RESULTS_DIR, "mappo_policy.npz"))
    np.savez(os.path.join(RESULTS_DIR, "training_history.npz"), **{k: np.array(v) for k, v in history.items()})
    plot_learning_curve(history)
    print(f"\nDone. Results written to {RESULTS_DIR}")


def plot_learning_curve(history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["iteration"], history["mean_return"])
    axes[0].set_title("Mean episode return (avg over agents)")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("return")
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["iteration"], history["success_rate"], color="tab:green")
    axes[1].set_title("Fraction of drones reaching their goal")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("success rate")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.3)

    axes[2].plot(history["iteration"], history["policy_loss"], label="policy loss")
    axes[2].plot(history["iteration"], history["value_loss"], label="value loss")
    axes[2].set_title("Losses")
    axes[2].set_xlabel("iteration")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "learning_curve.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved learning curve to {out_path}")


if __name__ == "__main__":
    main()
