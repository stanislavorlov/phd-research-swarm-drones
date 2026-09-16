"""Rollout storage + Generalized Advantage Estimation (GAE-Lambda)."""

from __future__ import annotations
import numpy as np


class RolloutBuffer:
    """
    Stores one on-policy rollout (a batch of episodes) with parameter
    sharing across agents: every agent's transition is a separate row,
    but a shared actor and a centralized critic (keyed by episode+timestep
    global state) are used for all of them.
    """

    def __init__(self):
        self.obs = []          # per-agent local observation
        self.actions = []
        self.logprobs = []     # old log pi(a|obs) at collection time
        self.values = []       # V(global_state) at collection time
        self.rewards = []
        self.dones = []        # episode-terminal flag (shared across agents at that step)

    def add(self, obs, action, logprob, value, reward, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.obs)

    def compute_gae(self, last_values, gamma=0.99, lam=0.95):
        """
        `last_values` is a dict episode_end_index -> bootstrap value, but for
        simplicity we instead require the caller to have already appended a
        trailing bootstrap value/reward=0/done marker per episode; see
        train.py for how this is used. Returns (advantages, returns) arrays
        aligned with self.rewards (dones mark episode boundaries).
        """
        rewards = np.array(self.rewards, dtype=np.float64)
        values = np.array(self.values, dtype=np.float64)
        dones = np.array(self.dones, dtype=np.float64)

        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float64)
        last_gae = 0.0
        for t in reversed(range(T)):
            if dones[t]:
                next_value = 0.0
                next_nonterminal = 0.0
            else:
                next_value = values[t + 1] if t + 1 < T else 0.0
                next_nonterminal = 1.0
            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + gamma * lam * next_nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    def get(self):
        return {
            "obs": np.array(self.obs, dtype=np.float64),
            "actions": np.array(self.actions, dtype=np.int64),
            "logprobs": np.array(self.logprobs, dtype=np.float64),
            "values": np.array(self.values, dtype=np.float64),
            "rewards": np.array(self.rewards, dtype=np.float64),
            "dones": np.array(self.dones, dtype=np.float64),
        }

    def clear(self):
        self.__init__()
