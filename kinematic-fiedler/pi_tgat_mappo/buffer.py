"""
EpisodeBuffer: stores one full episode's worth of transitions (graphs are
kept as raw numpy so nothing pins a torch computation graph), plus GAE
computation over the SCALAR team-level reward/value sequence.

Design note: the centralized critic here outputs a single permutation-
invariant V(s) for the whole swarm (see networks.CentralizedCritic), not a
per-agent value. So the GAE advantage is computed once per timestep from
the team-averaged reward and that scalar value, and the SAME advantage is
then applied to every agent's policy-gradient term at that timestep -- a
standard simplification for a fully centralized critic in CTDE.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class EpisodeBuffer:
    node_feats: list = field(default_factory=list)   # each (n, 7)
    edge_index: list = field(default_factory=list)    # each (2, E)
    edge_attr: list = field(default_factory=list)     # each (E, 17)
    actions: list = field(default_factory=list)       # each (n, 3)
    old_logprob: list = field(default_factory=list)   # each (n,) sum over action dims
    old_value: list = field(default_factory=list)     # each scalar
    team_reward: list = field(default_factory=list)   # each scalar (mean over agents)
    n_agents: int = 0

    def add(self, node_feats, edge_index, edge_attr, actions, old_logprob, old_value, reward):
        self.node_feats.append(node_feats)
        self.edge_index.append(edge_index)
        self.edge_attr.append(edge_attr)
        self.actions.append(actions)
        self.old_logprob.append(old_logprob)
        self.old_value.append(old_value)
        self.team_reward.append(float(np.mean(reward)))
        self.n_agents = node_feats.shape[0]

    def __len__(self):
        return len(self.node_feats)

    def compute_gae(self, gamma: float, lam: float):
        rewards = np.array(self.team_reward, dtype=np.float64)
        values = np.array(self.old_value, dtype=np.float64)
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float64)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_value = values[t + 1] if t + 1 < T else 0.0
            delta = rewards[t] + gamma * next_value - values[t]
            last_gae = delta + gamma * lam * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns
