"""
DroneSwarmEnv: a small cooperative multi-agent environment for the MAPPO
experiment.

N drones share a bounded 2D world. Each drone has its own goal point and is
rewarded for closing the distance to it, while the whole swarm is jointly
penalized whenever two drones fly too close to each other (a soft
collision-avoidance constraint). This keeps the task genuinely multi-agent:
agents must reach individual goals *and* coordinate to avoid collisions,
which a purely independent (non-cooperative) policy tends to violate.

The environment intentionally has no external dependencies (no gym /
pettingzoo) so the whole experiment stays runnable with just numpy.
"""

from __future__ import annotations
import numpy as np

# Discrete action set: stay, up, down, left, right.
ACTIONS = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [0.0, -1.0],
    [-1.0, 0.0],
    [1.0, 0.0],
], dtype=np.float64)
N_ACTIONS = len(ACTIONS)


class DroneSwarmEnv:
    def __init__(
        self,
        n_agents: int = 4,
        world_size: float = 10.0,
        step_size: float = 0.5,
        max_steps: int = 60,
        goal_radius: float = 0.4,
        collision_radius: float = 0.6,
        collision_penalty: float = 1.0,
        goal_bonus: float = 5.0,
        move_noise: float = 0.02,
        seed: int | None = None,
    ):
        self.n_agents = n_agents
        self.world_size = world_size
        self.step_size = step_size
        self.max_steps = max_steps
        self.goal_radius = goal_radius
        self.collision_radius = collision_radius
        self.collision_penalty = collision_penalty
        self.goal_bonus = goal_bonus
        self.move_noise = move_noise
        self.rng = np.random.default_rng(seed)

        # obs_i = [own pos (2), vec to own goal (2), reached flag (1),
        #          relative pos to every other agent (2*(n_agents-1))]
        self.obs_dim = 2 + 2 + 1 + 2 * (n_agents - 1)
        self.global_state_dim = self.obs_dim * n_agents
        self.n_actions = N_ACTIONS

        self.positions = None
        self.goals = None
        self.reached = None
        self.t = 0

    # ------------------------------------------------------------------
    def reset(self):
        self.positions = self.rng.uniform(0, self.world_size, size=(self.n_agents, 2))
        self.goals = self.rng.uniform(0, self.world_size, size=(self.n_agents, 2))
        self.reached = np.zeros(self.n_agents, dtype=bool)
        self.t = 0
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float64)
        s = self.world_size
        for i in range(self.n_agents):
            own_pos = self.positions[i] / s
            to_goal = (self.goals[i] - self.positions[i]) / s
            others = np.delete(self.positions, i, axis=0)
            rel_others = ((others - self.positions[i]) / s).reshape(-1)
            obs[i] = np.concatenate([
                own_pos, to_goal, [float(self.reached[i])], rel_others
            ])
        return obs

    def step(self, actions):
        """actions: int array of shape (n_agents,), values in [0, N_ACTIONS)."""
        moves = ACTIONS[actions] * self.step_size
        moves += self.rng.normal(0, self.move_noise, size=moves.shape)
        self.positions = np.clip(self.positions + moves, 0, self.world_size)
        self.t += 1

        dists = np.linalg.norm(self.positions - self.goals, axis=1)
        newly_reached = (dists < self.goal_radius) & (~self.reached)
        self.reached |= dists < self.goal_radius

        # Per-agent shaping reward: progress toward goal.
        rewards = -dists / self.world_size
        rewards = rewards + newly_reached.astype(np.float64) * self.goal_bonus

        # Shared collision penalty, split across the involved agents.
        collision_pen = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                if d < self.collision_radius:
                    collision_pen[i] -= self.collision_penalty
                    collision_pen[j] -= self.collision_penalty
        rewards = rewards + collision_pen

        done = self.t >= self.max_steps or bool(np.all(self.reached))
        info = {
            "success_rate": float(np.mean(self.reached)),
            "n_collisions": float(np.sum(collision_pen < 0) / 2),
        }
        return self._get_obs(), rewards, done, info

    def global_state(self, obs):
        return obs.reshape(-1)
