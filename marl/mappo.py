"""
MAPPO (Multi-Agent PPO) with parameter sharing and a centralized critic.

- Actor: one shared policy network pi_theta(a | local_obs), used by every
  (homogeneous) drone -- this is the standard MAPPO / "shared parameters"
  setup for teams of identical agents.
- Critic: one centralized value network V_phi(global_state) that sees the
  concatenated observations of all agents (Centralized Training, 
  Decentralized Execution -- CTDE). Only the actor is needed at deployment
  time; the critic is a training-time-only crutch for lower-variance
  advantages.
- Update: clipped PPO surrogate objective + entropy bonus for the actor,
  MSE for the critic, both trained with a hand-rolled Adam optimizer over
  the tiny autodiff engine in autodiff.py.
"""

from __future__ import annotations
import numpy as np

from .autodiff import Tensor, Adam
from .networks import CategoricalActor, CentralizedCritic


class MAPPOAgent:
    def __init__(
        self,
        obs_dim,
        global_state_dim,
        n_actions,
        hidden_dim=64,
        actor_lr=3e-4,
        critic_lr=1e-3,
        clip_eps=0.2,
        entropy_coef=0.01,
        vf_coef=0.5,
        n_epochs=4,
        minibatch_size=256,
        max_grad_norm=0.5,
        seed=0,
    ):
        self.rng = np.random.default_rng(seed)
        self.actor = CategoricalActor(obs_dim, n_actions, hidden_dim, self.rng)
        self.critic = CentralizedCritic(global_state_dim, hidden_dim, self.rng)
        self.actor_opt = Adam(self.actor.params(), lr=actor_lr)
        self.critic_opt = Adam(self.critic.params(), lr=critic_lr)

        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm

    # ------------------------------------------------------------------
    def act(self, obs_np):
        return self.actor.act(obs_np, self.rng)

    def value(self, global_state_np):
        return self.critic.value(global_state_np)

    # ------------------------------------------------------------------
    def update(self, obs, actions, old_logprobs, advantages, returns, global_states):
        """
        obs:            (N, obs_dim)      per-agent local observations
        actions:        (N,)              int actions taken
        old_logprobs:   (N,)              log pi_old(a|obs) at collection time
        advantages:     (N,)              GAE advantages (already per-agent-row)
        returns:        (N,)              value targets (GAE returns)
        global_states:  (N, global_state_dim)  centralized critic inputs
        """
        n = obs.shape[0]
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n_updates": 0}

        for _ in range(self.n_epochs):
            idx = self.rng.permutation(n)
            for start in range(0, n, self.minibatch_size):
                mb = idx[start:start + self.minibatch_size]
                stats_mb = self._update_minibatch(
                    obs[mb], actions[mb], old_logprobs[mb], adv[mb], returns[mb], global_states[mb]
                )
                for k in ("policy_loss", "value_loss", "entropy"):
                    stats[k] += stats_mb[k]
                stats["n_updates"] += 1

        for k in ("policy_loss", "value_loss", "entropy"):
            stats[k] /= max(stats["n_updates"], 1)
        return stats

    def _update_minibatch(self, obs, actions, old_logprobs, adv, returns, global_states):
        # ---- actor (PPO-clip) ----
        self.actor_opt.zero_grad()
        obs_t = Tensor(obs, requires_grad=False)
        probs = self.actor.forward_probs(obs_t)          # (B, n_actions)
        logp_all = probs.log()
        new_logp = logp_all.gather(actions)               # (B,)

        ratio = (new_logp - Tensor(old_logprobs, requires_grad=False)).exp()
        adv_t = Tensor(adv, requires_grad=False)
        surr1 = ratio * adv_t
        surr2 = ratio.clip(1 - self.clip_eps, 1 + self.clip_eps) * adv_t
        policy_obj = surr1.minimum(surr2).mean()

        entropy = (probs * logp_all).sum(axis=1).mean() * -1.0
        actor_loss = policy_obj * -1.0 + entropy * (-self.entropy_coef)
        actor_loss.backward()
        self.actor_opt.step(clip_norm=self.max_grad_norm)

        # ---- critic (MSE) ----
        self.critic_opt.zero_grad()
        state_t = Tensor(global_states, requires_grad=False)
        values = self.critic.forward_value(state_t)  # (B, 1)
        returns_t = Tensor(returns.reshape(-1, 1), requires_grad=False)
        value_loss = ((values - returns_t) ** 2).mean()
        (value_loss * self.vf_coef).backward()
        self.critic_opt.step(clip_norm=self.max_grad_norm)

        return {
            "policy_loss": float(-policy_obj.data),
            "value_loss": float(value_loss.data),
            "entropy": float(entropy.data),
        }

    # ------------------------------------------------------------------
    def save(self, path):
        params = {}
        for i, p in enumerate(self.actor.params()):
            params[f"actor_{i}"] = p.data
        for i, p in enumerate(self.critic.params()):
            params[f"critic_{i}"] = p.data
        np.savez(path, **params)

    def load(self, path):
        data = np.load(path)
        for i, p in enumerate(self.actor.params()):
            p.data = data[f"actor_{i}"]
        for i, p in enumerate(self.critic.params()):
            p.data = data[f"critic_{i}"]
