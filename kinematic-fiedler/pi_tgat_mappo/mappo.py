"""
MAPPOTrainer: collects episodes with the current (shared) PI-TGAT actor and
centralized critic, then updates both via a clipped-PPO-surrogate objective.

Because the actor is recurrent (a GRUCell carries temporal memory across a
communication blackout, per the paper), each PPO epoch replays every stored
episode from t=0 with a fresh hidden state and recomputes the whole forward
pass (full-episode backprop-through-time) to get new log-probs/values --
this is the standard way to do recurrent PPO correctly, at the cost of
being more expensive per update than a feedforward MLP policy would be.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from .config import Config
from .env import UAVSwarmEnv
from .buffer import EpisodeBuffer
from .networks import PITGATActor, CentralizedCritic


def _to_tensor(x, device, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=device)


class MAPPOTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = cfg.resolve_device()
        self.actor = PITGATActor(cfg).to(self.device)
        self.critic = CentralizedCritic(cfg).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def collect_episode(self, env: UAVSwarmEnv, n_agents: int | None = None) -> tuple[EpisodeBuffer, dict]:
        cfg = self.cfg
        graph = env.reset(n_agents=n_agents)
        hidden = self.actor.init_hidden(env.n, self.device)
        buf = EpisodeBuffer()
        ep_return, fiedler_hist, connected_hist = 0.0, [], []

        done = False
        info = {}
        while not done:
            node_feats = _to_tensor(graph["node_feats"], self.device)
            edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long, device=self.device)
            edge_attr = _to_tensor(graph["edge_attr"], self.device)

            mean, log_std, hidden = self.actor(node_feats, edge_index, edge_attr, hidden)
            std = log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
            action = dist.sample()
            logprob = dist.log_prob(action).sum(dim=-1)  # (n,)
            value = self.critic(node_feats)               # (1,)

            action_np = action.clamp(-1.0, 1.0).cpu().numpy()
            next_graph, reward, done, info = env.step(action_np)

            buf.add(
                node_feats=graph["node_feats"], edge_index=graph["edge_index"],
                edge_attr=graph["edge_attr"], actions=action_np,
                old_logprob=logprob.cpu().numpy(), old_value=float(value.item()),
                reward=reward,
            )
            ep_return += float(np.mean(reward))
            fiedler_hist.append(info["fiedler_value"])
            connected_hist.append(info["connected"])
            graph = next_graph

        stats = dict(
            n_agents=env.n,
            episode_return=ep_return,
            connectivity_ratio=float(np.mean(connected_hist)),
            mission_complete=float(info.get("mission_complete", False)),
            mission_progress=info.get("mission_progress", 0.0),
            mean_fiedler=float(np.mean(fiedler_hist)),
        )
        return buf, stats

    # ------------------------------------------------------------------
    def _replay_episode(self, buf: EpisodeBuffer):
        """Recomputes new log-probs, entropy, and values for a stored episode
        under the CURRENT actor/critic parameters (i.e. WITH gradients)."""
        T = len(buf)
        hidden = self.actor.init_hidden(buf.n_agents, self.device)
        new_logprobs, entropies, values = [], [], []
        for t in range(T):
            node_feats = _to_tensor(buf.node_feats[t], self.device)
            edge_index = torch.as_tensor(buf.edge_index[t], dtype=torch.long, device=self.device)
            edge_attr = _to_tensor(buf.edge_attr[t], self.device)
            actions = _to_tensor(buf.actions[t], self.device)

            mean, log_std, hidden = self.actor(node_feats, edge_index, edge_attr, hidden)
            std = log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
            new_logprobs.append(dist.log_prob(actions).sum(dim=-1))   # (n,)
            entropies.append(dist.entropy().sum(dim=-1))               # (n,)
            values.append(self.critic(node_feats).squeeze(-1))         # scalar tensor

        return torch.stack(new_logprobs), torch.stack(entropies), torch.stack(values)

    # ------------------------------------------------------------------
    def update(self, episodes: list[EpisodeBuffer]):
        cfg = self.cfg
        all_adv, all_ret = [], []
        for buf in episodes:
            adv, ret = buf.compute_gae(gamma=0.99, lam=cfg.gae_lambda)
            all_adv.append(adv)
            all_ret.append(ret)
        flat_adv = np.concatenate(all_adv)
        adv_mean, adv_std = flat_adv.mean(), flat_adv.std() + 1e-8

        stats = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0, n_updates=0)
        for _ in range(cfg.n_epochs):
            for buf, adv, ret in zip(episodes, all_adv, all_ret):
                adv_norm = (adv - adv_mean) / adv_std
                adv_t = _to_tensor(adv_norm, self.device)          # (T,)
                ret_t = _to_tensor(ret, self.device)                 # (T,)
                old_logprob_t = _to_tensor(np.stack(buf.old_logprob), self.device)  # (T, n)

                new_logprob, entropy, values = self._replay_episode(buf)  # (T, n), (T, n), (T,)

                ratio = (new_logprob - old_logprob_t).exp()                   # (T, n)
                adv_broadcast = adv_t.unsqueeze(1)                             # (T, 1) -> broadcasts over agents
                surr1 = ratio * adv_broadcast
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_broadcast
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()
                value_loss = ((values - ret_t) ** 2).mean()

                self.actor_opt.zero_grad()
                (policy_loss + cfg.entropy_coef * entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                self.actor_opt.step()

                self.critic_opt.zero_grad()
                (cfg.vf_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.critic_opt.step()

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(-entropy_loss.item())
                stats["n_updates"] += 1

        for k in ("policy_loss", "value_loss", "entropy"):
            stats[k] /= max(stats["n_updates"], 1)
        stats["decay_rate"] = float(self.actor.decay_rate().item())
        stats["w_align"] = float(self.actor.w_align.item())
        stats["w_cohesion"] = float(self.actor.w_cohesion.item())
        stats["w_separation"] = float(self.actor.w_separation.item())
        return stats

    # ------------------------------------------------------------------
    def save(self, path):
        torch.save(dict(actor=self.actor.state_dict(), critic=self.critic.state_dict()), path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
