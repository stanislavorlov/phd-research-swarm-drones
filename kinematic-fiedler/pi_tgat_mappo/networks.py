"""
PI-TGAT actor (shared across all agents -- MAPPO parameter sharing) and a
centralized critic, both in PyTorch + torch_geometric.

PI-TGAT actor pipeline, matching the manuscript's Methods section:
  1. Combine each edge's raw features into ONE effective (relative position,
     relative velocity) pair:
       - real edges: use the true relative pos/vel directly.
       - synthetic edges: combine the raw Reynolds alignment/cohesion/
         separation components with LEARNABLE weights w_align, w_cohesion,
         w_separation (paper: "learnable scalar parameters embedded
         directly in the PI-TGAT actor network... initialized to the
         standard flocking heuristic").
  2. Apply a LEARNABLE exponential temporal decay to that effective vector,
     governed by a learnable decay rate and the edge's time-since-last-
     packet (paper: "decays attention weights exponentially over time...
     applied directly to this cached state before any spatial aggregation
     occurs"). Real edges have dt_since = 0, so decay = 1 automatically.
  3. Feed the (now decayed) edge features + node features through a 2-layer,
     4-head Graph Attention network (GATv2Conv, which natively supports
     edge features in its attention logits).
  4. Pass the resulting spatial message through a GRUCell to update each
     agent's persistent temporal memory (paper: "GRU maintains agent's
     internal memory across communication blackouts").
  5. A Gaussian policy head outputs the 3D target velocity (tanh-squashed to
     [-1, 1], scaled by cfg.v_max downstream in the environment).

The centralized critic is a plain MLP over the concatenated global state
(every agent's node features, padded/masked to a fixed max agent count) --
it is only used during training and discarded at deployment, exactly as
the paper describes for CTDE.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, MessagePassing

from .config import Config
from .env import EDGE_FEAT_DIM, NODE_FEAT_DIM


class IsotropicConv(MessagePassing):
    """DGN-style isotropic mean-aggregation graph conv (no learned attention
    weights, every neighbor contributes equally) — a deliberately simple
    proxy for Jiang et al. 2020's DGN, not a literal reproduction."""

    def __init__(self, in_channels, out_channels, edge_dim):
        super().__init__(aggr="mean")
        self.lin_msg = nn.Linear(in_channels + edge_dim, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return F.relu(out + self.lin_self(x))

    def message(self, x_j, edge_attr):
        return self.lin_msg(torch.cat([x_j, edge_attr], dim=-1))

class PITGATActor(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        heads = cfg.gat_heads
        per_head = cfg.embed_dim // heads
        assert per_head * heads == cfg.embed_dim, "embed_dim must be divisible by gat_heads"

        # Learnable kinematic-prior weights (paper: w1=align, w2=cohesion, w3=separation).
        w1, w2, w3 = cfg.kinematic_prior_init
        self.w_align = nn.Parameter(torch.tensor(float(w1)))
        self.w_cohesion = nn.Parameter(torch.tensor(float(w2)))
        self.w_separation = nn.Parameter(torch.tensor(float(w3)))
        # Learnable temporal decay rate (paper: lambda, init 0.1). Softplus keeps it positive.
        self._decay_raw = nn.Parameter(torch.tensor(float(cfg.decay_rate_init)))

        self.input_proj = nn.Linear(NODE_FEAT_DIM, cfg.embed_dim)
        self.use_graph = cfg.use_graph
        self.aggregator = cfg.aggregator
        if self.use_graph:
            if self.aggregator == "attention":
                self.gat_layers = nn.ModuleList([
                    GATv2Conv(cfg.embed_dim, per_head, heads=cfg.gat_heads, edge_dim=8, concat=True)
                    for _ in range(cfg.gat_layers)
                ])
            elif self.aggregator == "conv":
                self.gat_layers = nn.ModuleList([
                    IsotropicConv(cfg.embed_dim, cfg.embed_dim, edge_dim=8)
                    for _ in range(cfg.gat_layers)
                ])
            else:
                raise ValueError(f"Unknown aggregator: {self.aggregator!r}")
        else:
            self.gat_layers = None  # Vanilla MAPPO: no cross-agent communication at all
        self.gru = nn.GRUCell(cfg.embed_dim, cfg.gru_hidden)
        self.policy_mean = nn.Linear(cfg.gru_hidden, 3)
        self.log_std = nn.Parameter(torch.zeros(3) - 0.5)  # state-independent std, std(0) ~= 0.6

    def decay_rate(self):
        return F.softplus(self._decay_raw)

    def _effective_edge_features(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Collapses the 17-dim raw edge_attr from env.py into the 8-dim
        (rel_pos, rel_vel, dt_since, is_synthetic) tensor GATv2Conv consumes,
        applying the learnable kinematic-prior weights + temporal decay."""
        rel_pos_real = edge_attr[:, 0:3]
        rel_vel_real = edge_attr[:, 3:6]
        align = edge_attr[:, 6:9]
        cohesion = edge_attr[:, 9:12]
        separation = edge_attr[:, 12:15]
        dt_since = edge_attr[:, 15:16]
        is_synth = edge_attr[:, 16:17]

        synth_vel = self.w_align * align + self.w_cohesion * cohesion + self.w_separation * separation
        synth_pos = synth_vel * self.cfg.dt  # small position-delta proxy, see env.py docstring

        rel_pos = torch.where(is_synth.bool(), synth_pos, rel_pos_real)
        rel_vel = torch.where(is_synth.bool(), synth_vel, rel_vel_real)

        decay = torch.exp(-self.decay_rate() * dt_since)  # == 1 for real edges (dt_since == 0)
        rel_pos = rel_pos * decay
        rel_vel = rel_vel * decay

        return torch.cat([rel_pos, rel_vel, dt_since, is_synth], dim=1)

    def forward(self, node_feats, edge_index, edge_attr, hidden):
        """
        node_feats: (n, NODE_FEAT_DIM)
        edge_index: (2, E) long
        edge_attr:  (E, EDGE_FEAT_DIM) raw, as produced by env.py
        hidden:     (n, gru_hidden) previous GRU state for these n agents
        Returns: action_mean (n, 3), log_std (3,), new_hidden (n, gru_hidden)
        """
        x = F.elu(self.input_proj(node_feats))
        if self.use_graph:
            eff_edge_attr = self._effective_edge_features(edge_attr)
            for i, layer in enumerate(self.gat_layers):
                x = layer(x, edge_index, edge_attr=eff_edge_attr)
                if i < len(self.gat_layers) - 1:
                    x = F.elu(x)
        # else: x is the raw per-agent embedding, untouched by any neighbor --
        # this is what makes use_graph=False equivalent to Vanilla MAPPO
        # (no cross-agent communication at all).
        spatial_message = x  # (n, embed_dim)

        new_hidden = self.gru(spatial_message, hidden)
        # Raw (unbounded) Gaussian mean; the sampled action is clipped to
        # [-1, 1] downstream in env.py rather than tanh-squashed here, to avoid
        # the tanh log-prob (change-of-variables) correction a squashed
        # Gaussian would otherwise need for a correct PPO ratio.
        mean = self.policy_mean(new_hidden)
        return mean, self.log_std, new_hidden

    def init_hidden(self, n_agents, device):
        return torch.zeros(n_agents, self.cfg.gru_hidden, device=device)


class CentralizedCritic(nn.Module):
    """A permutation-invariant critic: mean-pools per-agent embeddings of the
    FULL global state (every agent's node features) through a small MLP,
    then a value head -- this naturally handles the paper's variable swarm
    size N without needing a fixed-size global state vector."""

    def __init__(self, cfg: Config, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(NODE_FEAT_DIM, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, all_node_feats: torch.Tensor) -> torch.Tensor:
        """all_node_feats: (n, NODE_FEAT_DIM) for the CURRENT episode's swarm."""
        per_agent = self.encoder(all_node_feats)
        pooled = per_agent.mean(dim=0, keepdim=True)
        return self.value_head(pooled).squeeze(-1)  # scalar V(s)
