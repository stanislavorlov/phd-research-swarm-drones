"""
models.py -- set-based actor and critic.

Both use the SAME encoder class (self token queries attention over variable-length user and
neighbour sets, plus a DeepSets sum-pool so entity *counts* survive). The only thing that makes a
critic "masked" or "unmasked" is the attention mask it is handed (commenv.Obs masks at radius k vs
commenv.full_masks): same architecture, same parameters, same initialisation under the same seed.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from commenv import Obs, EnvConfig


class SetEncoder(nn.Module):
    def __init__(self, sd, ud, ad, d=64, heads=4):
        super().__init__()
        mlp = lambda i: nn.Sequential(nn.Linear(i, d), nn.ReLU(), nn.Linear(d, d))
        self.self_mlp, self.user_mlp, self.uav_mlp = mlp(sd), mlp(ud), mlp(ad)
        self.null = nn.Parameter(torch.zeros(1, 1, d))        # always-visible token: no empty softmax
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.out = nn.Sequential(nn.Linear(4 * d, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.d = d

    def forward(self, o: Obs, um=None, am=None):
        um = o.um if um is None else um
        am = o.am if am is None else am
        M = o.sf.shape[0]
        q = self.self_mlp(o.sf)    # [M,d]
        eu, ea = self.user_mlp(o.uf), self.uav_mlp(o.af)                     # [M,U,d], [M,N,d]
        keys = torch.cat([self.null.expand(M, 1, -1), eu, ea], 1)
        vis = torch.cat([torch.ones(M, 1, dtype=torch.bool, device=um.device), um, am], 1)
        ctx, _ = self.attn(q.unsqueeze(1), keys, keys, key_padding_mask=~vis)
        su = (eu * um.unsqueeze(-1)).sum(1) / 8.0
        sa = (ea * am.unsqueeze(-1)).sum(1) / 4.0
        return self.out(torch.cat([q, ctx.squeeze(1), su, sa], -1))


def dims(cfg: EnvConfig):
    ex = bool(getattr(cfg, "obs_extra", False))
    bc = int(getattr(cfg, "bcast", 0))
    # bcast 1/2 append a per-channel COUNT, 3/4/5 a per-channel sensed POWER -- both C wide; only 2 adds the 4 user rings
    sd = (2 + cfg.n_channels + 1 + (cfg.n_channels if bc >= 1 else 0) + (cfg.n_channels if bc == 6 else 0)
          + (4 if bc == 2 else 0) + (2 if ex else 0))
    return sd, 4 + (2 if ex else 0), 2 + cfg.n_channels


class Actor(nn.Module):
    """pi(a | o^k): Gaussian displacement (state-independent log-std) + categorical channel."""
    def __init__(self, cfg: EnvConfig, d=64, log_std=math.log(0.2)):
        super().__init__()
        self.enc = SetEncoder(*dims(cfg), d=d)
        self.pos_head = nn.Linear(d, 2)
        self.ch_head = nn.Linear(d, cfg.n_channels)
        self.log_std = nn.Parameter(torch.full((2,), float(log_std)))
        nn.init.zeros_(self.pos_head.weight); nn.init.zeros_(self.pos_head.bias)

    def forward(self, o: Obs):
        h = self.enc(o)
        return self.pos_head(h), self.ch_head(h)

    def dist(self, o: Obs):
        mu, logits = self(o)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu)), \
               torch.distributions.Categorical(logits=logits)

    def log_prob_entropy(self, o, a, c):
        dn, dc = self.dist(o)
        lp = dn.log_prob(a).sum(-1) + dc.log_prob(c)
        ent = dn.entropy().sum(-1) + dc.entropy()
        return lp, ent


class StudentActor(Actor):
    """
    Actor with two BC-only options; everything the online phase uses is inherited unchanged.

      mdn_k > 0   position head is a mixture of mdn_k diagonal Gaussians trained with NLL (so a multimodal teacher is
                  not averaged into a hovering UAV). forward() returns the mean of the most probable component, so
                  Actor.dist / log_prob_entropy / NetPolicy / MAPPO work unchanged (Gaussian around the mode).
      chunk  > 1  auxiliary head predicting the next chunk-1 displacements of the plan (action chunking as a
                  representation-shaping loss; never executed).
    """
    def __init__(self, cfg: EnvConfig, d=64, mdn_k=0, chunk=1, log_std=math.log(0.2)):
        super().__init__(cfg, d=d, log_std=log_std)
        self.mdn_k, self.chunk = int(mdn_k), int(chunk)
        if self.mdn_k > 0:
            self.mdn = nn.Linear(d, self.mdn_k * 5)       # per component: mu(2), log-std(2), mixture logit(1)
        if self.chunk > 1:
            self.chunk_head = nn.Linear(d, 2 * (self.chunk - 1))

    def _mdn(self, h):
        p = self.mdn(h).view(h.shape[0], self.mdn_k, 5)
        return p[..., 0:2], p[..., 2:4].clamp(-4.0, 1.0), torch.log_softmax(p[..., 4], -1)

    def forward(self, o: Obs):
        h = self.enc(o)
        if self.mdn_k > 0:
            mu, _, lw = self._mdn(h)
            pos = mu[torch.arange(h.shape[0], device=h.device), lw.argmax(-1)]
        else:
            pos = self.pos_head(h)
        return pos, self.ch_head(h)

    def bc_loss(self, o: Obs, a_chunk, mask, c, w_chunk=0.5, w_ch=1.0):
        """a_chunk [n,H,2] expert displacements for steps t..t+H-1, mask [n,H] valid, c [n] expert channel."""
        h = self.enc(o)
        a0 = a_chunk[:, 0]
        if self.mdn_k > 0:
            mu, ls, lw = self._mdn(h)
            z = (a0.unsqueeze(1) - mu) / ls.exp()
            logp = lw - 0.5 * (z ** 2).sum(-1) - ls.sum(-1) - math.log(2 * math.pi)
            pos_loss = -torch.logsumexp(logp, -1).mean()
            mode = mu[torch.arange(h.shape[0], device=h.device), lw.argmax(-1)]
        else:
            mode = self.pos_head(h)
            pos_loss = F.mse_loss(mode, a0)
        logits = self.ch_head(h)
        loss = pos_loss + w_ch * F.cross_entropy(logits, c)
        if self.chunk > 1:
            pred = self.chunk_head(h).view(-1, self.chunk - 1, 2)
            m = mask[:, 1:self.chunk].float().unsqueeze(-1)
            loss = loss + w_chunk * (((pred - a_chunk[:, 1:self.chunk]) ** 2) * m).sum() / (2 * m.sum() + 1e-6)
        stats = dict(mse0=float(F.mse_loss(mode, a0).detach()), acc=float((logits.argmax(-1) == c).float().mean().detach()))
        return loss, stats


class Critic(nn.Module):
    def __init__(self, cfg: EnvConfig, d=64):
        super().__init__()
        self.enc = SetEncoder(*dims(cfg), d=d)
        self.head = nn.Linear(d, 1)

    def forward(self, o: Obs, masked: bool, um=None, am=None):
        from commenv import full_masks
        if not masked:
            um, am = full_masks(o)
        return self.head(self.enc(o, um, am)).squeeze(-1)


class NetPolicy:
    """Deterministic evaluation policy wrapper (mean displacement, argmax channel)."""
    def __init__(self, actor: Actor, name="net"):
        self.actor, self.name = actor, name
    @torch.no_grad()
    def act(self, env, obs: Obs):
        B, N = env.B, env.cfg.n_agents
        mu, logits = self.actor(obs.flat())
        return mu.view(B, N, 2), logits.argmax(-1).view(B, N)
