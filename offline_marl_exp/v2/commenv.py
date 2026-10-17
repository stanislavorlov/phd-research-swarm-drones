"""
commenv.py -- batched UAV-swarm communications environment (v2).

Fixes review items D1-D4:
  D3  real channel model: A2G-style path loss g(d)=(d^2+H^2)^(-alpha/2), SINR with aggregate
      co-channel interference from *all* UAVs, Shannon rate per user, strongest-signal association.
  D2  genuinely sequential: user clusters drift (and reflect off the boundary), so the optimal
      plan is a trajectory, not one constant target.
  D1  there is no hard interference cutoff: coupling is long-range, so a k-radius mask hides real
      reward-relevant information, and L(k) can be *measured* (see measure_Lk.py).
  D4  set-based, relative, normalised observations with explicit presence masks, and no fixed
      entity cap (the policy attends over every visible entity).

Action (per UAV, per step): displacement command a in R^2 (clipped to the unit ball, times vmax)
and a channel index. The recorded observation never contains the action.

Reward (team, shared): r_t = mean_u log2(1 + SINR_u) - energy * mean_i ||d_i||^2 after the move at step t.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import NamedTuple
import math
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EnvConfig:
    n_agents: int = 8
    n_users: int = 32
    n_channels: int = 3
    world: float = 20.0
    T: int = 20
    vmax: float = 1.0          # max UAV displacement per step
    H: float = 2.0             # UAV altitude (path-loss floor)
    alpha: float = 2.5         # path-loss exponent
    noise: float = 4e-4        # N0 / P  (P normalised to 1)
    n_clusters: int = 4
    cluster_std: float = 2.0
    user_speed: float = 0.4    # max cluster drift speed per step
    energy: float = 0.5        # propulsion-energy price per UAV per step: lambda * ||d||^2 (d in unit-ball units)
    k: float = float("inf")    # observation radius (actor)
    # opt-in (default False keeps every v2 result reproducible): add the UAV's own previous displacement to the
    # self features and the user position extrapolated EXTRAP_STEPS steps ahead to the user features.
    obs_extra: bool = False
    # opt-in (defaults reproduce every earlier result): k_a >= 0 gives the OTHER-UAV set its own radius (k then only
    # governs users); bcast=1 appends the per-channel count of ALL other UAVs (visible or not, / N) to the self features,
    # bcast=2 additionally appends the share of ALL users in 4 distance rings around the UAV (<5,5-10,10-15,>=15 world
    # units). Both are fixed-size, rotation-invariant global summaries (a cheap broadcast), not entity lists.
    k_a: float = -1.0
    bcast: int = 0

    def with_k(self, k):
        return replace(self, k=float(k))


# ----------------------------------------------------------------------------- physics
def link_rates(uav, users, ch, cfg: EnvConfig, interference_free: bool = False):
    """
    uav   [..., N, 2]   users [..., U, 2]   ch [..., N] (long)
    returns rate [..., U] (bit/s/Hz), serving UAV index [..., U]
    Differentiable in uav / users through the selected gains.
    """
    d2 = ((users.unsqueeze(-2) - uav.unsqueeze(-3)) ** 2).sum(-1)       # [..., U, N]
    g = (d2 + cfg.H ** 2) ** (-cfg.alpha / 2.0)
    serv = g.argmax(-1)                                                  # [..., U]
    S = g.gather(-1, serv.unsqueeze(-1)).squeeze(-1)
    if interference_free:
        I = torch.zeros_like(S)
    else:
        ch_s = ch.gather(-1, serv)                                       # [..., U]
        same = (ch.unsqueeze(-2) == ch_s.unsqueeze(-1)).to(g.dtype)      # [..., U, N]
        I = (g * same).sum(-1) - S
    return torch.log2(1.0 + S / (cfg.noise + I)), serv


def team_reward(uav, users, ch, cfg, interference_free=False):
    r, _ = link_rates(uav, users, ch, cfg, interference_free)
    return r.mean(-1)


def agent_rewards(uav, users, ch, cfg):
    """Per-agent decomposition r_i = (1/U) * sum of rates of users served by i. sum_i r_i == team reward."""
    r, serv = link_rates(uav, users, ch, cfg)
    onehot = F.one_hot(serv, cfg.n_agents).to(r.dtype)                  # [..., U, N]
    return (onehot * r.unsqueeze(-1)).sum(-2) / cfg.n_users


# ----------------------------------------------------------------------------- scenarios
def gen_scenario(seed: int, cfg: EnvConfig):
    rng = np.random.default_rng([int(seed), 20260101])
    W, U, N = cfg.world, cfg.n_users, cfg.n_agents
    centers = rng.uniform(0.15 * W, 0.85 * W, (cfg.n_clusters, 2))
    ang = rng.uniform(0, 2 * math.pi, cfg.n_clusters)
    sp = rng.uniform(0.0, cfg.user_speed, cfg.n_clusters)
    cvel = np.stack([np.cos(ang), np.sin(ang)], 1) * sp[:, None]
    assign = rng.integers(0, cfg.n_clusters, U)
    pos = np.clip(centers[assign] + rng.normal(0, cfg.cluster_std, (U, 2)), 0, W)
    vel = cvel[assign] + rng.normal(0, 0.05, (U, 2))
    traj = [pos.copy()]
    for _ in range(cfg.T):
        pos = pos + vel
        lo, hi = pos < 0, pos > W
        pos = np.where(lo, -pos, pos); pos = np.where(hi, 2 * W - pos, pos)
        vel = np.where(lo | hi, -vel, vel)
        traj.append(pos.copy())
    uvel = np.diff(np.stack(traj), axis=0)                  # displacement actually realised
    uvel = np.concatenate([uvel, uvel[-1:]], 0)             # [T+1, U, 2]
    uav0 = rng.uniform(0, W, (N, 2))
    ch0 = rng.integers(0, cfg.n_channels, N)
    return np.stack(traj), uvel, uav0, ch0


EXTRAP_STEPS = 4.0
RING_EDGES = (5.0, 10.0, 15.0)      # bcast=2 user-distance rings


class Obs(NamedTuple):
    sf: torch.Tensor   # [B,N,sd]      own pos/W, own channel one-hot, t/T
    uf: torch.Tensor   # [B,N,U,4]     user rel pos / (W/2), user velocity / vmax
    um: torch.Tensor   # [B,N,U] bool  user visible (within k)
    af: torch.Tensor   # [B,N,N,2+C]   other-UAV rel pos, channel one-hot
    am: torch.Tensor   # [B,N,N] bool  other UAV visible (within k, j != i)

    def flat(self):
        f = lambda x: x.reshape(-1, *x.shape[2:])
        return Obs(*[f(x) for x in self])

    def index(self, idx):
        return Obs(*[x[idx] for x in self])


def full_masks(obs: Obs):
    """Masks of an unmasked (k=inf) observer: every user, every other UAV. Works on [B,N,..] and flattened [B*N,..]."""
    um = torch.ones_like(obs.um)
    N = obs.am.shape[-1]
    am = torch.ones_like(obs.am)
    if obs.am.dim() == 3:
        am = am & ~torch.eye(N, dtype=torch.bool, device=am.device)
    else:                                   # flattened rows: row r belongs to agent r % N
        am[torch.arange(am.shape[0]), torch.arange(am.shape[0]) % N] = False
    return um, am


class CommEnv:
    """B scenarios stepped in parallel. Scenario b is a pure function of seeds[b]."""

    def __init__(self, cfg: EnvConfig, seeds):
        self.cfg = cfg
        self.seeds = list(map(int, seeds))
        sc = [gen_scenario(s, cfg) for s in self.seeds]
        f = lambda i, dt=torch.float32: torch.tensor(np.stack([s[i] for s in sc]), dtype=dt)
        self.users_traj = f(0)            # [B,T+1,U,2]
        self.user_vel = f(1)
        self.uav0 = f(2)
        self.ch0 = torch.tensor(np.stack([s[3] for s in sc]), dtype=torch.long)
        self.B = len(self.seeds)
        self.reset()

    def reset(self):
        self.uav = self.uav0.clone()
        self.ch = self.ch0.clone()
        self.t = 0
        self.prev = torch.zeros(self.B, self.cfg.n_agents, 2)               # last executed (clipped) displacement
        return self.obs(self.cfg.k)

    # --------------------------------------------------------------- observation
    def obs(self, k: float) -> Obs:
        cfg, B, N = self.cfg, self.B, self.cfg.n_agents
        t = min(self.t, cfg.T)
        users = self.users_traj[:, t]
        vel = self.user_vel[:, t]
        sc = cfg.world / 2.0
        rel_u = users.unsqueeze(1) - self.uav.unsqueeze(2)                # [B,N,U,2]
        um = rel_u.norm(dim=-1) <= k
        uf = torch.cat([rel_u / sc, (vel / cfg.vmax).unsqueeze(1).expand(-1, N, -1, -1)], -1)
        oh = F.one_hot(self.ch, cfg.n_channels).float()                   # [B,N,C]
        rel_a = self.uav.unsqueeze(1) - self.uav.unsqueeze(2)             # [B,N,N,2] (i,j)=pos_j-pos_i
        eye = torch.eye(N, dtype=torch.bool)
        ka = k if cfg.k_a < 0 else cfg.k_a
        am = (rel_a.norm(dim=-1) <= ka) & ~eye
        af = torch.cat([rel_a / sc, oh.unsqueeze(1).expand(-1, N, -1, -1)], -1)
        sf = torch.cat([self.uav / cfg.world, oh, torch.full((B, N, 1), t / cfg.T)], -1)
        # column order of sf: [pos(2), channel one-hot(C), t/T, bcast...(C | 4), prev displacement(2) if obs_extra]
        if cfg.bcast >= 1:
            sf = torch.cat([sf, (oh.sum(1, keepdim=True) - oh) / N], -1)             # other UAVs per channel, all of them
        if cfg.bcast >= 2:
            dist = rel_u.norm(dim=-1)                                                 # [B,N,U]
            e = RING_EDGES
            rings = torch.stack([dist < e[0], (dist >= e[0]) & (dist < e[1]),
                                 (dist >= e[1]) & (dist < e[2]), dist >= e[2]], -1).float()
            sf = torch.cat([sf, rings.sum(2) / cfg.n_users], -1)                      # [B,N,4], sums to 1
        if cfg.obs_extra:
            uf = torch.cat([uf, (rel_u + EXTRAP_STEPS * vel.unsqueeze(1)) / sc], -1)    # rel. pos. 4 steps ahead
            sf = torch.cat([sf, self.prev], -1)                                           # own previous displacement
        return Obs(sf, uf, um, af, am)

    # --------------------------------------------------------------- dynamics
    def step(self, disp, ch_new):
        """disp [B,N,2] in unit-ball units (norm-clipped here), ch_new [B,N] long."""
        cfg = self.cfg
        n = disp.norm(dim=-1, keepdim=True)
        disp = disp / n.clamp(min=1.0)
        self.prev = disp.clone()
        self.uav = (self.uav + cfg.vmax * disp).clamp(0.0, cfg.world)
        self.ch = ch_new.clamp(0, cfg.n_channels - 1).long()
        self.t += 1
        users = self.users_traj[:, self.t]
        rate, _ = link_rates(self.uav, users, self.ch, cfg)
        en = (disp ** 2).sum(-1)                                           # [B,N]
        r = rate.mean(-1) - cfg.energy * en.mean(-1)                       # [B]
        info = dict(local=agent_rewards(self.uav, users, self.ch, cfg) - cfg.energy * en / cfg.n_agents,
                    rate=rate.mean(-1),
                    outage=(rate < 1.0).float().mean(-1))                  # SINR < 1 (rate<1 bit/s/Hz)
        return self.obs(cfg.k), r, self.t >= cfg.T, info


# ----------------------------------------------------------------------------- reference policies
class StayPut:
    name = "stay_put"
    def act(self, env, obs):
        return torch.zeros(env.B, env.cfg.n_agents, 2), env.ch.clone()


class RandomPolicy:
    name = "random"
    def __init__(self, seed=0):
        self.g = torch.Generator().manual_seed(seed)
    def act(self, env, obs):
        B, N = env.B, env.cfg.n_agents
        ang = torch.rand(B, N, generator=self.g) * 2 * math.pi
        rad = torch.rand(B, N, generator=self.g).sqrt()
        d = torch.stack([ang.cos(), ang.sin()], -1) * rad.unsqueeze(-1)
        return d, torch.randint(0, env.cfg.n_channels, (B, N), generator=self.g)


def run_episode(env: CommEnv, policy):
    """Deterministic evaluation rollout. Returns per-scenario return [B] (sum of team reward)."""
    obs = env.reset()
    ret = torch.zeros(env.B)
    outage = torch.zeros(env.B)
    while True:
        disp, ch = policy.act(env, obs)
        obs, r, done, info = env.step(disp, ch)
        if hasattr(policy, "observe_reward"):
            policy.observe_reward(r)
        ret += r; outage += info["outage"] / env.cfg.T
        if done:
            return ret, outage
