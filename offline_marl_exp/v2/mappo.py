"""
mappo.py -- MAPPO (parameter-shared actor, clipped-ratio PPO) for the online phase.

What the v1 "fine-tuning" lacked (review F1, S1) and this has:
  * actions are SAMPLED from a stochastic policy (Gaussian displacement + categorical channel) and their
    log-probs under the behaviour policy are stored; the surrogate uses the ratio exp(logp_new - logp_old),
    which is exactly 1 on the first pass and gives a non-zero gradient to EVERY head (incl. the channel head
    and log-std);
  * clipped surrogate (eps=0.2) + approx-KL early stopping = the trust region; entropy bonus;
  * GAE(gamma, lambda) advantages from a learned value function (not the immediate reward);
  * minibatching, several epochs per batch, gradient clipping;
  * the critic is a masked or unmasked attention critic. The two differ in NOTHING but the attention mask
    (same class, same init under the same seed). That mask is the single factor of the ablation.

Not HARL: this is a compact in-repo MAPPO validated by the sanity gates in tests/test_gates.py. The review
recommends HARL (arXiv:2304.09870) for the final paper numbers; the env/actor/critic here are written so the
update step is the only thing to swap.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from commenv import CommEnv, EnvConfig, Obs, run_episode
from models import Actor, Critic, NetPolicy
from evalproto import returns


@dataclass
class PPOConfig:
    n_envs: int = 32
    iters: int = 150
    epochs: int = 3
    minibatches: int = 2
    clip: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    ent_coef: float = 1e-3
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    max_grad: float = 0.5
    target_kl: float = 0.03
    critic_warmup: int = 15          # critic-only iterations (actor frozen), identical for every arm
    reward_scale: float = 0.05
    eval_every: int = 15
    critic_d: int = 32
    # --- coverage constraint (section 5): subtract a penalty from the TRAINING reward only. Evaluation, the logged
    # ep_return and every held-out number stay on the original objective, so constrained and unconstrained runs are
    # directly comparable. Both default to 0.0, which reproduces every earlier run exactly.
    outage_coef: float = 0.0        # penalty per unit of outage fraction (users below 1 bit/s/Hz)
    shortfall_coef: float = 0.0     # penalty per unit of mean rate shortfall below EnvConfig.rate_floor


def cat_obs(lst):
    return Obs(*[torch.cat([getattr(o, f) for o in lst], 0) for f in Obs._fields])


@torch.no_grad()
def collect(cfg: EnvConfig, seeds, actor: Actor, critic: Critic, masked: bool, ppo: PPOConfig):
    env = CommEnv(cfg, seeds)
    obs = env.reset()
    B, N, T = env.B, cfg.n_agents, cfg.T
    O, A, Cc, LP, V, R, EN = [], [], [], [], [], [], []
    R_true, PEN = [], []
    for t in range(T):
        o = obs.flat()
        dn, dc = actor.dist(o)
        a, c = dn.sample(), dc.sample()
        lp = dn.log_prob(a).sum(-1) + dc.log_prob(c)
        v = critic(o, masked)
        obs, r, done, info = env.step(a.view(B, N, 2), c.view(B, N))
        pen = ppo.outage_coef * info["outage"] + ppo.shortfall_coef * info["shortfall"]      # [B], 0 by default
        O.append(o); A.append(a); Cc.append(c); LP.append(lp); V.append(v)
        R.append((r - pen).unsqueeze(1).expand(B, N).reshape(-1) * ppo.reward_scale)
        R_true.append(r); PEN.append(pen)
    V.append(torch.zeros_like(V[0]))                                   # finite horizon, t/T in obs -> V(T)=0
    adv = torch.zeros(T, B * N)
    last = torch.zeros(B * N)
    for t in reversed(range(T)):
        delta = R[t] + ppo.gamma * V[t + 1] - V[t]
        last = delta + ppo.gamma * ppo.lam * last
        adv[t] = last
    ret = adv + torch.stack(V[:-1])
    batch = dict(o=cat_obs(O), a=torch.cat(A), c=torch.cat(Cc), lp=torch.cat(LP),
                 adv=adv.reshape(-1), ret=ret.reshape(-1))
    batch["ep_return"] = float(torch.stack(R_true).sum(0).mean())          # the TRUE objective, never the penalised one
    batch["ep_penalty"] = float(torch.stack(PEN).sum(0).mean())
    return batch


def ppo_update(actor, critic, masked, batch, a_opt, c_opt, ppo: PPOConfig, update_actor=True, gen=None):
    """One PPO update on a collected batch. Returns diagnostics (used by the sanity gates)."""
    n = batch["a"].shape[0]
    adv = (batch["adv"] - batch["adv"].mean()) / (batch["adv"].std() + 1e-8)
    stats = dict(clipfrac=[], kl=[], ratio_min=[], ratio_max=[], actor_loss=[], critic_loss=[], first_ratio_dev=None,
                 gnorm_actor=[], gnorm_critic=[], stopped_early=False)
    mb = (n + ppo.minibatches - 1) // ppo.minibatches
    first = True
    for ep in range(ppo.epochs):
        perm = torch.randperm(n, generator=gen)
        ep_kl = []
        for i in range(0, n, mb):
            idx = perm[i:i + mb]
            o = batch["o"].index(idx)
            v = critic(o, masked)
            closs = 0.5 * ((v - batch["ret"][idx]) ** 2).mean()
            c_opt.zero_grad(); closs.backward()
            gc = nn.utils.clip_grad_norm_(critic.parameters(), ppo.max_grad); c_opt.step()
            stats["critic_loss"].append(closs.item()); stats["gnorm_critic"].append(float(gc))
            if update_actor:
                lp, ent = actor.log_prob_entropy(o, batch["a"][idx], batch["c"][idx])
                ratio = (lp - batch["lp"][idx]).exp()
                if first:
                    stats["first_ratio_dev"] = float((ratio - 1).abs().max().detach()); first = False
                a_i = adv[idx]
                surr = torch.min(ratio * a_i, ratio.clamp(1 - ppo.clip, 1 + ppo.clip) * a_i)
                aloss = -surr.mean() - ppo.ent_coef * ent.mean()
                a_opt.zero_grad(); aloss.backward()
                ga = nn.utils.clip_grad_norm_(actor.parameters(), ppo.max_grad); a_opt.step()
                with torch.no_grad():
                    kl = (batch["lp"][idx] - lp).mean().item()
                stats["actor_loss"].append(aloss.item()); stats["gnorm_actor"].append(float(ga))
                stats["clipfrac"].append(((ratio - 1).abs() > ppo.clip).float().mean().item())
                stats["kl"].append(kl); ep_kl.append(kl)
                stats["ratio_min"].append(ratio.min().item()); stats["ratio_max"].append(ratio.max().item())
        if update_actor and ep_kl and np.mean(ep_kl) > 1.5 * ppo.target_kl:     # trust-region early stop
            stats["stopped_early"] = True
            break
    return {k: ((float(np.mean(v)) if v else None) if isinstance(v, list) else v) for k, v in stats.items()}


def train_mappo(cfg: EnvConfig, actor: Actor, masked_critic: bool, run_seed: int, ppo: PPOConfig,
                eval_seeds, train_seed_base=1_000_000, label="", verbose=True, log=None):
    """Returns history dict: iteration -> eval returns on the (paired) held-out seeds."""
    torch.manual_seed(run_seed)
    critic_gen = torch.Generator().manual_seed(1000 + run_seed)
    torch.manual_seed(1000 + run_seed); critic = Critic(cfg, d=ppo.critic_d)   # identical init for masked/unmasked arms
    torch.manual_seed(run_seed)
    a_opt = torch.optim.Adam(actor.parameters(), lr=ppo.lr_actor)
    c_opt = torch.optim.Adam(critic.parameters(), lr=ppo.lr_critic)
    hist = {}
    ev = lambda: returns(NetPolicy(actor), eval_seeds, cfg)                 # (return, outage) on the ORIGINAL objective
    hist[0], out0 = ev()
    t0 = time.time()
    diag = []
    for it in range(1, ppo.iters + 1):
        seeds = [train_seed_base + run_seed * 100_000 + it * ppo.n_envs + j for j in range(ppo.n_envs)]
        batch = collect(cfg, seeds, actor, critic, masked_critic, ppo)
        st = ppo_update(actor, critic, masked_critic, batch, a_opt, c_opt, ppo, update_actor=(it > ppo.critic_warmup))
        st["train_return"] = batch["ep_return"]; st["train_penalty"] = batch["ep_penalty"]; st["it"] = it
        diag.append(st)
        if it % ppo.eval_every == 0 or it == ppo.iters:
            hist[it], o_ev = ev()
            st["eval_outage"] = float(o_ev.mean())
            if verbose:
                msg = (f"[{label}] it {it:4d} eval {hist[it].mean():7.2f} out {st['eval_outage']:.4f}  "
                       f"train {batch['ep_return']:7.2f}  "
                       f"kl {st['kl'] if st['kl'] is not None else float('nan'):.4f}  clipfrac {st['clipfrac'] if st['clipfrac'] is not None else float('nan'):.3f}  "
                       f"logstd {actor.log_std.mean().item():+.2f}  ({time.time()-t0:.0f}s)")
                print(msg, flush=True)
                if log: log.write(msg + "\n"); log.flush()
    return hist, diag, critic
