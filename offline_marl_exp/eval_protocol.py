"""
eval_protocol.py — paired evaluation + training sanity gates for the
range-bounded asymmetric distillation pipeline.

Two things live here:

  sanity_gates(...)  cheap assertions that must pass BEFORE any comparison is
                     plotted. Every one of them fails on the pipeline as
                     shipped, which is why the comparison plot was noise.

  paired_eval(...)   scenario-paired evaluation (common random numbers) with
                     bootstrap CIs on the paired difference and expert-
                     normalised scores. Use this, not per-epoch training
                     reward, for every number that goes in the paper.

Policies are plain callables: f(obs, env) -> (pos_actions, channel_actions).
"""
from __future__ import annotations
import copy
import numpy as np
import torch

from env import DroneSwarmEnv


# ----------------------------------------------------------------- policies
def make_net_policy(actor, deterministic: bool = True):
    def f(obs, env):
        with torch.no_grad():
            pos, logits = actor(torch.tensor(obs, dtype=torch.float32))
        ch = (torch.argmax(logits, 1) if deterministic
              else torch.distributions.Categorical(logits=logits).sample())
        return pos.numpy(), ch.numpy()
    return f


def stay_put_policy(obs, env):
    return env.positions.copy(), env.channels.copy()


def make_random_policy(seed=0):
    rng = np.random.default_rng(seed)
    def f(obs, env):
        return (rng.uniform(0, env.world_size, size=(env.n_agents, 2)),
                rng.integers(0, env.n_channels, env.n_agents))
    return f


# ----------------------------------------------------------------- rollouts
def rollout(policy, seed, env_kwargs=None):
    """One episode on the scenario identified by `seed`. Same seed == same
    users, same initial positions, for every arm: that is the pairing."""
    env = DroneSwarmEnv(seed=seed, **(env_kwargs or {}))
    obs = env.reset()
    total, cov, interf = 0.0, 0.0, 0.0
    while True:
        pos, ch = policy(obs, env)
        obs, r, done, info = env.step(np.asarray(pos, dtype=float),
                                      np.asarray(ch, dtype=int))
        total += float(np.mean(r))
        cov += info["coverage_cost"]; interf += info["interference_penalty"]
        if done:
            return dict(ret=total, coverage=cov, interference=interf)


def paired_eval(arms: dict, seeds, env_kwargs=None, reference=None,
                expert_returns=None, n_boot=10000, rng_seed=0):
    """
    arms       : {name: policy}
    seeds      : iterable of scenario seeds, SHARED by every arm
    reference  : arm name to difference against (default: first key)
    expert_returns : optional {seed: return} for expert-normalised scores
    """
    seeds = list(seeds)
    names = list(arms)
    reference = reference or names[0]
    R = {n: np.array([rollout(arms[n], s, env_kwargs)["ret"] for s in seeds])
         for n in names}
    rng = np.random.default_rng(rng_seed)
    out = {}
    for n in names:
        d = R[n] - R[reference]
        idx = rng.integers(0, len(seeds), size=(n_boot, len(seeds)))
        boot = d[idx].mean(1)
        row = dict(mean=R[n].mean(), sd=R[n].std(ddof=1), n=len(seeds),
                   paired_delta=d.mean(),
                   ci95=(float(np.percentile(boot, 2.5)),
                         float(np.percentile(boot, 97.5))))
        if expert_returns is not None:
            e = np.array([expert_returns[s] for s in seeds])
            base = np.array([rollout(stay_put_policy, s, env_kwargs)["ret"] for s in seeds])
            row["expert_normalised"] = float(np.mean((R[n] - base) / (e - base)))
        out[n] = row
    return out, R


def min_detectable_effect(sd, n, paired_sd=None, alpha=0.05, power=0.8):
    """Smallest difference in return units that the design can resolve."""
    z = (1.959964 + 0.8416212) ** 2
    return (np.sqrt(z * paired_sd ** 2 / n) if paired_sd is not None
            else np.sqrt(z * 2 * sd ** 2 / n))


# ------------------------------------------------------------ sanity gates
def sanity_gates(actor, critic, update_step, env_kwargs=None, n_updates=20,
                 floors=("stay_put", "random"), seeds=range(64), verbose=True):
    """
    update_step(actor, critic, env) -> dict with at least {'actor_loss'}.
    Raises AssertionError on the first failed gate. Run this in CI.
    """
    env_kwargs = env_kwargs or {}
    report = {}

    # G1 every actor head receives gradient
    env = DroneSwarmEnv(seed=0, **env_kwargs)
    env.reset()
    actor.zero_grad(set_to_none=True)
    info = update_step(actor, critic, env)
    missing = [n for n, p in actor.named_parameters() if p.grad is None]
    gnorm = sum(float((p.grad ** 2).sum()) for _, p in actor.named_parameters()
                if p.grad is not None) ** 0.5
    report["G1_grad_norm"] = gnorm
    report["G1_params_without_grad"] = missing
    assert not missing, f"G1 FAILED: no gradient reaches {missing}"
    assert gnorm > 1e-8, f"G1 FAILED: actor gradient norm is {gnorm:.2e} (loss is constant in the parameters)"

    # G2 weights actually move
    before = torch.cat([p.flatten() for p in actor.parameters()]).detach().clone()
    env = DroneSwarmEnv(seed=1, **env_kwargs); env.reset()
    for _ in range(n_updates):
        update_step(actor, critic, env)
    after = torch.cat([p.flatten() for p in actor.parameters()]).detach()
    delta = float((after - before).abs().max())
    report["G2_max_weight_change"] = delta
    assert delta > 1e-8, "G2 FAILED: policy weights unchanged after fine-tuning"

    # G3 the policy clears the trivial floors
    pol = make_net_policy(actor)
    r_pol = np.array([rollout(pol, s, env_kwargs)["ret"] for s in seeds])
    for name, f in (("stay_put", stay_put_policy), ("random", make_random_policy())):
        if name not in floors:
            continue
        r_f = np.array([rollout(f, s, env_kwargs)["ret"] for s in seeds])
        d = r_pol - r_f
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
        report[f"G3_vs_{name}"] = dict(delta=float(d.mean()), t=float(t))
        assert d.mean() > 0, (f"G3 FAILED: policy loses to '{name}' by "
                              f"{-d.mean():.0f} return (t={t:.1f})")

    # G4 arms that are supposed to differ are not the same network
    report["G4_note"] = ("compare state_dicts of every pair of arms after training; "
                         "identical tensors mean the ablation is not an ablation")
    if verbose:
        for k, v in report.items():
            print(f"  {k}: {v}")
    return report


def arms_are_distinct(arm_a, arm_b, tol=1e-10):
    sa, sb = arm_a.state_dict(), arm_b.state_dict()
    return any(float((sa[k] - sb[k]).abs().max()) > tol for k in sa)
