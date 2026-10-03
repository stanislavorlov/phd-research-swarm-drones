"""
evalproto.py -- paired evaluation (common random numbers), bootstrap CIs, expert-normalised score.

Every arm is rolled out on the identical seed set. All numbers that go in a paper come from here,
never from training reward.
"""
from __future__ import annotations
import numpy as np
import torch

from commenv import CommEnv, EnvConfig, StayPut, RandomPolicy, run_episode


def returns(policy, seeds, cfg: EnvConfig):
    ret, outage = run_episode(CommEnv(cfg, seeds), policy)
    return ret.numpy(), outage.numpy()


def bootstrap_ci(x, n_boot=10000, seed=0, q=(2.5, 97.5)):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    b = x[idx].mean(1)
    return tuple(float(np.percentile(b, p)) for p in q)


def paired_diff(a, b, n_boot=10000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    return dict(mean=float(d.mean()), ci95=bootstrap_ci(d, n_boot, seed), sd=float(d.std(ddof=1)), n=len(d))


def expert_normalised(r, r_stay, r_expert):
    """(student - do-nothing) / (expert - do-nothing), averaged over scenarios."""
    return float(np.mean((r - r_stay) / (r_expert - r_stay)))


def min_detectable_effect(sd, n, paired_sd=None, alpha=0.05, power=0.8):
    z = (1.959964 + 0.8416212) ** 2
    return float(np.sqrt(z * paired_sd ** 2 / n) if paired_sd is not None else np.sqrt(z * 2 * sd ** 2 / n))


def seed_level_ci(per_seed_values, n_boot=10000, seed=0):
    """CI over TRAINING seeds (the unit of analysis), not over episodes."""
    x = np.asarray(per_seed_values, float)
    return float(x.mean()), bootstrap_ci(x, n_boot, seed)
