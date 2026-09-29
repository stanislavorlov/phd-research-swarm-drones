"""
Random-policy (untrained) baseline for Table 2's held-out evaluation cells.

Answers the reviewer question directly: are the attention arms' waypoint-
completion fractions (0.034-0.048) meaningfully above what a policy that has
learned nothing at all would produce by chance, given 600 steps (60s) of
directed-but-uninformed motion in a 200x200x40 m box?

Uses the exact same env, config (pilot v3 scale), and metric definitions as
evaluate.py / mappo.py's instrumentation -- CR, Waypoint-Completion Fraction,
reached-wp1 / Time-to-First-Waypoint, MCR -- but replaces the learned policy
with actions sampled uniformly at random from [-1,1]^3 every step (the same
action space and env.step() the trained policies use; env internally scales
by v_max and applies the first-order-lag dynamics).
"""
import numpy as np
from pi_tgat_mappo.config import Config
from pi_tgat_mappo.env import UAVSwarmEnv

N_MIN, N_MAX = 8, 15
BOX = (200.0, 200.0, 40.0)
R_COMM, KAPPA = 110.0, 0.055
N_EPISODES = 20


def run(dropout_rate, seed0=123000):
    cfg = Config(
        n_min=N_MIN, n_max=N_MAX, box_size=BOX,
        r_comm=R_COMM, kappa=KAPPA, node_dropout_rate=dropout_rate,
        max_steps=600, n_waypoints=3, waypoint_radius=25.0,
    )
    crs, progresses, reached_first_flags, first_wp_steps, completions = [], [], [], [], []
    for ep in range(N_EPISODES):
        env = UAVSwarmEnv(cfg, seed=seed0 + ep)
        rng = np.random.default_rng(seed0 + ep + 500000)
        env.reset()
        n = env.n
        done = False
        t = 0
        first_wp_step = None
        connected_hist = []
        info = {}
        while not done:
            actions = rng.uniform(-1.0, 1.0, size=(n, 3))
            _, _, done, info = env.step(actions)
            connected_hist.append(float(env.is_graph_connected()))
            t += 1
            if first_wp_step is None and np.mean(env.wp_idx) >= 1.0:
                first_wp_step = t
            if t >= cfg.max_steps:
                break
        crs.append(np.mean(connected_hist))
        progresses.append(info.get("mission_progress", 0.0))
        completions.append(float(info.get("mission_complete", False)))
        reached_first_flags.append(first_wp_step is not None)
        first_wp_steps.append(first_wp_step)

    reached_first = [s for s in first_wp_steps if s is not None]
    cr_mean, cr_std = np.mean(crs) * 100, np.std(crs) * 100
    wf_mean, wf_std = np.mean(progresses), np.std(progresses)
    reached_pct = np.mean(reached_first_flags) * 100
    ttw_mean = np.mean(reached_first) if reached_first else float("nan")
    ttw_sec = ttw_mean * cfg.dt if reached_first else float("nan")
    mcr_mean, mcr_std = np.mean(completions) * 100, np.std(completions) * 100

    print(f"Random policy @ {int(dropout_rate*100):3d}% node dropout (n={N_EPISODES}): "
          f"CR = {cr_mean:5.1f}% +/- {cr_std:4.1f}%   "
          f"waypoint-frac = {wf_mean:.3f} +/- {wf_std:.3f}   "
          f"reached-wp1 = {reached_pct:5.1f}%   "
          f"time-to-wp1 = {ttw_mean:.1f} steps ({ttw_sec:.1f}s)   "
          f"MCR = {mcr_mean:5.1f}% +/- {mcr_std:.1f}%")
    return dict(dropout=dropout_rate, cr_mean=cr_mean, cr_std=cr_std, wf_mean=wf_mean, wf_std=wf_std,
                reached_pct=reached_pct, ttw_mean=ttw_mean, ttw_sec=ttw_sec, mcr_mean=mcr_mean, mcr_std=mcr_std)


print(f"Pilot v3 scale: box={BOX}, Rcomm={R_COMM}, kappa={KAPPA}, N~U({N_MIN},{N_MAX}), "
      f"max_steps={Config().max_steps}, n_waypoints={Config().n_waypoints}, "
      f"waypoint_radius={Config().waypoint_radius}, v_max={Config().v_max}")
print("-" * 110)
results = [run(dr) for dr in (0.0, 0.15, 0.30)]
