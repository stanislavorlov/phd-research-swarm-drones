"""
Feasibility check for the pilot mission (K=3 waypoints, 25 m acceptance
radius, 200x200x40 m box, 600-step / 60 s episode budget, v_max as in
config.py) -- BEFORE attributing 0% Mission Completion Rate to any learned
policy's failure.

Two oracle policies that require no learning at all:
  - greedy: every agent steers its full-speed velocity command straight at
    its own current target waypoint every step (first-order-lag physics,
    same env.step() used by the trained policies).
  - greedy_noisy: same, but with 15% isotropic action noise added, as a more
    "realistic" upper bound (an imperfect but purposeful controller) rather
    than a literal straight-line ideal.

If a straight-line/near-straight-line policy with no coordination, no RL,
and no network at all reaches all 3 waypoints well within the 600-step
budget, then the fixed budget is not the reason PI-TGAT/TarMAC/DGN/Vanilla
show 0% MCR -- the ceiling is reachable in principle.
"""
import numpy as np
from pi_tgat_mappo.config import Config
from pi_tgat_mappo.env import UAVSwarmEnv

N_EPISODES = 30
N_MIN, N_MAX = 8, 15
BOX = (200.0, 200.0, 40.0)
R_COMM, KAPPA = 110.0, 0.055


def run(mode, noise_std=0.0, seed0=5000):
    cfg = Config(
        n_min=N_MIN, n_max=N_MAX, box_size=BOX,
        r_comm=R_COMM, kappa=KAPPA, node_dropout_rate=0.0,
        max_steps=600, n_waypoints=3, waypoint_radius=25.0,
    )
    completions, progresses, steps_used, first_wp_steps = [], [], [], []
    for ep in range(N_EPISODES):
        env = UAVSwarmEnv(cfg, seed=seed0 + ep)
        env.reset()
        n = env.n
        done = False
        t = 0
        first_wp_step = None
        info = {}
        while not done:
            targets = env.waypoints[np.clip(env.wp_idx, 0, cfg.n_waypoints - 1)]
            direction = targets - env.pos
            norm = np.linalg.norm(direction, axis=1, keepdims=True) + 1e-8
            unit = direction / norm
            actions = unit  # full-speed toward target; env clips to [-1,1] and scales by v_max
            if noise_std > 0:
                actions = actions + env.rng.normal(0, noise_std, size=actions.shape)
                actions = np.clip(actions, -1.0, 1.0)
            _, _, done, info = env.step(actions)
            t += 1
            if first_wp_step is None and np.mean(env.wp_idx) >= 1.0:
                first_wp_step = t
            if t >= cfg.max_steps:
                break
        completions.append(float(info.get("mission_complete", False)))
        progresses.append(info.get("mission_progress", 0.0))
        steps_used.append(t)
        first_wp_steps.append(first_wp_step)

    reached_first = [s for s in first_wp_steps if s is not None]
    frac_reached_first = len(reached_first) / N_EPISODES
    mean_time_first = np.mean(reached_first) if reached_first else float("nan")

    print(f"mode={mode:12s} noise={noise_std:.2f}  "
          f"MCR={np.mean(completions)*100:5.1f}%   "
          f"mean waypoint-progress={np.mean(progresses):.3f}  "
          f"frac reaching wp1={frac_reached_first*100:5.1f}%  "
          f"mean steps-to-wp1={mean_time_first:6.1f} ({mean_time_first*cfg.dt:5.1f}s)  "
          f"mean episode length={np.mean(steps_used):.1f} steps ({np.mean(steps_used)*cfg.dt:.1f}s) "
          f"of {cfg.max_steps}-step ({cfg.max_steps*cfg.dt:.0f}s) budget")


print(f"v_max = {Config().v_max} m/s, dt = {Config().dt}s, max_steps = {Config().max_steps} "
      f"({Config().max_steps*Config().dt:.0f}s episode budget)")
print(f"box = {BOX}, waypoint_radius = {Config().waypoint_radius}m, K = {Config().n_waypoints}")
print("-" * 100)
run("greedy", noise_std=0.0)
run("greedy_noisy", noise_std=0.15)
run("greedy_noisy", noise_std=0.30)
