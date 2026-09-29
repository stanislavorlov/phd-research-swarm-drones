"""
connectivity_check.py -- sanity-check the rescaled pilot environment BEFORE
spending any training compute: at 0% node dropout, with the swarm mean
degree preserved at the smaller N in [8,15] / 200x200x40 m box, is the
graph connected most of the time?

Two motion regimes, since CR is a time-average over an episode and we
don't have a trained policy yet for this config:
  - idle:   zero actions (agents don't move) -- the easiest case, checks
            the initial random deployment's connectivity.
  - random: uniform random actions each step -- a harsher, adversarial
            case (an untrained/random policy would look roughly like this
            before any training pushes it toward the reward).

Reports the fraction of timesteps the graph is connected (== the CR metric
evaluate.py computes), averaged over n_episodes, for a small grid of
r_comm / kappa candidates in the requested range.
"""
import numpy as np
from pi_tgat_mappo.config import Config
from pi_tgat_mappo.env import UAVSwarmEnv

N_EPISODES = 20
MAX_STEPS = 200          # short episodes are enough for a geometry sanity check
BOX = (200.0, 200.0, 40.0)
N_MIN, N_MAX = 8, 15

GRID = [
    (100.0, 0.06),
    (110.0, 0.055),
    (120.0, 0.05),
]

def run(r_comm, kappa, mode, seed0=1000):
    cfg = Config(
        n_min=N_MIN, n_max=N_MAX, box_size=BOX,
        r_comm=r_comm, kappa=kappa, max_steps=MAX_STEPS,
        node_dropout_rate=0.0,
    )
    connected_fracs = []
    degrees = []
    for ep in range(N_EPISODES):
        env = UAVSwarmEnv(cfg, seed=seed0 + ep)
        env.reset()
        n = env.n
        connected_flags = [env.is_graph_connected()]
        # mean degree at reset, from the realized link_up adjacency
        deg = env._last_link_up.sum(axis=1).mean()
        degrees.append(deg)
        for t in range(MAX_STEPS):
            if mode == "idle":
                actions = np.zeros((n, 3))
            else:
                actions = env.rng.uniform(-1, 1, size=(n, 3))
            _, _, done, info = env.step(actions)
            connected_flags.append(info["connected"])
            if done:
                break
        connected_fracs.append(np.mean(connected_flags))
    return np.mean(connected_fracs), np.std(connected_fracs), np.mean(degrees)

print(f"{'r_comm':>8} {'kappa':>7} {'mode':>7}   CR (idle/random policy, 0% dropout)   mean degree @ reset")
print("-" * 80)
for r_comm, kappa in GRID:
    for mode in ("idle", "random"):
        mean_cr, sd_cr, mean_deg = run(r_comm, kappa, mode)
        print(f"{r_comm:8.0f} {kappa:7.3f} {mode:>7}   CR = {mean_cr*100:5.1f}% +/- {sd_cr*100:4.1f}%        mean_degree = {mean_deg:4.2f}")
    print()
