import numpy as np
from config import Config
from env import UAVSwarmEnv

# Confirm target-design invariant kappa*Rcomm
print("target kappa*Rcomm:", 0.02*300.0)

def episode_lambda2_series(r_comm, kappa, n_episodes=15, max_steps=200, sample_every=2, seed=123):
    cfg = Config(r_comm=r_comm, kappa=kappa, box_size=(200.0,200.0,40.0), n_min=8, n_max=15,
                 max_steps=max_steps, n_waypoints=3, waypoint_radius=25.0)
    rng = np.random.default_rng(seed)
    vals = []
    for ep in range(n_episodes):
        env = UAVSwarmEnv(cfg, seed=int(rng.integers(0,1_000_000)))
        env.reset()
        for t in range(max_steps):
            actions = env.rng.uniform(-1.0,1.0,size=(env.n,3))
            # step signature check
            out = env.step(actions)
            if t % sample_every == 0:
                vals.append(env.fiedler_value())
    return np.array(vals)

vals = episode_lambda2_series(110.0, 0.055)
print("n samples:", len(vals))
print("mean", vals.mean(), "median", np.median(vals))
for thresh in [0.2, 0.85]:
    frac = (vals <= thresh).mean()
    print(f"fraction <= {thresh}: {frac*100:.2f}%")
