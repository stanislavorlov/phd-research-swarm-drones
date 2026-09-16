# MARL drone-swarm experiment (MAPPO)

A small, dependency-light Multi-Agent PPO (MAPPO) experiment: a swarm of
drones in a shared 2D world, each with its own goal point, trained to fly
there while avoiding collisions with its teammates.

No PyTorch/JAX/Gym is required — everything runs on `numpy` (+ `matplotlib`
for the plot), including a ~250-line reverse-mode autodiff engine
(`autodiff.py`) used to train the small MLP actor/critic networks. This
keeps the whole experiment runnable anywhere and easy to read end-to-end.

## Files

- `env.py` — `DroneSwarmEnv`: N drones, continuous positions, 5 discrete
  actions (stay/up/down/left/right). Reward = per-agent progress toward its
  own goal + a bonus on first arrival, minus a shared penalty whenever two
  drones get closer than `collision_radius`. This is what makes the task
  genuinely multi-agent: a selfish policy that ignores teammates tends to
  collide and gets penalized for it.
- `autodiff.py` — minimal reverse-mode autodiff (`Tensor` + `Adam`) so the
  networks can be trained without an ML framework dependency.
- `networks.py` — `CategoricalActor` (shared policy, softmax over 5
  actions) and `CentralizedCritic` (value function over the concatenated
  global state). Both are 2-hidden-layer ReLU MLPs.
- `buffer.py` — per-episode rollout storage + GAE-Lambda advantage/return
  computation.
- `mappo.py` — `MAPPOAgent`: ties actor + centralized critic together with
  the clipped-PPO-surrogate update (parameter sharing across agents,
  Centralized-Training-Decentralized-Execution / CTDE via the centralized
  critic).
- `train.py` — training loop: collects on-policy rollouts, runs the MAPPO
  update, logs progress, and saves a learning-curve plot + trained weights
  to `results/`.

## Run it

From the project root:

```bash
python3 -m marl.train
```

Useful flags (see `python3 -m marl.train --help`):

```bash
python3 -m marl.train --n-agents 6 --iterations 500 --episodes-per-iter 10 --max-steps 60
```

A default run (4 agents, 500 iterations) takes well under a minute on a
laptop CPU and reaches ~100% of drones reaching their goal with near-zero
collisions.

## Outputs (`results/`)

- `learning_curve.png` — mean episode return, success rate (fraction of
  drones that reached their goal by episode end), and actor/critic losses
  over training.
- `mappo_policy.npz` — trained actor + critic weights (`agent.save()` /
  `agent.load()` in `mappo.py`).
- `training_history.npz` — raw per-iteration metrics, for further analysis.

## Design notes / how this maps onto "real" MAPPO

- **Parameter sharing**: since the drones are homogeneous, one actor
  network is shared across all agents rather than training N separate
  policies — the standard MAPPO setup for teams of identical agents, and
  far more sample-efficient than independent PPO per agent.
- **CTDE**: the critic is *centralized* — it sees the concatenation of
  every agent's local observation (the "global state") — while the actor
  only ever sees one agent's local observation. This gives lower-variance
  advantage estimates during training without requiring communication at
  deployment/execution time (only the actor is needed to fly a drone).
- **Clipped surrogate objective**: standard PPO-clip on the policy ratio
  `pi_new(a|o) / pi_old(a|o)`, plus an entropy bonus for exploration.
- **GAE(lambda)**: advantages are computed per-agent with Generalized
  Advantage Estimation before being pooled into the shared-policy update.

## Extending this

This is intentionally a minimal starting point for the PhD project, not a
finished benchmark. Natural next steps:
- Swap the hand-rolled autodiff engine for PyTorch if larger networks or
  GPU training become useful.
- Add a continuous action space (e.g. commanded velocity/acceleration)
  instead of the 5 discrete moves, closer to the MAVLink-level commands
  used elsewhere in this repo (see `../boids/`, `../swarm_bridge/`).
- Add obstacle avoidance, limited sensing range/partial observability, or
  a formation-keeping reward instead of independent goals.
- Log to TensorBoard/Weights & Biases instead of a static PNG once runs
  get longer.
