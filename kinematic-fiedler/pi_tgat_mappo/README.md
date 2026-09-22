# PI-TGAT + MAPPO: swarm-cohesion experiment (kinematic priors + spectral connectivity reward)

A runnable implementation of the core architecture from *"Optimization of
Swarm Cohesion: Resilient Graph-Based MARL for UAV Swarms using Kinematic
Priors and Spectral Connectivity Rewards"* (revision 3): a graph-attention
actor with a temporal (GRU) memory and a physics-informed kinematic prior
for bridging communication blackouts, trained with MAPPO under a
Fiedler-value (algebraic connectivity) reward penalty.

Requires `torch` + `torch_geometric` (MPS-accelerated on Apple Silicon --
see the setup notes you were given separately). This is the **main
architecture only** (PI-TGAT); the three literature baselines the paper
compares against (Vanilla MAPPO, TarMAC, DGN) are not implemented here --
see "Adding the baselines later" below for how this code is already
structured to make that cheap.

## Quick start

```bash
cd kinematic-fiedler
python3 -m pi_tgat_mappo.train --smoke-test     # ~seconds, verifies the install end-to-end
python3 -m pi_tgat_mappo.train                  # default demo-scale run (see config.py)
python3 -m pi_tgat_mappo.train --n-min 20 --n-max 50 --iterations 300 --max-steps 6000
                                                 # closer to the paper's actual scale (much slower)
```

Always run `--smoke-test` first. Outputs land in `pi_tgat_mappo/results/`:
`learning_curve.png` (return, Connectivity Ratio, Mission Completion Rate,
mean Fiedler value, losses, and the learned kinematic-prior weights over
training), `pi_tgat_mappo.pt` (actor + critic weights), and
`training_history.npz` (raw per-iteration metrics).

## Files

- `config.py` -- every hyperparameter, each annotated with its Table 1
  value and whether/why it was changed. This file *is* the answer to
  "which parameters did you use" -- read it alongside the table below.
- `env.py` -- `UAVSwarmEnv`: point-mass drone physics, the shared K-waypoint
  mission, the logistic RF dropout model, the soft-Laplacian Fiedler-value
  reward, and the raw (unweighted) Reynolds alignment/cohesion/separation
  components computed for dropped links.
- `networks.py` -- `PITGATActor` (GATv2Conv x2, learnable kinematic-prior
  weights + temporal decay, GRUCell, Gaussian policy head) and
  `CentralizedCritic` (permutation-invariant pooled MLP over the global
  state).
- `buffer.py` -- per-episode storage + GAE over the team-level reward.
- `mappo.py` -- `MAPPOTrainer`: episode collection and the recurrent
  (full-episode BPTT) PPO-clip update.
- `train.py` -- CLI entry point, logging, plotting.

## Table 1 parameter mapping

Every row of the manuscript's Table 1, and what this implementation
actually uses. "Unchanged" means the paper's number is used exactly.

| Category | Hyperparameter | Paper value | Used value | Notes |
|---|---|---|---|---|
| Physical & RF | Swarm scale | N in [20,50] | **sampled per episode** from [20,50] (`n_min`/`n_max`) | unchanged range; paper doesn't say per-episode vs fixed, we sample to also get the size-invariant-transfer property for free |
| Physical & RF | Bounding volume | 1000x1000x200 m | unchanged | |
| Physical & RF | Transmission range R_comm | 300 m | unchanged | |
| Physical & RF | Channel decay steepness kappa | **0.02 m^-1 (Table 1) vs 0.01667 m^-1 (body text)** | 0.02 (Table 1) | genuine inconsistency in the manuscript -- these two numbers are not the same; we default to the table since that's what was asked to be replicated. Override with `--kappa 0.01667` for the other one. |
| Physical & RF | Physical step interval | Delta t = 0.1 s (10 Hz) | unchanged | |
| Physical & RF | *(none -- not a Table 1 row)* | not stated | `v_max = 15 m/s` | the paper never gives a max quadrotor speed; added so the action head has a concrete scale. Override with `--v-max`. |
| Task bounds | Mission horizon | 600 s (6000 steps) | **150 steps (15 s)** by default | scaled down so a full-episode BPTT update through the GRU is tractable on a laptop; `--max-steps 6000` reproduces the paper's horizon (slow) |
| Task bounds | Spatial waypoints K | 6 | unchanged | implemented as ONE shared team mission path; each agent advances its own index through it (paper doesn't specify per-agent vs shared) |
| Task bounds | Waypoint acceptance radius | 15 m | unchanged | |
| Task bounds | Safe connectivity ratio tau_safe | 0.85 | unchanged (recorded, not yet used to gate an explicit success metric beyond CR/MCR logging) | |
| Task bounds | Spectral reward weight omega | 2.5 | unchanged | |
| Task bounds | Critical Fiedler threshold lambda_crit | 0.2 | unchanged | confirmed consistent between Table 1 and the Model Architecture section in this revision |
| Neural arch. | GAT layers / heads | 2 layers, 4 heads | unchanged | `torch_geometric.nn.GATv2Conv` |
| Neural arch. | Node/edge embedding size | 128 | unchanged | |
| Neural arch. | GRU hidden size | 128 | unchanged | `nn.GRUCell` |
| Neural arch. | Attention decay rate init | lambda = 0.1 | unchanged (learnable, softplus-parameterized) | |
| Neural arch. | Kinematic prior weights init | w1=1.0, w2=1.0, w3=1.5 | unchanged (learnable `nn.Parameter`s in `PITGATActor`) | moved from a fixed environment constant to a true backprop-trained network parameter, per the paper's own description |
| MAPPO opt. | Parallel environments | 16 (SITL workers) | **nominal only** -- episodes collected sequentially | no SITL, no vectorized runner; `episodes_per_iter` (default 4) is the real "how many episodes per update" knob |
| MAPPO opt. | Rollout length | 200 | superseded by `max_steps` | each collected episode runs to its own `done`, not a fixed truncation |
| MAPPO opt. | PPO epochs | 5 | unchanged | |
| MAPPO opt. | PPO clip epsilon | 0.2 | unchanged | |
| MAPPO opt. | GAE lambda | 0.95 | unchanged | |
| MAPPO opt. | Value loss coefficient | 0.5 | unchanged | |
| MAPPO opt. | Entropy coefficient | 0.01 | unchanged | |
| MAPPO opt. | Actor LR | 5e-4 | unchanged | |
| MAPPO opt. | Critic LR | 1e-3 | unchanged | |
| Verification | Seeds | 5 seeds {42,107,219,314,501} | default seed 42 only | run once per seed yourself (`--seed 107`, etc.) to reproduce the paper's error bands |
| Verification | Total env. steps | 1.0e7 over 18.4h (M4 Pro, 16 SITL workers) | **~90k steps by default** (`iterations=150` x `episodes_per_iter=4` x `max_steps=150`), minutes not hours | see "Scaling up" below |

## What was NOT replicated (and why)

- **ArduPilot SITL + pymavlink.** SITL is a real flight-controller binary
  process, not a Python simulation you can step from inside a training
  loop -- reproducing it would mean running N ArduPilot instances per
  environment and bridging them over MAVLink/UDP, which is an integration
  project on its own, not a "training parameters" question. `env.py`
  instead integrates the same `v_target` velocity command the paper's
  actor outputs, directly, as a first-order-lag point mass at the same
  10 Hz physical rate.
- **The three baselines (Vanilla MAPPO, TarMAC, DGN).** Out of scope for
  this pass per your earlier answer -- see below for how to add them.
- **torch_scatter / torch_sparse.** Not needed -- base `torch_geometric`
  (>=2.3) runs `GATv2Conv` in pure PyTorch, which is fine at N <= 50.

## Design choices worth knowing about

- **Fiedler value uses a soft (reliability-weighted) Laplacian**, not the
  hard binary in/out-of-range adjacency the Methods section literally
  describes ("edge e_ij instantiated if j is within range"). This repo's
  own `fiedler_sim.py` (a review script written against this manuscript)
  demonstrates that a hard-cutoff Laplacian makes lambda_2 piecewise
  constant in agent position -- a discontinuous reward signal that is bad
  for policy-gradient learning. Using `w_ij = 1 - P_drop(d_ij)` keeps
  lambda_2 continuous, consistent with the manuscript's own claim of using
  eigenvalue solvers "to ensure numerical stability and continuous
  differentiability through the spectral theorem."
- **Action bounding: clip, not tanh-squash.** The Gaussian policy head
  outputs an unbounded mean; sampled actions are clamped to [-1,1] before
  being sent to the environment, rather than passing the mean through
  `tanh` (which would need a change-of-variables correction to the PPO
  log-prob ratio to stay correct). This is a common simplification in
  continuous-action PPO/MAPPO implementations.
- **One centralized, non-agent-specific critic.** `CentralizedCritic`
  mean-pools per-agent embeddings into a single V(s) for the whole swarm
  (handles the variable swarm size N for free). GAE is computed once per
  timestep from the team-averaged reward, and that single advantage value
  is applied identically to every agent's policy-gradient term at that
  step -- a standard simplification when using a fully centralized critic.
- **`use_kinematic_prior` toggle.** Setting `--no-kinematic-prior` drops
  every out-of-range link entirely instead of projecting a synthetic
  state for it -- i.e. exactly the paper's stated *memoryless-GAT
  baseline* architecture. This was included because it's nearly free
  given the edge-schema design, and is the natural first ablation to run.

## Evaluating a trained checkpoint (Table 1's CR / MCR protocol)

`train.py` trains under organic, distance-based RF dropout throughout.
The paper's headline numbers (Table 1: CR and MCR at 0% / 15% / 30% node
dropout) are an EVALUATION-time stress test on top of that: a fixed
fraction of agents forced silent each step, independent of geometry. That
condition is not exercised during training at all -- it needs a separate
pass over a frozen policy, which `evaluate.py` does.

```bash
# Train two configurations into separate results directories so their
# checkpoints (and the config.json train.py saves alongside each one)
# don't overwrite each other:
python3 -m pi_tgat_mappo.train --n-min 20 --n-max 50 --iterations 300
mv pi_tgat_mappo/results pi_tgat_mappo/results_pi_tgat

python3 -m pi_tgat_mappo.train --n-min 20 --n-max 50 --iterations 300 --no-kinematic-prior
mv pi_tgat_mappo/results pi_tgat_mappo/results_baseline

# Evaluate each at 0/15/30% node dropout (20 held-out episodes per rate;
# --seed defaults to 123, different from training's default seed 42, so
# these are genuinely unseen episodes, not replays of training data):
python3 -m pi_tgat_mappo.evaluate \
    --checkpoint pi_tgat_mappo/results_pi_tgat/pi_tgat_mappo.pt --label "PI-TGAT (ours)"
python3 -m pi_tgat_mappo.evaluate \
    --checkpoint pi_tgat_mappo/results_baseline/pi_tgat_mappo.pt --label "Memoryless GNN"
```

Each invocation prints a CR/MCR table for that one checkpoint, appends its
rows to `pi_tgat_mappo/results/evaluation_table.json` (keyed by label +
dropout rate, so re-running a label overwrites just its own rows), and
regenerates `pi_tgat_mappo/results/evaluation_comparison.png` -- a grouped
bar chart across every label/rate evaluated so far, in the same shape as
the paper's Table 1. Run `python3 -m pi_tgat_mappo.evaluate --summarize`
with no `--checkpoint` needed to just rebuild the table/plot from what's
already been evaluated.

`evaluate.py` uses the DETERMINISTIC policy mean (no exploration sampling)
by default -- pass `--stochastic` to sample instead. It also auto-loads
the exact `Config` a checkpoint was trained with from that checkpoint's
`config.json` (written automatically by `train.py`), so a baseline
checkpoint's `--no-kinematic-prior` architecture is reconstructed
correctly without you having to remember or re-specify it -- just keep
each checkpoint and its `config.json` together in the same directory, as
the `mv ... results_*` step above does. You can override `--n-min`/
`--n-max` at eval time to test zero-shot transfer to a swarm size the
checkpoint wasn't trained on (the paper's Table 2 experiment).

**If you're running this on a remote GPU instance** (e.g. via the
`deploy/` scripts), re-sync after pulling these changes -- `evaluate.py`
is new and `mappo.py`/`env.py`/`config.py`/`train.py` all changed to
support it:
```bash
./deploy/deploy_vastai.sh <HOST> <PORT>
```

## Adding the baselines later

The env/graph layer is already baseline-agnostic:
- **Memoryless GAT baseline**: already available via `--no-kinematic-prior`.
- **DGN** (Jiang et al., 2020): swap `GATv2Conv` for a plain graph-
  convolution aggregator (isotropic neighbor averaging instead of learned
  attention) in `networks.py`; drop the GRU (DGN is memoryless).
- **TarMAC** (Das et al., 2020): replace the distance-gated edge set with
  a learned "who-to-address" attention/gating network that decides which
  neighbors to listen to, independent of physical range.
- **Vanilla MAPPO** (Yu et al., 2022): replace `PITGATActor` with a plain
  per-agent MLP over local features only (no graph, no neighbor
  information at all) -- structurally close to the boids-MARL experiment
  already in this repository's `marl/` folder.

All four would then share `env.py`, `buffer.py`, and the MAPPO update loop
in `mappo.py`, so the actual comparison script mainly swaps the actor
class and re-runs `train.py`.

## Scaling up

`config.py`'s defaults are tuned to finish in minutes on a laptop, not to
match the paper's reported numbers. To move toward paper scale:
`--n-min 20 --n-max 50 --max-steps 6000 --iterations 2000
--episodes-per-iter 16`, and run once per seed in `{42,107,219,314,501}`.
Expect this to take substantially longer than the demo config --
full-episode BPTT through a 6000-step, 20-50 agent recurrent GAT policy is
considerably heavier than the paper's own 18.4-hour SITL run, since SITL
parallelizes across 16 real worker processes and this implementation
collects episodes sequentially.
