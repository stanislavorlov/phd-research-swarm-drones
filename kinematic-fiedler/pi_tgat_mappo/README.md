# PI-TGAT + MAPPO: swarm-cohesion experiment (kinematic priors + spectral connectivity reward)

A runnable implementation of the core architecture from *"Optimization of
Swarm Cohesion: Resilient Graph-Based MARL for UAV Swarms using Kinematic
Priors and Spectral Connectivity Rewards"* (revision 3): a graph-attention
actor with a temporal (GRU) memory and a physics-informed kinematic prior
for bridging communication blackouts, trained with MAPPO under a
Fiedler-value (algebraic connectivity) reward penalty.

Requires `torch` + `torch_geometric` (MPS-accelerated on Apple Silicon --
see the setup notes you were given separately). All four architectures the
paper compares -- PI-TGAT (ours), TarMAC-lite, DGN-lite, and Vanilla
MAPPO -- share this one codebase, selected via CLI flags at train time; see
"Baselines, combined" below for exactly what each flag combination gives
you and the honest caveats on how close each "-lite" baseline actually is
to its cited paper. See "Pilot run results & known limitations" for what a
first reduced-scale run of all four actually showed, and what it means for
interpreting any comparison this code produces.

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

## Pilot run results & known limitations (2026-09-22)

A first end-to-end run of all four architectures (PI-TGAT, TarMAC-lite,
DGN-lite, Vanilla MAPPO) was completed at a deliberately reduced scale to
fit a compute/time budget far below the paper's own (8-15 agents instead
of 20-50, a 200x200x40 m box instead of 1000x1000x200 m with `r_comm`/
`kappa` scaled proportionally, 600-step episodes, 150 training iterations).
Two things worth knowing before trusting any comparison from this run, or
a similarly-scaled one:

**A real code bug was found and fixed**: `connectivity_ratio` (CR) was
originally computed from `fiedler_value()`, which recomputes edge weights
from distance alone via a sigmoid that is never exactly zero -- meaning
the graph was *structurally* almost always "connected" by construction,
completely independent of the evaluation's forced node-dropout stress
test. This produced CR = 100.0% +/- 0.0% at every dropout rate for every
architecture, which is meaningless, not a real robustness result. Fixed
by adding `env.py`'s `is_graph_connected()`, a hard topological check on
the actually-realized `link_up` adjacency (which does reflect node
dropout), used for the `connected` info field that CR is averaged from.
`fiedler_value()` itself (used only for the training reward, `r_conn`)
was left untouched, so existing checkpoints remain valid -- only
`evaluate.py`'s measurement needed correcting, not retraining.

**Mission completion never emerged (MCR = 0.0% for all four,
all dropout rates)**, and `mission_progress` stayed flat near 0.01-0.04
across training with no upward trend. The likely cause: `omega = 2.5`
weights the spectral-connectivity reward term heavily relative to the
task-progress term, and at this compressed box scale a tightly-clustered,
stationary swarm is *already* fully connected almost for free -- so PPO
converged toward a connectivity-preserving, low-mobility local optimum
rather than active waypoint-seeking. This is consistent across all four
architectures (not specific to one), so it doesn't bias an architecture
comparison against any single baseline, but it does mean the CR-vs-dropout
numbers below should **not** be read as "robustness while performing the
mission" -- none of the four were meaningfully performing it.

That confound shows up directly in the corrected results:

```
            0% dropout: Vanilla 46.1%  |  PI-TGAT 44.8%  |  TarMAC-lite 37.9%  |  DGN-lite 33.5%
           15% dropout: Vanilla 28.3%  |  PI-TGAT 23.4%  |  TarMAC-lite 21.8%  |  DGN-lite 13.7%
           30% dropout: Vanilla 21.3%  |  PI-TGAT 16.1%  |  TarMAC-lite 17.2%  |  DGN-lite 13.9%
   (n = 20 episodes per cell; standard deviations are roughly as large as
   the means themselves -- e.g. 44.8% +/- 33.7% -- so none of these
   architecture-to-architecture differences would survive a real
   significance test at this n.)
```

Vanilla MAPPO -- the architecture with **no** cross-agent communication
mechanism at all -- has the *highest* CR at every dropout level. That is
the opposite of what the paper's core claim would predict, and it is best
explained by the same clustering-optimum above: with no reason to ever
coordinate movement, a non-communicating policy has no reason to spread
out either, so it stays maximally (and trivially) connected. In this
regime, CR is measuring "how tightly does each policy happen to cluster
by default," not "how well does its communication mechanism preserve
connectivity while navigating," which is what the paper's robustness
claim is actually about.

**Conclusion**: this pilot validates the pipeline end-to-end -- environment,
all four architectures sharing one codebase, the node-dropout evaluation
harness, and a now-correct hard-topology CR metric -- but does **not**
support ranking the four architectures on robustness. The next run should
lower `omega` (or add a stronger progress-shaping term) so `mission_progress`
actually grows during training, then re-run this same evaluation pipeline;
only once mission-seeking behavior emerges does a CR-vs-dropout comparison
across architectures become a meaningful robustness result rather than a
measurement of idle-clustering tendency.

## Pilot run v2: reward rebalancing and its effect (2026-09-24)

Following the pilot run above, `omega` (spectral-connectivity reward
weight) was identified as the likely cause of the clustering optimum:
`r_conn = -max(0, exp(lambda_crit - lam2) - 1)` is an *exponential*
penalty once connectivity drops below `lambda_crit`, and at `omega=2.5`
that penalty dominated the linear, small-magnitude `task_reward` enough
that PPO learned to avoid movement risking any disconnection at all,
rather than risk it for mission progress.

A second training pass (`--run-name <name>_v2` throughout) changed four
things together, isolating none of them individually but all pointing the
same direction -- reduce the cost of movement, increase the reward
signal's usefulness, and shorten the task:
- `omega`: 2.5 -> **0.3** (via a new `--omega` CLI flag)
- `entropy_coef`: 0.01 -> **0.03** (via a new `--entropy-coef` CLI flag,
  more exploration pressure)
- `n_waypoints`: 6 -> **3** (via `--n-waypoints`, shorter mission)
- `waypoint_radius`: 15 -> **25** (via `--waypoint-radius`, more forgiving
  target zone)
- `max_steps`, box scale, `r_comm`/`kappa` left unchanged from the
  original pilot (600 steps, 200x200x40 m box, `r_comm=60`, `kappa=0.1`) --
  a quick diagnostic run confirmed the episode-length budget was never
  the binding constraint (expected travel distance for 3 waypoints in
  this box is ~90-140 steps, well under 600).

**Effect on training**: this measurably broke the clustering optimum.
`connectivity_ratio` during training went from pinned at 1.0000 for every
iteration (original pilot) to varying freely (0.01-0.87 across a single
250-iteration run) -- the policy is now actually willing to risk
disconnection. `mission_progress` also rose ~5x in typical magnitude
(from a 0.01-0.04 ceiling to a 0.05-0.07 plateau, occasionally spiking to
0.15-0.2), though it plateaued rather than continuing to climb past
roughly iteration 60-80 of 250 (linear-fit slope over the full run:
-0.0001/iteration, i.e. flat to very slightly declining) -- more
iterations at this same configuration would not likely have helped
further; `mission_complete` never left 0.0% for any of the four
architectures even after this fix.

**Corrected CR-vs-dropout comparison, all four architectures retrained
under the v2 configuration:**

```
            0% dropout:  DGN-lite 27.8%  |  PI-TGAT 30.4%  |  TarMAC-lite 29.7%  |  Vanilla 39.7%
           15% dropout:  DGN-lite 26.0%  |  PI-TGAT 25.8%  |  TarMAC-lite 22.7%  |  Vanilla 33.0%
           30% dropout:  DGN-lite 20.5%  |  PI-TGAT 25.4%  |  TarMAC-lite 22.2%  |  Vanilla 32.2%
   (n = 20 episodes per cell, seed=123; standard deviations again roughly
   as large as the means, e.g. 39.7% +/- 33.0%.)
```

Two things changed and one thing didn't, relative to the original pilot.
Changed: the three *communicating* architectures (DGN-lite, PI-TGAT,
TarMAC-lite) are now much closer to each other -- differences of 2-5
points against 20-30 point standard deviations, i.e. statistically
indistinguishable from one another, consistent with all three plateauing
at similar `mission_progress` levels during training. Unchanged, and
worth taking seriously: **Vanilla MAPPO -- the architecture with no
cross-agent communication mechanism at all -- is still highest at every
dropout level**, and the gap did not shrink after specifically addressing
the reward-shaping issue that was the leading hypothesis for causing it:

```
                     0% dropout    15% dropout    30% dropout
original pilot:   Vanilla +1.3pt  Vanilla +4.9pt  Vanilla +5.2pt   (vs. PI-TGAT)
v2 (rebalanced):  Vanilla +9.3pt  Vanilla +7.2pt  Vanilla +6.8pt   (vs. PI-TGAT)
```

**Revised conclusion**: a pattern that only appeared once could plausibly
have been a training artifact or noise; one that replicates -- and if
anything strengthens -- across two training runs with substantially
different reward weighting is more likely a structural property of this
task/reward setup than an incidental bug. The likely fuller explanation:
the three communicating architectures are specifically designed (via the
kinematic prior / cached neighbor state) to keep attempting coordinated
movement through communication blackouts, which is a mechanism that
*encourages* tolerating disconnection in service of the mission. Vanilla,
with no such mechanism, has no comparable incentive to ever risk
disconnection, and simply defaults to whatever movement pattern keeps it
near its own local optimum -- which happens to preserve topology better.
In other words: CR alone, absent meaningfully differentiated mission
completion (`mission_complete` stayed 0.0% for all four in both pilot
runs), may structurally favor "does not communicate" regardless of
further reward tuning, because there is nothing in the task forcing a
communicating architecture's willingness-to-disconnect to pay off. A
future run that achieves actual, differentiated mission completion across
architectures (not just connectivity behavior) would be needed before a
CR-vs-dropout comparison can be read as evidence about the kinematic-prior
mechanism's value, one way or the other.

## Baselines, combined

All four architectures the paper compares are combinations of two
independent axes in `PITGATActor` (`networks.py`), controlled by
`train.py` CLI flags, rather than four separately-written models:

| Architecture     | flags                                        | aggregator  | graph? |
|-------------------|----------------------------------------------|-------------|--------|
| PI-TGAT (ours)    | *(default)*                                   | attention (GATv2Conv) | yes |
| TarMAC-lite       | `--no-kinematic-prior`                        | attention (GATv2Conv) | yes |
| DGN-lite          | `--no-kinematic-prior --aggregator conv`      | isotropic mean (`IsotropicConv`) | yes |
| Vanilla MAPPO     | `--no-graph`                                  | n/a (no cross-agent communication at all) | no |

**Honest caveat**: "TarMAC-lite" and "DGN-lite" are architectural
*proxies* -- same actor/critic scaffolding, GRU, and reward as PI-TGAT,
with only the neighbor-aggregation mechanism swapped -- not literal
reproductions of Das et al. 2020 or Jiang et al. 2020. TarMAC's actual
"who-to-address" learned gating and DGN's specific multi-hop convolution
stack are not implemented; what's here isolates one architectural
variable (attention vs. isotropic aggregation vs. no communication) while
holding everything else fixed, which is useful for an ablation but should
not be cited as a faithful reproduction of either paper's full method in
a comparison table.

All four share `env.py`, `buffer.py`, and the MAPPO update loop in
`mappo.py` -- the CLI flags above are the entire "comparison script."

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
