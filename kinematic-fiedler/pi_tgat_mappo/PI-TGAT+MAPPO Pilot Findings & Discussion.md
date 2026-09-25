# PI-TGAT+MAPPO Pilot Findings & Discussion

Sep 24, 2026 · @Stanislav Orlov

## Overview

This note summarizes two pilot training/evaluation passes of the PI-TGAT+MAPPO replication against three baselines (TarMAC-lite, DGN-lite, Vanilla MAPPO), run at a reduced scale (8-15 agents, 200x200x40 m box, 600-step episodes, 150 training iterations) to validate the pipeline before committing to paper-scale runs (20-50 agents, 6000-step episodes). All four architectures share one codebase and are compared via CLI flags controlling the neighbor-aggregation mechanism (attention, isotropic mean, or none).

The headline metrics follow the paper's Table 1 protocol: Connectivity Ratio (CR) and Mission Completion Rate (MCR), each evaluated at 0%, 15%, and 30% forced node dropout on a frozen policy (20 held-out episodes per cell, seed 123).

Two passes were run: an initial pilot (v1), and a second pass (v2) after rebalancing the reward function in response to what v1 showed.

## Pilot v1 results

A metric bug was found and fixed before these numbers were trusted: connectivity\_ratio (CR) was originally computed from `fiedler_value()`, a smooth distance-based signal that is never exactly zero, so the graph was structurally almost always "connected" regardless of the forced node-dropout stress test -- CR read 100.0% +/- 0.0% at every dropout rate for every architecture. This was replaced with a hard topological connectivity check on the realized (dropout-affected) adjacency matrix. The training reward itself was untouched, so no retraining was needed -- only the evaluation measurement.

Mission completion never emerged: MCR = 0.0% for all four architectures at all dropout rates, and `mission_progress` stayed flat near 0.01-0.04 through training. The likely cause is that the spectral-connectivity reward weight (omega = 2.5) dominated the small, linear task-progress term, so PPO converged to a stationary, connectivity-preserving swarm rather than one that pursued the mission. This affected all four architectures equally, so it doesn't bias the comparison toward one baseline -- but it does mean CR here reflects "how tightly a policy clusters by default," not robustness while performing the mission.

| Dropout | Vanilla MAPPO | PI-TGAT (ours) | TarMAC-lite | DGN-lite |
| --- | --- | --- | --- | --- |
| 0% | 46.1% | 44.8% | 37.9% | 33.5% |
| 15% | 28.3% | 23.4% | 21.8% | 13.7% |
| 30% | 21.3% | 16.1% | 17.2% | 13.9% |

n = 20 episodes per cell; standard deviations are roughly as large as the means (e.g. 44.8% +/- 33.7%), so none of these architecture-to-architecture differences would survive a significance test at this n.

The notable and unexpected result: Vanilla MAPPO -- the architecture with no cross-agent communication at all -- has the highest CR at every dropout level, the opposite of the paper's core claim. This pilot validated the pipeline (environment, all four architectures, the dropout evaluation harness, and a now-correct CR metric) but does not support ranking the architectures on robustness.

## Pilot v2 results (reward rebalancing)

Four changes were made together, all aimed at reducing the cost of movement and shortening the task: omega (spectral-connectivity reward weight) 2.5 to 0.3, entropy coefficient 0.01 to 0.03, waypoint count 6 to 3, waypoint acceptance radius 15 m to 25 m. Episode length and environment scale were left unchanged; a diagnostic run confirmed the step budget was never the binding constraint.

This broke the clustering optimum: connectivity\_ratio during training went from pinned at 1.0000 every iteration to varying freely (0.01-0.87 across a run), and mission\_progress rose roughly 5x in typical magnitude (0.01-0.04 ceiling to a 0.05-0.07 plateau, occasional spikes to 0.15-0.2). Progress plateaued after roughly iteration 60-80 of 250 (linear-fit slope over the full run: -0.0001/iteration, essentially flat) -- more iterations at this configuration would not likely help further. mission\_complete still never left 0.0% for any architecture.

| Dropout | Vanilla MAPPO | PI-TGAT (ours) | TarMAC-lite | DGN-lite |
| --- | --- | --- | --- | --- |
| 0% | 39.7% | 30.4% | 29.7% | 27.8% |
| 15% | 33.0% | 25.8% | 22.7% | 26.0% |
| 30% | 32.2% | 25.4% | 22.2% | 20.5% |

n = 20 episodes per cell, seed 123; standard deviations again roughly as large as the means (e.g. 39.7% +/- 33.0%).

Two things changed and one didn't, relative to v1. Changed: the three communicating architectures (PI-TGAT, TarMAC-lite, DGN-lite) are now close to each other -- 2-5 point differences against 20-30 point standard deviations, i.e. statistically indistinguishable. Unchanged: Vanilla MAPPO is still highest at every dropout level, and its lead over PI-TGAT did not shrink after addressing the leading hypothesis for causing it -- if anything it grew.

|  | 0% dropout | 15% dropout | 30% dropout |
| --- | --- | --- | --- |
| v1 (Vanilla vs. PI-TGAT) | +1.3 pt | +4.9 pt | +5.2 pt |
| v2 (Vanilla vs. PI-TGAT) | +9.3 pt | +7.2 pt | +6.8 pt |

## Discussion

A pattern observed once could plausibly be a training artifact or noise. The same pattern -- Vanilla MAPPO, the architecture with no cross-agent communication mechanism, achieving the highest connectivity ratio at every dropout level -- replicated, and if anything strengthened, across two training runs with substantially different reward weighting. That is stronger evidence for a structural property of the task and reward design than for an incidental bug in either run.

The most plausible explanation is a mismatch between what CR measures and what the architectures are doing. PI-TGAT, TarMAC-lite, and DGN-lite are all built around a mechanism -- the kinematic prior and cached neighbor state, in PI-TGAT's case -- specifically meant to let an agent keep attempting coordinated movement through a communication blackout rather than stall. That mechanism's entire purpose is to make tolerating disconnection acceptable in service of the mission. Vanilla MAPPO has no such mechanism and therefore no comparable incentive to ever risk disconnection; absent a reason to coordinate movement, it also has no reason to spread out, so it defaults to a compact, low-mobility formation that happens to preserve topology well. In this light, CR alone measures a byproduct of each policy's baseline mobility, not the robustness of its communication mechanism -- and it may structurally favor "does not communicate" regardless of further reward tuning, because nothing in the current task setup makes a communicating architecture's willingness to risk disconnection pay off in a way CR can register.

This reading is consistent with the other persistent finding: mission\_complete stayed at 0.0% for all four architectures across both pilots, even after v2 substantially increased mission\_progress and broke the training-time clustering optimum. Without differentiated mission completion, CR-vs-dropout has no accompanying evidence that any architecture is actually using its communication mechanism to accomplish the task under stress -- it is only evidence about default movement behavior under a reward that never strongly rewarded task completion in the first place.

**Limitations to state plainly in the paper.** These pilots ran at reduced scale (8-15 agents vs. the paper's 20-50; 600-step episodes; 150-250 training iterations vs. paper-scale budgets) and a single seed (123 at evaluation), so standard deviations were roughly as large as the reported means and no architecture-to-architecture difference here would survive a formal significance test. The CR-vs-dropout comparison in both pilots should be reported as a validation of the pipeline and a diagnostic finding about the reward design, not as a robustness ranking of the four architectures -- that ranking is not yet supported by evidence at this scale.

## Recommendations for the next run

1. **Get mission\_complete off zero before trusting any CR comparison.** This is the gating requirement -- without differentiated task completion across architectures, CR-vs-dropout cannot yet be read as evidence about the value of the communication mechanism. A shrunk-box configuration is ready to test this: smaller box (130x130x26 m, `r_comm` and `kappa` scaled to preserve relative connectivity/dropout dynamics), fewer/closer waypoints, the v2 reward weights (omega 0.3, entropy\_coef 0.03) carried forward. Command:

```
nohup python3 -m pi_tgat_mappo.train --run-name box_shrink_check --n-min 8 --n-max 15 --box-size 130 130 26 --r-comm 40 --kappa 0.15 --waypoint-radius 16 --n-waypoints 3 --max-steps 600 --iterations 60 --episodes-per-iter 2 --n-epochs 3 --omega 0.3 --entropy-coef 0.03 --save-every 10 > box_shrink_check.log 2>&1 &
disown
```

2. **Run multiple seeds before reporting any architecture ranking.** Both pilots used a single evaluation seed (123); standard deviations were roughly as large as the means. The paper's own protocol calls for 5 seeds ({42, 107, 219, 314, 501}) -- at minimum 3 seeds are needed before a Vanilla-vs-communicating-architecture gap can be reported as a finding rather than noise.
3. **If mission\_complete still doesn't move after the box-shrink check, treat this as a reward-design finding, not a tuning problem to keep chasing.** Two substantially different reward configurations (v1: omega 2.5; v2: omega 0.3, higher entropy, shorter/wider-target mission) both failed to produce non-zero mission completion. If a third, more targeted change (e.g. an explicit terminal completion bonus, or a denser progress-shaping term) doesn't resolve it either, the honest conclusion for the paper is that the reward formulation as specified needs revision before this environment can produce the paper's headline comparison, and that should be written up as a finding about the reward design rather than further hidden tuning.
4. **Once mission\_complete is non-trivial, re-run the full four-architecture x three-dropout-rate evaluation** using the same pipeline validated here (the corrected `is_graph_connected()` hard-topology CR metric, `evaluate.py`'s dropout stress test), and report both CR and MCR together -- CR alone, as both pilots show, is not sufficient to support a robustness claim.

## Final status: box-shrink check and the decision to stop tuning

The box-shrink run (recommendation 1 above) has completed for PI-TGAT: smaller box (130x130x26 m), scaled `r_comm`/`kappa`, 3 waypoints, v2's reward weights (omega 0.3, entropy\_coef 0.03), 60 iterations.

**mission\_complete stayed at 0.0%**, and mission\_progress was flat across the run (first-third mean 0.083, last-third mean 0.089 -- a 0.006 change, noise-level). This is the third substantially different configuration -- original pilot (omega 2.5), v2 (omega 0.3, more entropy, shorter/wider mission), box-shrink (v2's weights plus a denser environment) -- to produce zero mission completions.

Connectivity ratio for PI-TGAT did move: 30.4% / 25.8% / 25.4% (v2, at 0/15/30% dropout) to 60.1% / 55.4% / 54.5% in the shrunk box, roughly double, with mission completion unchanged (zero) before and after. Because only PI-TGAT was retrained at this scale, this number is not comparable to the other three architectures' v2 rows -- it confirms rather than refutes the earlier read that CR is highly sensitive to environment scale/density independent of task performance.

**Decision: stop tuning parameters here.** Three attempts across substantially different reward and scale configurations, with no movement on the one metric (MCR) that would validate any of them as evidence of task-directed behavior, is better read now as a finding about the reward/task formulation than as a search still waiting on the right setting. Continuing to guess at hyperparameters at this point would not be principled experimentation.

**What this supports writing in the paper, as of now:**

1. The replication pipeline is validated end-to-end: environment, all four architectures sharing one codebase, the node-dropout evaluation harness, and a corrected hard-topology CR metric.
2. The CR-vs-dropout numbers collected across all three pilot configurations are reportable as a diagnostic finding: under this reward formulation, architecture ranking on CR is dominated by baseline clustering/mobility behavior rather than communication-driven robustness, evidenced by MCR remaining at 0.0% in every configuration tested.
3. **The four-architecture robustness ranking the paper's Table 1 reports is not yet supported by this replication and should not be claimed.** That would require differentiated mission completion across architectures, which no tested configuration produced.
4. The open question -- whether the reward/task formulation itself needs revision (a denser progress-shaping term, an explicit completion bonus, or a longer/different task horizon) versus whether this is a genuine property of the described method -- is scoped as future work, not resolved here, and is the honest place to end this pilot phase.
