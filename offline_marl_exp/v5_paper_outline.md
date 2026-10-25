# Paper outline: what a decentralised UAV swarm must see, and what online RL does to it

Built from `claude/v4_uav_visibility_and_online_results.md` (2026-10-25). Structured around the five questions from the
external review, as the reviewer recommended, with the critic-mask ablation demoted to a bounded null in an appendix.
Every number below names the file it comes from, so each table can be regenerated without re-running an experiment.
All result files are on the Mac under `offline_marl_exp/v2/`.

**One-line claim.** For a decentralised UAV swarm serving ground users over shared channels, *what each agent can see*
caps achievable performance far more than how it is trained; two cheap, deployable signals recover 93% of the gap to
full observability; and the online RL step that is supposed to close the rest buys its gain by abandoning the
worst-served users -- which a single reward term fixes.

---

## 0. The five questions and where each lands in the paper

| review question | the answer we have | lands in |
|---|---|---|
| Fresh held-out seeds for final numbers? | Yes; dev optimism is 0.8-1.1 returns, 6/6 configs same direction | S3 (protocol) + every results table |
| More offline students or more PPO seeds? | Students: ~2.4x tighter CI at a fixed run budget | S3 (protocol) + App. B |
| Interference-sensing feature? | Recovers 48% of the gap with zero communication; 93% combined with the broadcast | **S5 (main contribution)** |
| What does the online step change? | Bigger moves + a full channel re-assignment, and it sacrifices the weak tail | **S7 (main contribution)** |
| Stronger or certified planner reference? | The planner does not converge with compute; certificate is valid but loose | S3 (protocol) + App. A |

Two results that were not review questions carry the paper's spine and its best new finding: the
information-sufficiency ladder (S4) and the constrained online step (S7.3).

---

## 1. Introduction

Claims to state, in the order the evidence supports them:
1. Observation scope, not optimisation, is the binding constraint for a k-limited decentralised actor.
2. Two deployable signals -- a C-number broadcast and a receiver-side interference measurement -- are complementary and
   nearly close the gap.
3. Distilling a classical planner gives large sample-efficiency gains at small online budgets, and no advantage at the
   plateau; the honest claim is speed.
4. Unconstrained online RL improves the mean objective by degrading coverage; a floor-based penalty removes that
   without costing return.
5. Methodological: a heuristic planner used as a normaliser does not converge, so normalised scores are a convention
   that must name their reference.

## 2. Setting and method (no new results)
- Environment: N=8 UAVs, U=32 clustered mobile users, C=3 shared channels, T=20, A2G path loss `(d^2+H^2)^(-alpha/2)`,
  alpha=2.5, H=2, reward `mean_u log2(1+SINR_u) - 0.5 * mean_i ||d_i||^2`. Source: `commenv.py`.
- Teacher: batched receding-horizon BCD planner (`expert.py`, `planner_state.py`).
- Student: set-based actor (attention + DeepSets sum-pool), MDN position head, action-chunk auxiliary loss,
  k-radius masked observation. Source: `models.py`, `bc2.py`.
- Distillation: open-loop BC on 600 planner scenarios + 8 DAgger rounds x 256 scenarios, labels from a
  receding-horizon re-solve at every visited state. Source: `student.py`.
- Online: in-repo MAPPO (`mappo.py`, `online2.py`). State clearly that this is not HARL, and why that is a threat (S9).

## 3. Evaluation protocol (this is a contribution, not boilerplate)

### 3.1 Three seed sets -- Table T6
| set | seeds | role | stay-put | expert (strong++) | denominator |
|---|---|---|---|---|---|
| monitor/selection | 700000+ | picks the best DAgger round; PPO monitor evals | - | - | - |
| dev | 800000..800255 | development, arm contrasts, all per-run `--final` numbers | 51.08 | 122.56 | 71.48 |
| test (frozen) | 900000..901023 | every headline number, reported once | 50.56 | 121.73 | 71.17 |

Source: `results2/final/refs_800000_256_strong++.npz`, `refs_900000_1024_strong++.npz`; printed by `final_eval.py`.

**Measured dev optimism** (review Q1): all 6 configs score lower on test, by 1.31-2.22 returns, while stay-put drops
0.52 and the planner 0.83; net optimism 0.8-1.1 returns (~0.015 normalised), 6/6 same direction.
Source: `final_eval_dev_strong++.json` vs `final_eval_test_strong++.json`.

### 3.2 The reference planner does not converge -- Table T5, Figure F4
| budget | restarts x rounds x iters | mean return | vs previous rung | cost |
|---|---|---|---|---|
| strong (used in v1-v4, and as the DAgger teacher) | 4 x 4 x 80 | 120.77 | - | 1x |
| strong+ | 16 x 8 x 80 | 123.06 | +2.291 (SEM 0.403, 53/64) | 4x |
| **strong++ (reference)** | 32 x 16 x 160 | 124.09 | +1.030 (SEM 0.247, 47/64) | 27x |
| strong3 | 64 x 16 x 320 | 124.88 | +0.785 (SEM 0.169, 48/64) | 96x |

Log-linear at ~0.58 returns per e-fold of compute, still positive at 96x. Source: `results2/final/planner_check.json`,
regenerate with `planner_check.py --n 64 --ladder`.
Consequence to state plainly: normalised scores are reference-dependent; **return-level contrasts are not**.
Certified ceiling (interference-free LP + Lipschitz slack): 170.22 +/- 2.33 at delta=0.5, of which 6.34 is slack.
Source: `v4_probes2.py --ub 32 --ub_delta 0.5` (re-run to regenerate; not currently in a json).

### 3.3 Where replicates belong (review Q2) -- Appendix B, Figure F7
Variance decomposition of the 9 paired mask contrasts: moving replicates from PPO seeds onto offline students tightens
the 95% CI ~2.4x at a fixed run budget (3 students x 3 seeds -> +/-3.81; 9 students x 1 seed -> +/-1.6-1.8 depending on
the model). Source: the per-pair contrasts in the v4 doc S2; `v4_probes2.contrast_halfwidth` / `probe_table` style
computation. **We did not act on this** -- say so, and state the resulting bound on the null (S8).

---

## 4. Result 1: what the actor must see (the spine)

### Table T1 -- information ablation, TEST set (n=1024, strong++)
| config | actor sees | norm [95% CI] | return | % of the A7->A8_kinf gap |
|---|---|---|---|---|
| A7 | users and UAVs within k=6 | 0.353 [0.349, 0.355] | 75.67 | 0 |
| B_uinf | ALL users, UAVs within 6 | 0.362 [0.361, 0.363] | 76.22 | 2.0% |
| B_ainf | users within 6, ALL UAVs | 0.794 [0.791, 0.800] | 107.62 | 98.9% |
| A8_kinf | everything | 0.799 [0.789, 0.809] | 108.04 | 100% |

Gap = 0.446 norm = 31.7 returns. Source: `results2/final/final_eval_test_strong++.json`; dev companion in
`final_eval_dev_strong++.json`. Regenerate: `final_eval.py --names ... --seed0 900000 --n 1024 --planner strong++ --tag test`.

**The claim that distant users are worthless is stronger on the frozen set than on dev**: B_uinf's dev advantage over A7
(+1.31 returns) shrinks to +0.55 on test, the largest dev->test drop of any config.

### 4.1 It is the observation, not the optimiser -- Table T3b
Same algorithm, same budget, same seeds, random init: k=inf scratch MAPPO reaches 101.50 (norm 0.716) against k=6
scratch 92.88 (0.595) -- **+8.62 returns purely from widening the observation**.
Source: `results2/final/online_test.json` (tags `scratch_kinf`, `scratch_4500`).
Supporting: a k=inf actor evaluated under a k=6 mask collapses to 58.25, below the A7 student (75.67) -- it genuinely
uses the wide view. Source: the mis-mapped run of `online_test_eval.py` (reproduce by omitting `--ckpt_map`).

### 4.2 Mechanism -- Figure F5, Figure F6, Table T9 (appendix)
- Exact joint channel-selection headroom at each policy's own visited states (all C^N=6561 assignments):
  A7 **37.6**/episode, C_b1 13.5, B_ainf 10.6, A8_kinf 9.8. The A7-A8_kinf difference (27.8) accounts for essentially
  the whole 31.7-return information gap, so the gap is channel-selection quality, not trajectory shape.
  Source: `results2/final/probes2.json` -> `*::channels.headroom_per_episode`.
- Spatial statistics barely differ (move 0.325 vs 0.344; swarm spread 7.55 vs 7.38; serving distance 2.67 vs 2.41) and
  A8_kinf tolerates MORE co-channel crowding (2.99 vs 2.74) at a higher rate. Same source.
- Channel switching is a ~5-step settling phase, not sustained agility: A7 switches 0.131 of agents at t=1 and 0.006
  after; informed students 0.23-0.26 at t=1 and 0.025-0.029 after. Source: `probes2.json` -> `switch_profile`.
  **This corrects the reviewer's "2.5-3x more channel agility" reading** -- worth one sentence in the paper.

---

## 5. Result 2: two cheap signals recover 93% of the gap (review Q3 -- main contribution)

### 5.1 Motivation from a decision probe -- Figure F5
At A7's own visited states, accuracy against the exact best-response channel (serving UAVs, ties handled, 3 students):
sensed power **0.760**, broadcast count 0.556, sensed restricted to k=6 0.558, A7's own choice 0.467, random ~0.34.
Causal (previous-step measurement) ~= oracle (simultaneous), so a one-step delay is nearly free.
Source: `results2/final/probe_offline.json` -> `*::channels.accuracy_serving`.

### 5.2 The arms -- Table T2, Figure F1
Feature appended to `sf`, all C wide except `bcast=6` (2C): `log1p(sum_{j!=i, ch_j=c} g_ij / noise)/8`.
TEST set (n=1024, strong++):

| arm | what it is | norm | return | % of gap |
|---|---|---|---|---|
| A7 | k=6 baseline | 0.353 | 75.67 | 0 |
| `bcast=4` | sensed power restricted to k -- **control** | 0.368 | 76.73 | 3.4% |
| `bcast=5` | channel-shuffled sensed power -- **placebo** | 0.382 | 78.13 | 6.5% |
| `bcast=3` | sensed power, all UAVs, zero communication | 0.566 | 91.68 | 47.8% |
| `bcast=1` (C_b1) | broadcast count of UAVs per channel | 0.714 | 102.23 | 80.9% |
| `bcast=2` (C_b2) | C_b1 + user distance rings | 0.718 | 102.44 | 81.8% |
| **`bcast=6`** | **count AND sensed power** | **0.769** | **105.97** | **93.3%** |
| A8_kinf | everything | 0.799 | 108.04 | 100% |

Source: `final_eval_test_strong++.json`. Dev companion: `final_eval_dev_strong++.json`.

**Points to make:**
- Both controls sit on the baseline (+1.06 and +2.46 returns), so neither extra input width nor the near field explains
  anything; the value is entirely in the long-range tail of the co-channel power.
- The two statistics are complementary, not redundant: 48% + 81% alone, 93% together; `bcast=6` beats the broadcast by
  +3.74 returns and sits 2.07 below full observability.
- **A k=6 actor with both signals (105.97) beats a fully-observing actor trained from scratch (101.50).**
- Name the quantity honestly: it is receiver-side interference feedback measured at the UAV's own position (what its
  own users see), not UAV-side spectrum sensing.
- **Report the failed pre-registration.** The probe (0.760 vs 0.556) predicted sensing would match or beat the
  broadcast; it reached 59% of it. Greedy per-state rule accuracy at one policy's state distribution does not estimate
  what a student trained on the feature achieves. This belongs in the paper as a caution about probe-driven feature
  selection.

---

## 6. Result 3: distillation buys speed, not a better policy

### Table T3 -- sample efficiency (TEST set unless noted)
| run | return | norm | outage |
|---|---|---|---|
| A7 offline (distilled only) | 75.67 | 0.353 | 0.0447 |
| A7 + online, 450 it | 90.94 | 0.567 | 0.0801 |
| A7 + online, 4500 it | 90.80 | 0.565 | 0.0920 |
| scratch k=6, 450 it | 65.99 | 0.217 | 0.1771 |
| scratch k=6, 4500 it | 92.88 | 0.595 | 0.1107 |
| scratch k=inf, 4500 it | 101.50 | 0.716 | 0.0875 |

Source: `results2/final/online_test.json`. Training curves: `results2/online/<tag>/run_masked_seed*.npz`
(`iters`, `returns`), tabulated by `scratch_compare.py`.

- Equal 450-iteration budget: **+24.95 returns** for the distilled start.
- Time to match: scratch's 3-point-averaged monitor curve first reaches the A7+online@450 level at iteration 1770
  (**3.9x**); for a 92 target it is ~2x. So quote **2-4x depending on the target level**.
- Equal 4500-iteration budget: **-2.08** (scratch ahead). State this plainly -- the claim is speed, not superiority.
- k=inf scratch had NOT converged (91.84 at it 2000 -> 101.83 at 4500), so the 6.54-return distillation lead at k=inf
  is a budget statement. Source: `results2/online/scratch_kinf/log_masked_seed*.txt`.

### 6.1 Total budget including the offline stage -- Table T3c
| item | amount |
|---|---|
| round-0 planner data | 600 scenarios = 12,000 team steps; 2 min, one-off, shared |
| DAgger | 8 x 256 x 20 = 40,960 team steps (40,960 receding-horizon solves); 551-626 s/student |
| BC + DAgger training | 394-532 s/student |
| offline total | 52,960 team steps; ~18 min/student |
| online 450 it | 288,000 team steps; ~64 min at 8.5 s/it |
| **distilled pipeline** | **340,960 team steps** |
| scratch to the same level (~1770 it) | 1,132,800 team steps |

Step ratio **3.3x**; the offline stage is 15% of the pipeline's steps. Source: `offline_cost.py` reading
`results2/A7/seed*/log.txt` and `data/expert_train.pt`. State the assumption: the planner needs a differentiable
simulator; the RL baseline does not.

### 6.2 The coverage column changes this comparison -- tie into S7
scratch@4500 is 2.48x the offline student's outage, A7+online@4500 2.06x, and the constrained 450-iteration run
**0.79x**. On two axes the constrained distilled route dominates: within 3.4 returns of scratch@4500 at a third of its
outage and a tenth of its online samples. Source: `online_test.json`.

---

## 7. Result 4: what online RL does, and how to fix it (review Q4 -- best new finding)

### 7.1 The diagnosis -- Table T4, Figure F3
Each tuned actor paired with its OWN offline student, 3 seeds, 64 dev scenarios:

| paired change | unconstrained masked | unconstrained unmasked |
|---|---|---|
| return | +11.68 (+15%) all + | +10.76 (+13%) all + |
| mean user rate | +0.627 (+15%) all + | +0.605 (+15%) all + |
| **rate q10** | **-0.486 (-33%) all -** | **-0.365 (-24%) all -** |
| rate q50 | +1.020 (+26%) all + | +0.884 (+22%) all + |
| **outage** | **+0.0562 (+129%) all +** | **+0.0438 (+110%) all +** |
| co-channel within 6 | +0.47 (+17%) all + | +0.55 (+20%) all + |
| move size | +0.121 (+38%) all + | +0.169 (+53%) all + |
| switches at t=1 | +0.454 (+396%) all + | +0.465 (+409%) all + |

Source: `results2/final/probe_masked_{0,1,2}.json`, `probe_unmasked_{0,1,2}.json`, `probe_offline.json`; tabulated by
`probe_table.py`.

**The reward is `mean_u log2(1+SINR_u) - energy`: utilitarian, with no floor on the worst-served user, and PPO exploits
exactly that.** Also report: the coverage loss does NOT buy the return (correlation between return gain and outage
increase across the 6 actors = **-0.52**), which is what makes it removable.

### 7.2 The pre-registration that failed, again worth reporting
Predicted "switch rate up, move size unchanged" by analogy with the information ablation. Both levers moved. State it.

### 7.3 The fix -- Table T4b, Figure F3
Penalty subtracted from the **training reward only**; evaluation, `ep_return` and every held-out number stay on the
original objective, and `info["outage"]` keeps its fixed 1 bit/s/Hz threshold. Source: `mappo.PPOConfig.outage_coef` /
`shortfall_coef`, `EnvConfig.rate_floor`, `online2.py --outage_coef/--shortfall_coef/--rate_floor`.

DEV (n=256, 3 seeds), paired against each seed's own unconstrained run:

| arm | return | outage | x offline | paired vs unconstrained | beats own student's coverage |
|---|---|---|---|---|---|
| unconstrained | 89.84 +/- 2.53 | 0.0917 +/- 0.0293 | 2.05x | - | 0/3 |
| `outage_coef 5` | 91.70 +/- 5.18 | 0.0460 +/- 0.0207 | 1.03x | +1.85 +/- 4.60, -50% | 2/3 |
| `outage_coef 15` | 89.23 +/- 6.59 | 0.0361 +/- 0.0277 | 0.81x | -0.61 +/- 4.17, -61% | 2/3 |
| **`shortfall 15, floor 2`** | **90.35 +/- 5.99** | **0.0356 +/- 0.0087** | **0.80x** | **+0.51 +/- 4.09, -61%** | **3/3** |

TEST (n=1024): `sf15 floor2` **89.61 / outage 0.0353 = 0.79x the start student**; unconstrained 90.94 / 0.0801 (1.79x).
Gains over the start student: **+13.94 with 21% less outage** vs +15.27 with 79% more.
Source: `results2/online/con_*/run_masked_seed*.npz` (`final_returns`, `final_outage`, `coefs`) and `online_test.json`.

**The mechanism, and the sentence the section is built on** -- constrained vs unconstrained, same pairing, 3 seeds:

| | `sf 15, floor 2` | unconstrained |
|---|---|---|
| return | +10.46 all + | +11.68 all + |
| rate q10 | **+0.134 (+9%) all +** | **-0.486 (-33%) all -** |
| outage | +0.0006 (flat) | +0.0562 (+129%) all + |
| co-channel within 6 | **-0.48 (-17%) all -** | **+0.47 (+17%) all +** |

Source: `probe_sf15f2_{0,1,2}.json` vs `probe_masked_{0,1,2}.json`, via `probe_table.py`.
Both arms gain the same return and the same mean rate. The unconstrained one redistributes; the constrained one finds a
less crowded configuration. **The floor must sit ABOVE the reporting threshold** -- at `floor 1` the graded penalty is no
better than the step function (90.45 / 0.0562 at seed 0).

### 7.4 Negative result worth keeping
Online fine-tuning an already-strong student (C_b1) gains **+0.09 returns** and takes outage from 0.0353 to 0.0902
(**2.55x**). Source: `online_test.json`. One paragraph; it sharpens S6's "no headroom" into "no headroom and a real cost".

---

## 8. Result 5: a bounded null (the demoted mask ablation) -- Appendix C
Masked vs unmasked critic, 3 students x 3 PPO seeds x 2 arms: dev pooled **-0.03** returns (pair-level 95% CI
[-1.75, +1.70]; student-level [-3.84, +3.79], n=3); test by PPO offset +0.85 / +0.41 / -1.32, mean **-0.02**.
Source: v4 doc S2 per-pair table; `online_test.json` for the test replication.
Frame as: any effect larger than ~4 returns is unlikely at this setting, which is far below the +15 online gain and the
+32 information gap. Note the 9-students x 1-seed design would have bounded it at +/-1.6-1.8 for the same 18 runs.

## 9. Threats to validity (write this before a reviewer does)
- 3 offline students per config; the mask null bounded only at +/-3.8; return CIs on the constrained arms +/-4-6.
- One environment size (N=8, U=32, C=3); no N=16 replication.
- The normaliser is a heuristic planner that does not converge (S3.2); return-level contrasts are unaffected.
- In-repo MAPPO rather than HARL; the update step is isolated so it can be swapped.
- k=inf scratch had not converged, so the k=inf distillation lead is a budget statement.
- Probes are 64 dev scenarios per actor; the coverage statistics in S7.1 were not recomputed on the test set.
- The constraint has only been applied from the A7 start, and `rate_floor` was tried at 1 and 2 only.

## 10. What is still missing for submission
| item | effort | blocking? |
|---|---|---|
| **All figures** (F1-F7) -- none exist; plotting scripts needed | ~half a day, local | yes |
| Certified ceiling as a stored json (currently only stdout) | minutes, local | no |
| `rate_floor` sweep (3, 4) and `floor 2` x `outage_coef` | ~1 h, one instance | no |
| Constraint from the C_b1 start | ~1 h, one instance | no |
| Tight planner-suboptimality on small instances (review Q5's second half) | ~half a day | no -- but it is the one part of Q5 still open |
| More offline students (review Q2) if a reviewer presses on the null | ~2 h offline + 18 online runs | no |

---

## Figure list
| id | content | source |
|---|---|---|
| **F1** | Information ladder: test norm for A7, B_uinf, bcast3, C_b1, bcast6, B_ainf, A8_kinf, with the gap annotated | `final_eval_test_strong++.json` |
| **F2** | Sample efficiency: monitor eval vs iteration for A7+online, scratch k=6, scratch k=inf; budget-matched markers at 450 and 4500 | `results2/online/*/run_masked_seed*.npz` |
| **F3** | **Return vs outage Pareto plot**: offline student, unconstrained online, oc5, oc15, sf15f2, scratch arms. One figure carries S6.2 and S7 together -- the paper's best single image | `online_test.json` |
| **F4** | Planner ladder: return vs compute on a log axis, with the certified ceiling as a horizontal band | `planner_check.json` |
| **F5** | Channel-rule accuracy bars (sensed / broadcast / k-restricted / own / random) at each policy's states | `probes2.json`, `probe_offline.json` |
| **F6** | Channel-switch rate vs step t, showing the ~5-step settling phase | `probes2.json` -> `switch_profile` |
| **F7** | CI half-width vs number of students at 1/2/3 PPO seeds (review Q2) | computed, no stored file |

## Table list
T1 information ablation (S4) | T2 sensing arms + controls (S5.2) | T3 sample efficiency (S6) | T3b k=inf scratch (S4.1) |
T3c budget accounting (S6.1) | T4 online-step diagnosis (S7.1) | T4b constrained arms (S7.3) | T5 planner ladder (S3.2) |
T6 seed sets (S3.1) | T7 mask null (App. C) | T8 scratch PPO sweep (App. D) | T9 joint channel headroom (S4.2)

## Regeneration commands
```bash
python planner_check.py --n 64 --ladder
python final_eval.py --names A7 A8_kinf B_uinf B_ainf C_b1 C_b2 D_sense3 D_sense4 D_sense5 D_sense6 \
  --seed0 900000 --n 1024 --planner strong++ --tag test
python online_test_eval.py --seed0 900000 --n 1024 --planner strong++ \
  --ckpt_map "scratch_kinf=results2/A8_kinf/seed{s}/best.pt,scratch" --out results2/final/online_test.json
python probe_table.py
python scratch_compare.py --a7_tag online_A7_long
python offline_cost.py --name A7 --seeds 0 1 2
python v4_probes2.py --ub 32 --ub_delta 0.5
```
