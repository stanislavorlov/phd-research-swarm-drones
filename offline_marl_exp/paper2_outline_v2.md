# Paper 2 — outline v2

Replaces `claude/paper2_offline_marl_outline.md`, which predates almost every result in it. Run-level detail
and the correction history live in `claude/p2_dagger_quality_result.md` (result 1),
`claude/p2_iql_result.md` (results 2 and 3) and `claude/p2_madt_result.md` (result 4); this document is the
paper plan and carries the numbers the paper should quote. **All figures below are the frozen test set**
(900000..901023, n = 1024, `strong++`, stay-put 50.56, denominator 71.17) unless stated. Offline arms have
**five seeds** as of 2026-11-14; online arms have three PPO seeds.

## The argument in three sentences

Classical solvers produce optimal trajectories too slowly to fly, so the pipeline distils them into a policy and
fine-tunes it online. Offline pre-training is worth **+45 returns** at a matched online budget, and a
behaviour-cloned start plus 450 MAPPO iterations reaches **98% of what DAgger reaches with 40,960 extra solver
calls**. But the offline algorithm barely matters: neither a learned critic nor return-conditioning beats plain
cloning, only selection with an *exact* planner-computed advantage helps, it lands at or marginally above
cloning the clean subset, and after fine-tuning all starts of equal offline quality are indistinguishable.

## Candidate titles

1. *Demonstration quality dominates offline algorithm choice: offline-to-online multi-agent RL from a classical
   trajectory solver*
2. *What offline reinforcement learning extracts from a classical solver, and what online fine-tuning does with it*

## Framing decision

Lead with offline-to-online, because the strongest positive result is the +45 returns and the project was posed
that way. The offline comparison becomes the *mechanism* section. The middle of the paper is a negative result
and the writing must make it load-bearing rather than apologetic: the branch gap of −0.001 against a true
+0.857 is a **measurement**, not an absence, and the `ft_hard` ≈ `ft_bc` null is what licenses the practical
recommendation.

## Sections

**1. Introduction.** Sample-inefficiency of online MARL; solvers optimal but slow; distillation plus fine-tuning
as the bridge. State the three findings up front, including the negative one.

**2. Related work.** Offline MARL (IQL, CQL, decision transformers); learning from optimisation-based experts;
DAgger and distribution shift; offline-to-online fine-tuning. Cite paper 1 for the shared environment.

**3. Methods.**
- 3.1 Environment and objective — compressed, citing paper 1.
- 3.2 The planner, and the **measured** quality ladder (`planner_check.py --ladder`), not asserted.
- 3.3 Datasets: single-trajectory, mixed-quality, and the **paired counterfactual** construction — two actions
  at every visited state, exact because the environment's whole mutable state is (positions, channels, t,
  previous action), so save/step/restore is exact. The methodological contribution; deserves a figure.
- 3.4 Exact advantages: `Q_true(s,a) = r(s,a) + J(s′)` with `J` from the planner, so the advantage carries no
  estimation error and can serve as ground truth for auditing a critic.
- 3.5 Offline extraction: BC, BC+DAgger, IQL/AWR, hard filtering, random-selection control, **MADT**.
- 3.6 Online fine-tuning: MAPPO, citing paper 1 §M10. **Flag explicitly** that `online2.py`'s defaults are not
  the protocol (lr_actor 3e-4 not 1e-3, critic_warmup 100 not 15).
- 3.7 Evaluation: three seed sets, cached references keyed by seed set and planner budget, ratio-of-means, the
  offline student as the unit of analysis. Same conventions as paper 1, so numbers cross-reference.

**4. Results.** Ordered so the positive claim lands first and the mechanism explains it.

| § | claim | rests on |
|---|---|---|
| 4.1 | Demonstrator quality and relabelling are **complements** | Table 1 (quality ladder), Table 2 (BC vs BC+DAgger over 4 rungs, 5 students, test): BC 75.35 / 79.92 / 78.44 / 84.30, DAgger 78.42 / 89.69 / 94.47 / 105.25; DAgger's gain falls 20.95 → 3.07 as the teacher weakens |
| 4.2 | Pre-training is worth **+45 returns** at a matched online budget | Table 3: `ft_hard` 103.99, `ft_bc` 103.45, `ft_scratch` 58.78 |
| 4.3 | Online RL reaches **98% of DAgger with zero solver calls** | `ft_bc` 103.45 against `QD_strong` 105.25 on the same seed set; 288,000 environment steps against 40,960 solves |
| 4.4 | The offline method **does not transfer** | paired `ft_hard` − `ft_bc` = **+0.55** (SE 0.46) on test and **−0.50** (SE 0.53) on dev — the sign flips at half a return |
| 4.5 | Offline, exact advantages **do** help: +13.9 returns | Table 4 (five seeds); selection worth +12.84 at identical data volume (`P_hard` − `P_rand`); data volume itself free (`P_rand` − `P0_pairs` = +1.02, intervals overlap) |
| 4.6 | **Neither** offline family beats plain cloning | learned critic: `L125_pairs` 68.72, 4.5 below unweighted BC, and no n_eff from 87.5% to 24.5% beats it. Return-conditioning: `M_mix` 67.99 ± 1.46 against `B_mix` 76.99 on the identical dataset |
| 4.7 | **Why (critic)**: it ignores its action argument | branch gap **−0.001** against a true **+0.857**, identical to three decimals across three seeds while the state value wanders by 0.66 |
| 4.8 | **Why (MADT)**: the fit is fine, the rollout is not | teacher-forced val MSE 0.046, flat by epoch 20; fails in closed loop where it conditions on its own actions *and* a tracked return-to-go; mean outage 0.147 against ~0.06 |
| 4.9 | The ceiling is **the clean subset** | `P_hard` 87.08 against `Q_strong` 84.30 — at or marginally above clean-data cloning, inside `Q_strong`'s seed range (81.5–86.6) — and 18 below DAgger, which buys new labels where the student goes |
| 4.10 | Fine-tuning buys return with **coverage** | outage 0.060 → 0.073 (`ft_hard`), 0.065 → 0.078 (`ft_bc`); `ft_scratch` 0.134 and erratic |

**5. Threats.** Three PPO seeds; one MAPPO implementation; `ft_scratch` differs in architecture as well as
initialisation (`mdn_k=0, chunk=1`); `M_mix` likewise differs in head from `B_mix`, with the teacher-forced fit
as the defence; one MADT seed of five selected at the sweep's lower endpoint; 450 iterations is a
sample-efficiency claim, not a final-performance one; a single environment size, citing paper 1's §4.6 for how
far that travels.

**6. Discussion.** The practitioner's sentence — *if you will fine-tune online, spend the budget on
demonstration quality, not on the offline algorithm.* Then the sharper point: advantage-weighted BC **selects**,
it does not **stitch**, so its ceiling is BC-on-the-best-subset by construction; MADT fails for the mirror
reason, compounding its own error in two coupled channels. DAgger beats both because it buys *new* labels where
the student actually goes, which is a distribution-shift fix, not a reweighting.

**7. Conclusion.**

## Figures

| # | content | status |
|---|---|---|
| 1 | The paired-counterfactual construction: commit branch A, evaluate branch B, roll back | to draw |
| 2 | Advantage distributions, exact against learned, with the branch gap marked | to draw, data in hand |
| 3 | Offline arms on the test set, normalised, with intervals — including `M_mix` and `B_mix` | to draw, data in hand |
| 4 | **Online learning curves**, three initialisations × three seeds, return against iteration | to draw — `iters` and `returns` already in `results2/online/ft_*/run_masked_seed*.npz`, no new runs |
| 5 | Score against effective sample size, with the `w_max` clamp marked | optional |

## Numbers the paper quotes (test set, n = 1024, `strong++`, stay 50.56, denominator 71.17)

| arm | seeds | return | norm | outage |
|---|---|---|---|---|
| `QD_strong` (BC + 8 DAgger, 40,960 solver calls) | 5 | 105.25 | 0.769 | — |
| `ft_hard` (filtered → 450 it) | 3 | 103.99 | 0.751 | 0.073 |
| `ft_bc` (clean-data BC → 450 it) | 3 | 103.45 | 0.743 | 0.078 |
| `P_hard` (offline, exact-advantage filtering) | 5 | 87.08 | 0.513 | — |
| `Q_strong` (offline, clean-data BC) | 5 | 84.30 | 0.474 | 0.065 |
| `P05_pairs` (exact advantage, soft) | 5 | 81.89 | 0.440 | — |
| `B_mix` (BC, mixed data) | 5 | 76.99 | 0.371 | — |
| `P_rand` | 5 | 74.24 | 0.333 | — |
| `P0_pairs` | 5 | 73.22 | 0.318 | — |
| `L125_pairs` (learned critic) | 5 | 68.72 | 0.255 | — |
| `M_mix` (MADT, mixed data) | 5 | 67.99 | 0.245 | 0.147 |
| `ft_scratch` (random → 450 it) | 3 | 58.78 | 0.116 | 0.134 |

Norms are ratio-of-means from the return column. Quality ladder, test: BC 75.35 / 79.92 / 78.44 / 84.30 and
BC+DAgger 78.42 / 89.69 / 94.47 / 105.25 at weak1 / weak2 / weak3 / strong. The online arms started from
`P_hard` and `Q_strong` seeds 0–2.

## Two claims withdrawn at five seeds

- "Filtering has a quarter of the seed variance of soft weighting" was a three-seed artefact: `P_hard` seed 3
  scored 0.553, widening its spread from 0.008 to 0.080. Filtering still beats soft weighting by 5.19 returns.
- "`P_hard` lands exactly where clean-data cloning lands" is now "at or marginally above": the gap to `Q_strong`
  grew from +1.13 to +2.78, still inside `Q_strong`'s seed range.

## Needed before submission

1. ~~A second offline algorithm~~ — done (result 4, MADT).
2. ~~Five seeds on the offline arms~~ — done 2026-11-14.
3. Optional: a wider return-to-go sweep so no MADT seed selects at an endpoint; the coverage-constrained online
   variant from the warm start (§4.10 reports the cost without testing the fix); the 4500-iteration arms (~52 h,
   defer and cite paper 1's catch-up result); the unclipped β = 3 point.

The experiments are complete; what remains is writing and the five figures.

## Relationship to paper 1

Same environment, planner, seed sets and reference convention (`strong++`, denominator 71.17 on test), so
numbers cross-reference directly — state once in §3.7. Paper 1 establishes that the observation specification
sets the ceiling; paper 2 asks what can be extracted from solver data *given* an observation, both at
`bcast = 6`. Paper 2 should cite paper 1's §4.6 for environment-size generality rather than re-arguing it.
