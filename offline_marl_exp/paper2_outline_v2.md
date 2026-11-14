# Paper 2 — outline v2

Replaces `claude/paper2_offline_marl_outline.md`, which predates almost every result in it. Run-level detail
and the correction history live in `claude/p2_dagger_quality_result.md` (result 1) and
`claude/p2_iql_result.md` (results 2 and 3); this document is the paper plan and carries the numbers the paper
should quote.

## The argument in three sentences

Classical solvers produce optimal trajectories too slowly to fly, so the pipeline distils them into a policy and
fine-tunes it online. Offline pre-training is worth **+45 returns** at a matched online budget, and a behaviour-cloned
start plus 450 MAPPO iterations reaches **97% of what DAgger reaches with 40,960 extra solver calls**. But the
offline algorithm barely matters: advantage weighting with an *exact* planner-computed advantage gains 12.2 returns
offline over plain cloning, a learned IQL critic gains nothing at any effective sample size, and after fine-tuning
all starts of equal offline quality are indistinguishable.

## Candidate titles

1. *Demonstration quality dominates offline algorithm choice: offline-to-online multi-agent RL from a classical
   trajectory solver*
2. *What offline reinforcement learning extracts from a classical solver, and what online fine-tuning does with it*
3. *Exact advantages, learned critics, and the limits of offline policy extraction on solver data*

(1) is the practitioner-facing claim and the most defensible; (3) oversells the offline half relative to the
evidence.

## Framing decision

Lead with offline-to-online, because the strongest positive result is the +45 returns, and the project was posed
that way. The offline comparison becomes the *mechanism* section rather than the headline. The middle of the
paper is a negative result, and the writing must make it load-bearing rather than apologetic: the branch gap of
−0.001 against a true +0.857 is a **measurement**, not an absence, and the `ft_hard` ≈ `ft_bc` null is what
licenses the paper's practical recommendation.

## Sections

**1. Introduction.** The sample-inefficiency of online MARL; classical solvers are optimal but take hours;
distillation plus fine-tuning as the bridge. State the three findings up front, including the negative one.

**2. Related work.** Offline MARL (IQL, CQL, decision transformers); learning from optimisation-based experts;
DAgger and distribution shift; offline-to-online fine-tuning. Position against the observability paper, which
shares the environment and is cited for it.

**3. Methods.**
- 3.1 Environment and objective — compressed, citing paper 1 for the full specification.
- 3.2 The planner, and the **measured** quality ladder (not asserted: `planner_check.py --ladder`).
- 3.3 Datasets. Single-trajectory, mixed-quality, and the **paired counterfactual** construction — two actions at
  every visited state, exact because the environment's whole mutable state is (positions, channels, t, previous
  action), so save/step/restore is exact. This is the paper's methodological contribution and deserves a figure.
- 3.4 Exact advantages. `Q_true(s,a) = r(s,a) + J(s′)` with `J` from the planner, so the advantage carries no
  estimation error and serves as ground truth for auditing a critic.
- 3.5 Offline extraction: BC, BC+DAgger, IQL/AWR, hard filtering, and the random-selection control.
- 3.6 Online fine-tuning: MAPPO, citing paper 1 §M10. **Flag explicitly** that `online2.py`'s defaults are not the
  protocol (lr_actor 3e-4 not 1e-3, critic_warmup 100 not 15).
- 3.7 Evaluation: three seed sets, cached references keyed by seed set and planner budget, ratio-of-means, the
  offline student as the unit of analysis.

**4. Results.** Ordered so the positive claim lands first and the mechanism explains it.

| § | claim | rests on |
|---|---|---|
| 4.1 | Demonstrator quality and relabelling are **complements** | Table 1 (quality ladder), Table 2 (BC vs BC+DAgger over 4 rungs, 5 students); the two slopes, 0.44 and 1.27 returns per demonstrator return; DAgger's gain falls 6.5× as the teacher weakens |
| 4.2 | Pre-training is worth **+45 returns** at a matched online budget | Table 3 (test set, 3 PPO seeds); `ft_hard` 103.99 and `ft_bc` 103.45 against `ft_scratch` 58.78 |
| 4.3 | Online RL reaches **97% of DAgger with zero solver calls** | `ft_bc` 103.45 against `QD_strong`; 288,000 environment steps against 40,960 receding-horizon solves |
| 4.4 | The offline method **does not transfer** | paired `ft_hard` − `ft_bc` = **+0.55** (SE 0.46) on test and **−0.50** (SE 0.53) on dev — the sign flips at half a return |
| 4.5 | Offline, exact advantages **do** help: +12.2 returns | Table 4 (offline arms, test); selection worth +10.74 at identical data volume (`P_hard` − `P_rand`); data volume itself free (`P_rand` − `P0_pairs` = +1.47, CIs overlap) |
| 4.6 | A **learned critic** recovers none of it | `L125_pairs` 3.6 returns *below* unweighted BC; learned arms bracket the solver arm's n_eff from 87.5% to 24.5% and none beats not weighting |
| 4.7 | **Why**: the critic ignores its action argument | Fig. 2 / Table 5 — branch gap **−0.001** against a true **+0.857**, identical to three decimals across three seeds while the state value wanders by 0.66 |
| 4.8 | The ceiling is **recovery, not improvement** | `P_hard` 85.43 ≈ `Q_strong` 84.30 ≈ BC on the clean half; filtering identifies and discards the bad half and stops there; DAgger sits 20.7 above |
| 4.9 | Fine-tuning buys return with **coverage** | outage 0.060 → 0.073 (`ft_hard`) and 0.065 → 0.078 (`ft_bc`); `ft_scratch` 0.134 and erratic |

**5. Threats.** Three PPO seeds and three-to-five offline students; one offline algorithm family (see Needed, below);
one MAPPO implementation; `ft_scratch` differs in architecture as well as initialisation (`mdn_k=0, chunk=1`);
450 iterations is a sample-efficiency claim, not a final-performance one; a single environment size, with paper 1's
scope section cited for how far that travels.

**6. Discussion.** The practitioner's sentence — *if you will fine-tune online, spend the budget on demonstration
quality, not on the offline algorithm; advantage filtering earns its keep only when you cannot tell which data is
good and will not fine-tune*. Then the sharper point: advantage-weighted BC **selects**, it does not **stitch**, so
its ceiling is BC-on-the-best-subset by construction. DAgger beats it because it buys *new* labels where the student
actually goes, which is a distribution-shift fix, not a reweighting.

**7. Conclusion.**

## Figures

| # | content | status |
|---|---|---|
| 1 | The paired-counterfactual construction: commit branch A, evaluate branch B, roll back | to draw |
| 2 | Advantage distributions, exact against learned, with the branch gap marked | to draw, data in hand |
| 3 | Offline arms on the test set, normalised, with intervals | to draw, data in hand |
| 4 | **Online learning curves**, three initialisations × three seeds, return against iteration | to draw — the `iters` and `returns` arrays are already in `results2/online/ft_*/run_masked_seed*.npz`, no new runs needed |
| 5 | Score against effective sample size, with the `w_max` clamp marked | optional |

Figure 4 is the one that will carry §4.2: the warm starts leave the critic-warmup plateau and climb immediately
while the cold start crawls, and it needs nothing but the files already fetched.

## Numbers the paper quotes (frozen test set, n = 1024, `strong++`, stay 50.56, denominator 71.17)

| arm | return | norm | outage |
|---|---|---|---|
| `ft_hard` (filtered → 450 it) | 103.99 | 0.751 | 0.073 |
| `ft_bc` (clean-data BC → 450 it) | 103.45 | 0.743 | 0.078 |
| `P_hard` (offline, filtered) | 85.43 | 0.490 | 0.060 |
| `Q_strong` (offline, clean-data BC) | 84.30 | 0.474 | 0.065 |
| `P_rand` | 74.69 | 0.339 | — |
| `P0_pairs` | 73.22 | 0.318 | — |
| `L125_pairs` | 69.58 | 0.267 | — |
| `ft_scratch` (random → 450 it) | 58.78 | 0.116 | 0.134 |

## Needed before submission

1. **A second offline algorithm — MADT.** The one real gap. Every offline claim rests on IQL/AWR, which invites
   the same criticism paper 1 took for having one MAPPO implementation. A decision transformer has **no critic**
   — it conditions on return-to-go — so it tests whether "the critic cannot resolve the action dimension" is
   specific to IQL or general to offline RL on solver data. `madt.py` exists and `pairs_train.pt` is unchanged.
   Either outcome is publishable; this is the experiment that decides how strong §4.6–4.7 may be stated.
2. **`Q_*` and `QD_*` arms on the frozen test set.** Result 1 is dev-only, so §4.1 and §4.3 currently mix seed
   sets — `QD_strong` is quoted at 106.92 on dev against `ft_bc` at 103.45 on test. One `final_eval.py` call with
   the cached `strong++` reference fixes it; no training.
3. **Five seeds on the offline arms.** `P_hard`, `P05_pairs`, `P_rand`, `L125_pairs` have three; `P0_pairs` has
   five. About an hour on a GPU.
4. Optional: the coverage-constrained online variant from the warm start (§4.9 currently reports the cost without
   testing the fix); the 4500-iteration arms (~52 h, defer and cite paper 1's catch-up result); the unclipped
   β = 3 point.

## Relationship to paper 1

Same environment, same planner, same seed-set definitions, same reference convention (`strong++`, denominator
71.17 on test) — so numbers cross-reference directly, which is worth stating once in §3.7. Paper 1 establishes
that the observation specification sets the ceiling; paper 2 asks what can be extracted from solver data *given*
an observation, and both use `bcast = 6`. Paper 2 should cite paper 1's scope section for environment-size
generality rather than re-arguing it.
