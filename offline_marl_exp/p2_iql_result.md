# Paper 2, result 2: offline RL works on solver data; advantage *estimation* is the bottleneck

Supersedes the 2026-10-29/30 version of this doc, which concluded that advantage-weighted offline RL cannot
improve on solver data at all. That conclusion was wrong in two ways, both recorded below under "Corrections".
Runs 2026-10-31 to 2026-11-14. Test seeds 900000..901023 (n = 1024), stay-put 50.56. Observation fixed at
k = 6, k_a = k, `bcast = 6`. **Offline arms have five seeds as of 2026-11-14**; the table and claims below use
them. The second offline family (MADT) is in `claude/p2_madt_result.md`.

**Reference convention.** Reported under `--planner strong++` (restarts 32, rounds 16, adam_iters 160), which
gives expert 121.73 and denominator 71.17 on test — identical to paper 1, so the two papers' normalised scores
are comparable. On the dev set the same reference gives stay-put 51.08 and denominator 71.48. `--planner strong`
(4/4/80) gives test expert 117.76, denominator 67.20; those runs are in
`results2/final/final_eval_test_strong.json`. The ladder does not converge, so the choice is a reporting
convention and must be stated. All differences quoted below are in raw returns and are independent of it.

## Headline

Two findings, and the second qualifies the first.

**Offline.** On a dataset with two actions per visited state, advantage-weighted extraction gains **+13.9
returns** over unweighted BC when the advantage is computed exactly by the planner, and **nothing at any
effective sample size tried** when it is estimated by a learned IQL critic — the sharpest learned setting scores
4.5 returns *below* unweighted BC. The machinery is not the problem; the critic is. But the gain is recovery,
not improvement: the filtered student lands at or marginally above BC on the good half alone.

**Online.** Offline pre-training is worth **+45 returns** at a matched budget of 450 MAPPO iterations. Which
offline method produced the initialisation makes **no difference** — filtered and plain-BC starts are
indistinguishable after fine-tuning (paired difference +0.55, SE 0.46 on test; −0.50, SE 0.53 on dev). The
offline advantage does not compound.

## Test-set results, offline (five seeds)

Normalised scores are ratio-of-means, `(return − stay)/(expert − stay)`, the convention paper 1 uses. Note that
`final_eval.py`'s `norm` column is the *mean of per-scenario ratios*, a different and systematically lower
estimator; the figures below are derived from the `return` column.

| arm | advantage | rows used | n | return | norm |
|---|---|---|---|---|---|
| `P_hard` | exact, filtered | 48,000 (50%) | 5 | **87.08** | **0.513** |
| `P05_pairs` | exact, soft β=0.5 | 41,302 eff. (43%) | 5 | 81.89 | 0.440 |
| `P_rand` | none, random half | 48,000 (50%) | 5 | 74.24 | 0.333 |
| `P0_pairs` | none, all rows | 96,000 (100%) | 5 | 73.22 | 0.318 |
| `L125_pairs` | learned, β=1.25 | 23,500 eff. (24.5%) | 5 | 68.72 | 0.255 |

Per-seed mean-of-ratios as printed: `P_hard` [0.473, 0.481, 0.475, 0.553, 0.509]; `P05_pairs`
[0.443, 0.387, 0.414, 0.444, 0.444]; `P_rand` [0.312, 0.351, 0.299, 0.293, 0.318]; `L125_pairs`
[0.249, 0.296, 0.215, 0.205, 0.241]. Three-seed figures, superseded: 85.43 / 81.13 / 74.69 / 69.58.

Four claims, each isolating one variable:

1. **Selection is worth +12.84 returns at identical data volume.** `P_hard` − `P_rand`: same 48,000 rows, same
   architecture, same 300 epochs; selection of the better branch replaced by a coin flip. CIs do not overlap.
2. **Data volume is free.** `P_rand` − `P0_pairs` = +1.02 returns with overlapping CIs. Halving the dataset at
   random costs nothing, so the dataset is saturated and no part of the gain is attributable to sample count.
3. **Exact advantage beats learned advantage by +13.17 returns**, and the learned arm scores 4.50 returns
   *below* the unweighted baseline. This is **bracketed, not matched**, on effective sample size: the learned
   arms span the solver arm's 43% from both sides — on the 128-scenario monitor set (three seeds), n_eff
   87.5% → 72.06, 70.8% → 71.85, 24.5% → 69.35, against an unweighted baseline of 71.98 — so no learned setting
   beats not weighting at all, and the sharpest actively hurts. Weighting by a meaningless advantage reallocates
   gradient away from good rows.
4. **Filtering beats soft weighting** by +5.19 returns. The earlier statement that it also has a quarter of the
   seed variance was a three-seed artefact and is withdrawn: `P_hard`'s spread went from 0.008 to 0.080 when
   seed 3 scored 0.553, comparable to the soft arm's.

## The ceiling is recovery, not improvement

On the test set with five seeds each, `P_hard` is 87.08 against `Q_strong` (BC on strong data only) at 84.30 —
**+2.78 returns**, inside `Q_strong`'s own seed range of 81.5 to 86.6. At three seeds the gap was +1.13 on test
and +0.80 on dev. Read it as "at or marginally above clean-data cloning", not "exactly at".

The dev-set comparison with result 1 (800000..800255, n = 256, three seeds for `P_hard`):

| arm | dev return | what it is |
|---|---|---|
| `QD_strong` | 106.92 | BC + 8 DAgger rounds on strong data |
| `P_hard` | **86.24** | exact-advantage filtering on the 50/50 paired set |
| `Q_strong` | **85.44** | BC on strong data only |
| `P0_pairs` | 73.35 | BC on the whole paired set |

Starting from the mixed set, filtering recovers 13 to 14 returns and arrives where BC on the good half alone
arrives. It identifies and discards the bad half, and does not meaningfully exceed the best demonstrator present.

So the mechanism is **selection, not stitching**. Advantage-weighted BC chooses among the actions in the data; it
cannot compose an action better than any it was shown, and its ceiling is therefore BC-on-the-best-subset, which
is where it landed. DAgger, which buys 40,960 *new* solver labels at states the student actually visits, is 18
returns above on test (105.25) — that gap is distribution shift, and no amount of reweighting a fixed dataset
addresses it.

**What this means for the contribution.** On clean data, run BC. The value of exact-advantage filtering is that
*you do not have to know which data is good* — the planner's own advantages identify it, at ~100 s for 96,000
rows. That is the defensible claim; "offline RL beats imitation learning" is not.

## Online fine-tuning: large, and indifferent to its starting point

`online2.py`, masked critic, 450 iterations at 32 envs × 20 steps = 640 team steps per iteration, so **288,000
team steps**; lr_actor 3e-4, init_std 0.2, critic_warmup 100 (the actor is frozen for the first 100 iterations,
so 450 iterations is 350 actor updates), target_kl 0.03. Three PPO seeds, each paired to one offline student
(seeds 0–2 of `P_hard` and `Q_strong`).

**Test set** (`online_test_eval.py`, n = 1024, `strong++`, denominator 71.17) — the figures the paper quotes:

| arm | initialisation | per-seed | mean | norm | outage |
|---|---|---|---|---|---|
| `QD_strong` | BC + 8 DAgger (40,960 solver calls) | — | 105.25 | 0.768 | — |
| `ft_hard` | `P_hard` + 450 it | 104.77 / 102.66 / 104.55 | **103.99** | 0.751 | 0.073 |
| `ft_bc` | `Q_strong` + 450 it | 104.60 / 102.65 / 103.09 | **103.45** | 0.743 | 0.078 |
| `Q_strong` | offline only | — | 84.30 | 0.474 | 0.065 |
| `ft_scratch` | random + 450 it | 58.47 / 56.77 / 61.11 | **58.78** | 0.116 | 0.134 |

**Dev set** (`--final`, n = 256, denominator 71.48), for comparison with result 1's original table: `ft_bc`
106.57 / 102.55 / 104.56 = 104.56; `ft_hard` 105.22 / 101.93 / 105.04 = 104.06; `ft_scratch` 59.07 / 57.74 /
60.38 = 59.06; `QD_strong` 106.92.

1. **Pre-training is worth +45.2 returns at a matched online budget** on test (`ft_hard` − `ft_scratch`; +44.7
   for `ft_bc`; +45.0 and +45.5 on dev). This is the offline-to-online claim, and it is unambiguous.
2. **The offline method does not transfer.** Paired by seed, `ft_hard` − `ft_bc` is +0.17, +0.01, +1.46 on
   test, mean **+0.55** with SE 0.46; on dev it is −1.35, −0.62, +0.48, mean **−0.50** with SE 0.53. The sign
   flips between seed sets at half a return, which is what a null looks like. Seed 1 is the low seed in *both*
   warm arms, so most of the spread is the PPO seed rather than the initialisation.
3. **Online RL reaches 98% of DAgger with zero solver calls.** `ft_bc` at 103.45 against `QD_strong` at 105.25
   on the same test seeds, having spent 288,000 environment steps and no receding-horizon solves against
   DAgger's 40,960.
4. **Outage rises as return does**, 0.060 → 0.073 for `ft_hard` and 0.065 → 0.078 for `ft_bc` — the same
   return/coverage trade paper 1 documents in its §4.4. `ft_scratch` is both worse and far more erratic in
   coverage (0.134 on test; 0.205 / 0.095 / 0.111 per seed on dev).

**The practitioner's sentence.** If you are going to fine-tune online, spend the budget on getting good
demonstrations, not on sophisticated offline RL: BC on your best data reaches the same place. Advantage
filtering earns its keep only when you cannot tell which data is good and will not fine-tune afterwards.

Caveats. `ft_scratch` is built with `mdn_k=0, chunk=1`, a plain Gaussian head, while the warm arms carry the
student's converted MDN and chunk heads — unavoidable in a warm-versus-cold comparison but it means the
45-return gap is initialisation *and* architecture. The claim is sample efficiency at a fixed budget, not final
performance: paper 1 found from-scratch overtaking the warm start by 4500 iterations at `bcast = 0`, and that
was not tested here.

## Mechanism: the critic cannot resolve the action dimension

The paired dataset puts two actions at every visited state, so for a pair sharing observation `o` the state value
cancels exactly and the measured branch gap is `mean(Q(o,a_A) − Q(o,a_B))`.

| source | branch gap | advantage sd |
|---|---|---|
| planner (exact, `solver_advantages.py`) | **+0.857** (sd 1.933) | 1.057 |
| learned IQL critic, 3 seeds | **−0.001 / −0.000 / −0.000** | 0.721 |

Across three seeds the learned critic's committed and counterfactual branch means are identical to three decimals
(−0.194/−0.194, −0.857/−0.857, −0.742/−0.742) while the absolute level wanders by 0.66 between seeds. The critic's
state value is unstable and seed-dependent; its action discrimination is exactly zero. It has not learned a wrong
action-value — it has learned a function that barely depends on its action argument, so the advantage handed to
AWR is close to pure state-value noise, and no β can extract an ordering that is not there.

This also explains the narrower spread: a Q that ignores `a` yields advantages driven only by `Q(o) − V(o)`,
i.e. expectile-vs-mean disagreement about the *state*, which is smaller than the true action gap.

The ground-truth signal is real and was verified independently (`true_advantage.py`, `Q_true = r + J(s′)`, no
critic): mean advantage +1.187, decomposing into +0.269 immediate and +0.918 future, decaying 2.36 → 0.39 with
horizon. Over the paired dataset the solver gap is +0.857 ± 1.933 with the weak action better at **20.6%** of
states, implying a demonstrator-improvement ceiling of ~0.42/step ≈ 8 returns.

## β is not a reportable axis

n_eff falls roughly as `exp(−(β·sd)²)` while β·sd is small, and the clamp distorts the top end. With solver
advantages (sd 1.057, bounded at p1/p99 = ∓3.382), β = 3 and `w_max = 100` puts over 1% of rows on the clamp —
`p99 == max` — which *de*-sharpens the distribution and pushes n_eff back up from 7.5% (β=1) to 12.2% (β=3).
β = 3 is a truncated version of β = 1, not a sharper one. Report against n_eff with the clamp stated, never
against β.

Single-seed monitor scores for the β sweep (86.02 / 81.68 / 84.42 at β = 0.5 / 1 / 3) were over-read once; the
soft arm's seed sd of 4.01 on the monitor set is as large as the entire spread being interpreted.

For the learned critic the same relation was used to aim at a matched n_eff: sd 0.721, target 43%, giving
β = 0.919/0.721 ≈ 1.25. **The estimate overshot.** Measured n_eff was 87.5% at β = 0.5, 70.8% at β = 0.75 and
**24.5%** at β = 1.25, against the 43% predicted — the approximation holds at small β·sd and fails in the tail,
because the learned advantage distribution is not lognormal there. Consequence: the exact-versus-learned
comparison is bracketed rather than matched, as stated in claim 3.

## Corrections to the previous version of this doc

1. **A fall-through bug invalidated the first exact-advantage runs.** In the `--solver_adv` patch, the line
   "skip phase 1 and the learned-weight block; go straight to the actor" was written as a *comment* where code
   was required. Execution fell out of the `if a.solver_adv:` block into the critic phase and `W` was
   subsequently overwritten by the learned advantages — diagnosable from two `AWR weights` lines per run with
   different n_eff (43.0% solver, 88.3% learned at β=0.5). The four runs reported as a flat exact-advantage
   sweep (73.55 / 73.35 / 73.53 / 71.23) were learned-advantage runs. `iql.py` now selects the weight source
   with a single ternary over two functions, so the critic cannot be constructed when `--solver_adv` is set, and
   `summary.json` records `critic_epochs: 0` on that path.
2. **The single-action-per-state mechanism was correct but not the whole story.** On single-trajectory solver
   data there genuinely is one action per state and the advantage cannot be meaningful. But supplying
   counterfactual actions does *not* fix it: the paired dataset has real diversity (channel agreement 0.877,
   mean action displacement 0.788) and the learned critic still reports a branch gap of −0.001. So the earlier
   prediction — "IQL should beat BC once the advantages distinguish real alternatives" — failed. The
   alternatives are present; the critic cannot see them.
3. **`P0_pairs` = 73.55 and related pre-fix figures should not be quoted.** Re-running overwrote
   `summary.json` for several seeds; the pooled β = 0 baseline over five seeds is 73.22 on test
   (71.98 on the 128-scenario monitor set).
4. Stale learned-critic seeds were contaminating `P05_pairs` (seeds 3, 4) and `P1_pairs` (seeds 1–4), since
   `final_eval.py` globs every `seed*` under a name. `P05_pairs` seeds 3 and 4 were re-run with solver
   advantages on 2026-11-14 and overwrite the stale ones; `P1_pairs` seeds 1–4 on the Mac are still the stale
   learned-critic runs and that arm should not be quoted. Use a distinct `--name` per experiment.
5. The denominator mismatch with paper 1 (67.20 vs 71.17 on identical seeds) was the reference-planner key,
   not environment drift: `strong` vs `strong++`. Resolved — stay-put agrees at 50.56 under both.
6. **Result 1's pre-registered prediction failed.** It expected offline RL to show "the mirror image of DAgger's
   pattern: largest advantage over BC at the weak rungs… IQL can in principle exceed its teacher by stitching."
   Filtering cannot stitch, so it should help *least* at a weak rung, where there is no good subset to recover.
7. **`L125_pairs` was reported at 43% n_eff; it is 24.5%.** The lognormal extrapolation used to pick β = 1.25
   overshot, so the learned arm is sharper than intended and the comparison is bracketed rather than matched.
   The conclusion is unchanged — no learned n_eff beats the unweighted baseline — but "matched n_eff" was the
   wrong description and has been removed.
8. **Two three-seed statements withdrawn at five seeds (2026-11-14).** "Filtering has a quarter of the seed
   variance of soft weighting" and "`P_hard` lands exactly where clean-data cloning lands". See claim 4 and the
   recovery section.

## Open

- **Does from-scratch catch up?** The 450-iteration result is sample efficiency, not a final-performance claim.
  Paper 1 found from-scratch overtaking the warm start by 4500 iterations at `bcast = 0`. On a contended
  instance at ~7 s/iteration, six 4500-iteration runs are ~52 hours; defer unless a reviewer asks, and cite
  paper 1's precedent meanwhile.
- **A stitching test**, to settle whether offline RL offers anything beyond selection: a dataset whose best
  achievable trajectory is not any single demonstrator's, so improvement *requires* composing across them.
- Whether filtering helps at all on the weak rungs of result 1's quality ladder — the direct test of correction 6.
- The coverage-constrained online variant (`--shortfall_coef`, `--rate_floor 2`) from the warm start, since
  outage rises across fine-tuning exactly as in paper 1.
- `final_eval.py` should print both normalisation columns (mean-of-ratios and ratio-of-means).
- Unclipped β = 3 (`--w_max 1e9`) for the shape of the n_eff curve. Low priority.
- Nothing here tests a *learnable* fix for the critic (action-conditioned architecture, centralised full-state
  Q, larger capacity). `--critic_full` fit roughly 2× better on the mixed dataset but still gave a +0.012 gap.
  A reviewer will ask whether a better critic closes the gap; the honest answer is that we showed the estimate
  is the bottleneck, not that no estimator can work.

## Reproduce

```bash
# exact advantages, soft
python3 iql.py --name P05_pairs --beta 0.5 --pairs_data data/pairs_train.pt \
  --solver_adv data/adv_train.pt --k 6 --k_a -1 --bcast 6 --head mdn \
  --d 128 --mdn_k 5 --chunk 4 --epochs0 300 --seed 0

# exact advantages, filtered; and the fixed-data-volume control
python3 iql.py --name P_hard --weight_mode hard --beta 0 ...   # same flags
python3 iql.py --name P_rand --weight_mode rand --beta 0 ...

# learned critic (n_eff 24.5% at this beta, not the 43% originally intended)
python3 iql.py --name L125_pairs --beta 1.25 --pairs_data data/pairs_train.pt \
  --k 6 --k_a -1 --bcast 6 --head mdn --d 128 --mdn_k 5 --chunk 4 \
  --epochs0 300 --critic_epochs 60 --seed 0

# clean-data BC baseline, the provenance control for the online arms
python3 student.py --name Q_strong --rounds 0 --train_data data/expert_train.pt \
  --val_data data/expert_val.pt --k 6 --k_a -1 --bcast 6 --head mdn --d 128 \
  --mdn_k 5 --chunk 4 --epochs0 300 --seed 0

# online: paper 1's protocol. online2.py's own defaults are NOT paper 1's
# (--lr_actor defaults to 1e-3, --critic_warmup to 15); both must be passed.
ONL="--seeds 0 1 2 --arms masked --iters 450 --lr_actor 3e-4 --init_std 0.2 \
     --critic_warmup 100 --final --out results2/online"
python3 online2.py --ckpt "results2/P_hard/seed{s}/best.pt"   $ONL --tag ft_hard
python3 online2.py --ckpt "results2/Q_strong/seed{s}/best.pt" $ONL --tag ft_bc
python3 online2.py --ckpt "results2/P_hard/seed{s}/best.pt"   $ONL --tag ft_scratch --scratch

# test-set scoring
python3 final_eval.py --names P_hard P_rand P05_pairs L125_pairs B_mix P0_pairs \
  --n 1024 --seed0 900000 --planner strong++ --tag offline5_test
python3 online_test_eval.py --tags ft_hard ft_bc ft_scratch --seed0 900000 --n 1024 --planner strong++ \
  --ckpt_map "ft_hard=results2/P_hard/seed{s}/best.pt" "ft_bc=results2/Q_strong/seed{s}/best.pt" \
             "ft_scratch=results2/P_hard/seed{s}/best.pt,scratch" --start_names P_hard Q_strong
```

`--beta 0` with `--solver_adv` is the implementation check: weights are exactly 1, n_eff 100%, no critic trained.
`online2.py` skips any seed whose `.npz` already exists, so an interrupted run resumes rather than repeating.
