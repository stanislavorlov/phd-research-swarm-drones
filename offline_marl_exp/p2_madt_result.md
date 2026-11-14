# Paper 2, result 4: a second offline family, same ceiling — return-conditioning loses to plain cloning

Runs 2026-11-13/14. **Five seeds per arm.** Frozen test seeds 900000..901023 (n = 1024), `strong++` reference,
stay-put 50.56, denominator 71.17. Observation fixed at k = 6, k_a = k, `bcast = 6`, as in results 1–3.
MADT trains on CPU (`madt.py` has no device handling); `B_mix` seeds 3–4 ran on a GPU instance.

**Why this arm exists.** Every offline claim in result 2 rested on IQL/AWR, which invites the same objection
paper 1 took for having one MAPPO implementation. A Multi-Agent Decision Transformer is the strongest available
contrast: it has **no critic at all** — it conditions on return-to-go — so it tests whether "the critic cannot
resolve the action dimension" is a property of IQL or of offline extraction on solver data generally.

## Setup

`madt.py` was already in the repository and is unmodified; `run_madt.py` supplies the CLI, the return-to-go
sweep and the evaluation, because `final_eval.py` loads checkpoints through `bc2.load_student`, which builds a
`StudentActor` — MADT is a different class with a different rollout protocol (per-agent history plus a
return-to-go it tracks by subtracting realised reward). `commenv.run_episode` already calls
`policy.observe_reward(r)` when the policy defines it, so `MADTPolicy` drops into the standard evaluator.

**Dataset.** `expert_train_mix.pt` / `expert_val_mix.pt` — weak1 and strong trajectories over the *same* 600
scenarios, so the dataset holds two quality levels of the same states. MADT cannot use the paired
counterfactual dataset of result 2: `build_sequences` replays whole trajectories, and the paired set's branch B
is a one-step deviation with no continuation. Mixed-quality trajectories are in any case the setting where
return-conditioning could in principle beat selection, by conditioning on a high target.

**Return-to-go target** is a hyperparameter, swept over {100, 120, 140, 160} and selected on the **monitor**
seeds only (700000+), never on dev or test. Seed 0's sweep: 63.36 / 67.50 / 67.03 / 66.70. Selected targets
across the five seeds: 120, 120, 100, 120, 120.

**Control.** `B_mix` — behaviour cloning on the identical dataset, `student.py --rounds 0`, 300 epochs. Without
it a MADT number floats free; with it the comparison isolates the algorithm.

## Result (test set, five seeds)

| arm | what it is | return | norm |
|---|---|---|---|
| `P_hard` | exact-advantage filtering, paired set | **87.08** | 0.513 |
| `Q_strong` | BC, clean (strong) data | **84.30** | 0.474 |
| `B_mix` | BC, mixed data | **76.99** | 0.371 |
| `M_mix` | **MADT**, mixed data | **67.99 ± 1.46** | 0.245 |

Per-seed test returns for `M_mix`: 69.09, 68.70, 65.46, 68.66, 68.03. `B_mix` per-seed normalised scores
(mean-of-ratios): 0.337, 0.382, 0.376, 0.384, 0.310. Three-seed monitor figures, for the record: MADT
67.24 ± 1.33 against BC 77.24 ± 1.66, non-overlapping.

**MADT is 9.0 returns below plain behaviour cloning on the identical dataset.** Return-conditioning does not
separate the two quality levels; it costs performance relative to cloning the mixture indiscriminately. Outage
agrees: 0.174 / 0.134 / 0.102 / 0.152 / 0.175, mean 0.147, against roughly 0.06 for the BC students.

## Mechanism: the fit is fine, the rollout is not

MADT's **teacher-forced** validation position-MSE runs 0.0931 → 0.0530 → 0.0498 → 0.0479 → 0.0463 → 0.0459 over
25 epochs, flat by epoch 20 — the last five epochs buy 0.0004. It is not undertrained, and it models the
demonstrations well.

That figure must **not** be compared with `B_mix`'s validation `mse0` of 0.136. MADT is graded with teacher
forcing — `forward_emb` receives the true action history and predicts each step given the real past — while the
BC student predicts from the observation alone. MADT is being scored on a strictly easier problem, and still
acts worse.

The failure is therefore in closed loop, where `MADTPolicy` conditions on **its own** past actions *and* on a
return-to-go it must track by subtracting realised reward. Errors compound in two coupled channels at once, and
asking for a higher return makes it act more aggressively without earning the return — seed 0's outage rises
0.132 → 0.187 → 0.222 → 0.246 across targets 100 to 160. This is the same distribution-shift mechanism the paper
uses to explain why filtering cannot reach DAgger: DAgger fixes it by buying new labels where the student
actually goes, and nothing that reweights or re-conditions a fixed dataset addresses it.

## What this does for the paper

The claim stops being about IQL. **Two offline families built on different principles — one estimating
advantages with a critic, one conditioning on return-to-go with no critic — and neither beats plain cloning.**
The only method that helps is selection with an advantage computed *exactly* by the planner, and it lands at or
marginally above cloning the clean subset (`P_hard` 87.08 against `Q_strong` 84.30). The ceiling becomes a
property of the data rather than of one algorithm.

## Caveats to state

- **Head mismatch.** `M_mix` has `mdn_k = 0, chunk = 1`; `B_mix` has `mdn_k = 5, chunk = 4`. The defence is the
  mechanism above: MADT's teacher-forced fit is excellent and flat, so the deficit is not capacity or fit.
- **One seed of five selected at the sweep's lower endpoint** (seed 2, target 100), so its hyperparameter was
  not bracketed. The four interior-selected seeds average 68.62, still 8.4 below `B_mix`. A wider sweep would
  tidy this; it would not change the conclusion.
- 25 epochs against the BC arms' 300 — justified by the flat validation curve, not by budget.

## Reproduce

```bash
# train, with the target swept and selected on monitor seeds only
python3 run_madt.py --name M_mix --seed 0 \
  --train_data data/expert_train_mix.pt --val_data data/expert_val_mix.pt \
  --k 6 --k_a -1 --bcast 6 --d 128 --epochs 25 --targets 100 120 140 160

# the control: BC on the identical dataset
python3 student.py --name B_mix --rounds 0 \
  --train_data data/expert_train_mix.pt --val_data data/expert_val_mix.pt \
  --k 6 --k_a -1 --bcast 6 --head mdn --d 128 --mdn_k 5 --chunk 4 --epochs0 300 --seed 0

# scoring: MADT needs its own evaluator and the observation flags repeated,
# because --eval_only rebuilds the config from the command line, not the checkpoint
python3 run_madt.py --name M_mix --seed 0 --eval_only --seed0 900000 --n_final 1024 \
  --planner strong++ --k 6 --k_a -1 --bcast 6 --d 128
python3 final_eval.py --names B_mix --n 1024 --seed0 900000 --planner strong++ --tag mix_test
```

`expert_train_mix.pt` predates the `expert.py` RNG fix of 2026-11-01, so it is valid data but cannot be
regenerated bit-identically; note that in the reproducibility statement.
