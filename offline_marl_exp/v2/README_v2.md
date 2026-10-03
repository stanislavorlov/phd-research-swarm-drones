# Offline→online MARL for UAV swarms — v2 rebuild

This folder replaces the pipeline critiqued in `pipeline_review.md`. The v1 files in the parent folder are untouched and should be
treated as superseded. Everything below is measured by code in this folder; nothing is inferred.

## Bottom line

* The pipeline now supports a **narrow, honest claim**: a decentralised student distilled from a centralised planner clears every trivial floor with a wide,
  CI-separated margin, and **online MAPPO fine-tuning measurably and reproducibly improves it** (+14 to +16 return, 5/5 seeds).
* It does **not** support "the masked critic beats the unmasked critic": the difference is −1.6 return with a CI that includes 0 (n = 5 seeds), and the design
  could only resolve ≈ 2.8. Report it as "the mask costs nothing detectable", exactly as the review suggested as the safe framing.
* The distilled student is **weak in absolute terms**: 21 % of the expert's margin over do-nothing (≈ 42–45 % after fine-tuning). It is not "optimal at neural speed" yet.
* From-scratch online RL is far worse and unstable at the same sample budget (below do-nothing on average), which is the sample-efficiency half of your original motivation — a genuine, now-measurable result.

## Results (all deterministic evaluation on 256 held-out scenarios, same scenarios for every policy; return = sum of team reward over 20 steps)

| policy | return | expert-normalised* |
|---|---|---|
| random targets | 46.1 | — |
| do nothing (stay-put) | 51.2 | 0 |
| **BC student only** (5 seeds) | 65.0 | **0.21** |
| **BC → MAPPO, k-masked critic** (5 seeds) | 79.4  [78.6, 80.3] | 0.42 |
| **BC → MAPPO, unmasked critic** (5 seeds) | 81.0  [79.7, 82.2] | 0.45 |
| scratch → MAPPO, masked critic | 47.7  [37.3, 56.7] | −0.05 |
| scratch → MAPPO, unmasked critic | 37.6  [30.6, 43.3] | −0.20 |
| expert, weak (1 start, 1 round) | 100.5 | 0.76 |
| **expert, strong** (planner, 4 starts × 4 BCD rounds) | 118.0 | 1.00 |
| interference-free relaxation (upper-bound *proxy*, heuristic) | 156.2 | — |

\*(return − stay-put)/(expert − stay-put). Brackets = 95 % bootstrap CI **over training seeds** (the unit of analysis), n = 5.

Paired differences at iteration 150 (same training scenarios and noise stream per seed; CI over seeds):

| contrast | Δ return | 95 % CI | verdict |
|---|---|---|---|
| online gain, masked critic (vs BC-only) | +14.4 | [+13.6, +15.3] | clear |
| online gain, unmasked critic (vs BC-only) | +16.0 | [+14.3, +17.5] | clear |
| BC init vs scratch (masked critic) | +31.8 | [+23.4, +41.6] | clear |
| **mask effect, BC init** (masked − unmasked) | **−1.6** | [−3.3, +0.1] | **not separated from 0** |
| mask effect, scratch init | +10.0 | [−1.3, +24.1] | not separated (huge variance) |

Minimum detectable effect for the mask contrast with 5 seeds (observed seed-level paired SD 2.2, 80 % power): **≈ 2.8 return**. The effect size was **not pre-registered**;
the MDE is computed after the fact from the observed SD.

**Confirmation on untouched seeds** (`confirm.py`, seeds 500000–500255; no design choice ever looked at them): the same picture — BC-only 63.1, BC→MAPPO masked 79.2, unmasked 79.7,
mask effect −0.45 [−2.3, +1.7], init effect +31.7 [+23.4, +41.7], online gain +16.1 [+14.6, +17.6]. See `results/confirm_fresh_seeds.txt`.

Speed (CPU, one thread): student **≈ 0.5 ms per decision step for the whole swarm** (≈ 11 ms per 20-step episode) vs **0.94 s to solve one scenario** with the planner (≈ 85×).
The planner is not a MINLP solver, so this is *not* the "hours vs milliseconds" comparison of the original pitch — see limitations.

Far-field influence (`measure_Lk.py`, `results/Lk_vs_bound.png`): hiding every interferer beyond k of the serving UAV costs, per user, 1.56 bit/s/Hz at k = 6 (≈ a quarter of the rate),
0.96 at k = 10, 0.21 at k = 15, 0.006 at k = 22. Pairwise influence decays with slope −2.51 against a path-loss exponent α = 2.5. The per-state perturbation lemma (Prop. 1) evaluated on the measured
interference tracks the measured loss to within ≈ 5 %; the worst-case aggregate bounds (Prop. 2) are **valid but 5–1000× loose**.

Corrected Decision Transformer (`madt.py`): **worse than plain BC in closed loop** — 52.3 / 53.0 / 54.0 for 3 seeds (vs 65.0 for BC), +1 to +3 over do-nothing — even though its teacher-forced
position MSE is lower (0.054 vs 0.083). My *untested* hypothesis is exposure bias / copying its own previous action. It was not tuned; BC is the primary student.

## What was wrong, and where each item went (review ID → status)

| ID | defect | status |
|---|---|---|
| F1 | actor gradient identically 0 | **fixed**: stochastic policy, stored behaviour log-probs, clipped ratio; `test_G1`, `test_G1b` (ratio = 1 on first pass, then moves), `test_G2` |
| F2 | two arms were the same network | **fixed**: arms differ by one factor each; `test_G4`; masked/unmasked critics are the same class with identical init, only the attention mask differs |
| F3 | BC target was an input feature | **fixed**: target = expert *displacement*; `test_G0b` checks no input column equals/linearly copies it; student beats stay-put (`test_G3`) |
| F4 | DT saw the action it predicts | **fixed** in `madt.py` (interleaved tokens, causal mask); `test_madt_has_no_label_leakage` |
| F5 | train/deploy observation mismatch | **fixed**: observations are regenerated by replaying the expert through the env |
| F6 | dataset key/file mismatch | **fixed**: one generator (`expert.py`), seeds + actions stored, one loader |
| S1 | no trust region / GAE / minibatches | **fixed** in own MAPPO (clip 0.2, KL early-stop, GAE 0.99/0.95, 3 epochs × 2 minibatches, entropy). **Not HARL** — see limitations |
| S2 | unpaired, underpowered, training-reward eval | **fixed**: paired scenarios, deterministic held-out evaluation at fixed checkpoints, 5 seeds, CI over seeds, three reference lines |
| S3 | arms differ in two factors | **fixed**: init and critic mask are separate factors (2×2) |
| D1 | mask hid nothing the reward depends on | **fixed**: no hard interference cutoff, long-range coupling, no entity cap; L(k) is now non-trivial and measured |
| D2 | static placement, constant targets | **fixed**: drifting user clusters, planner outputs genuine trajectories; (caveat: expert is a heuristic, below) |
| D3 | facility-location toy | **fixed**: path loss, SINR from all co-channel UAVs, Shannon rate per user, nearest-UAV association, propulsion-energy price |
| D4 | zero-padded unsorted MLP input | **fixed**: relative normalised features, presence masks, attention + sum-pool over variable-length sets |
| §4 theorem | | rewritten as `gap_bound_theorem_v2.md` (see its §4 table); what is proved vs open is marked |

A bug in **my own** first draft of the unmasked critic's mask (wrong on flattened batches) was caught by the new tests before any run used it.

## Limitations — read before quoting any number

1. **The expert is a heuristic, not a certified optimum.** BCD with restarts and an exactly-matched objective; the "upper-bound proxy" is itself a heuristic, so the 76 % figure is an *estimate*,
   not a certificate. No MINLP/SCA solver or optimality gap is reported (review §5 step 4 only partly met). The weak/strong expert numbers show expert quality matters; the planned
   "expert quality at fixed data volume" ablation was not run.
2. **The student is underfit** (loss still falling at 40 epochs; a single-seed k = ∞ pilot did no better, which suggests optimisation/budget rather than information). DAgger was not implemented, and it is the obvious next step
   (it is also what the O(T) vs O(T²) horizon discussion in the theorem note points to).
3. **Own MAPPO, not HARL.** The sanity gates validate the mechanics (ratio, gradients, masks, arms), not parity with a reference implementation. Re-run phase 3 with HARL for any number you publish.
4. **Online budget is small** (150 iterations × 32 scenarios, one config N = 8 UAVs, U = 32, C = 3). The masked-critic curve has plateaued by iteration 120; the unmasked one was still creeping up. Scratch-init runs are noisy
   (seed spread ≈ 20 return) so any claim about them needs more seeds.
5. **No k-sweep of learners yet.** The review's k ∈ {1, 2, 3, ∞} ablation was not run for BC / MAPPO; only the model-level L(k) was measured. The student uses k = 6 throughout. ε_info(k) (teacher unrealisability) is not measured.
6. **Design choices were tuned while looking at part of the main eval set** (seeds 300000–300127: energy price 0.5, BC epochs/lr; symmetry augmentation was left on after a single-seed pilot showed no measurable effect). The confirmation on untouched seeds above agrees, but treat the main table as exploratory.
7. **Environment is still a simplified comms model**: equal transmit power, per-user dedicated bandwidth, no fading, nearest-UAV association, fixed altitude. Better than v1; reviewers at a comms venue will still ask for more (A2G LoS probability, fading, power control).
8. **Theory**: the arXiv numbers in `gap_bound_theorem_v2.md` are taken from the review and were not re-verified by me. Proposition 3's T(T+1)/2 is an upper bound that is probably loose here (recoverable states).
9. Environment constants and observation features (e.g. energy price, user-velocity observation) are modelling choices, not derived quantities.

## Reproduce

```bash
pip install torch numpy scipy matplotlib pytest
python -m pytest tests -q                                  # gates; 10 tests, ~10 s
python expert.py --n 600 --start 100000 --out data/expert_train.pt      # ~2.5 min, CPU
python expert.py --n 100 --start 200000 --out data/expert_val.pt
python run_experiment.py refs                              # stay / random / experts / UB proxy / timings
python run_experiment.py bc --seeds 0 1 2 3 4              # ~7 min per seed; gate G3 asserts student > floors
python run_experiment.py online --seeds 0 1 2 3 4 --iters 150   # 20 runs, ~9 min each on one core
python run_experiment.py madt --seeds 0 1 2                # optional offline DT arm
python run_experiment.py analyze                           # table + results/online_comparison.png
python measure_Lk.py                                       # results/Lk_vs_bound.png
python confirm.py                                          # fresh-seed confirmation
```

## Files

| file | role |
|---|---|
| `commenv.py` | batched SINR environment, scenarios, masked set observations, reference policies |
| `expert.py` | BCD planner with restarts, channel canonicalisation, replay-consistency check, UB proxy |
| `models.py` | `SetEncoder`, `Actor`, `Critic` (masked/unmasked = a flag) |
| `bc.py` | behaviour cloning (displacement target, env-replayed observations, D4 augmentation) |
| `mappo.py` | MAPPO update, collection, training loop |
| `madt.py` | corrected decentralised Decision Transformer |
| `evalproto.py`, `analyze.py`, `confirm.py` | paired evaluation, seed-level CIs, figure, fresh-seed check |
| `measure_Lk.py` | empirical L(k) vs bounds, pairwise influence |
| `gap_bound_theorem_v2.md` | repaired theorem note |
| `tests/test_gates.py` | the sanity gates |
| `results/` | figures, tables, per-run histories (`run_*.npz`), logs, checkpoints (bundled in `runs_and_checkpoints.tar.gz`) |
