# RUNBOOK v3 — improving the BC student (items 1–5), for Vast.ai

**Status: all code in this runbook was written but NOT executed by its author** (you asked me not to run anything).
Steps 1–2 below are therefore real tests of new code. Time figures are guesses from one old CPU measurement
(100-scenario full-horizon plan ≈ 22 s on 2 CPU cores); treat them as order-of-magnitude.

What each item maps to (flags in `student.py`):

| item | what | flag |
|---|---|---|
| 1 | DAgger with a receding-horizon expert (planner restarted from the student's own states) | `--rounds R --scen 256 --beta0 0.5` |
| 2 | bigger budget / capacity / EMA (EMA always on) | `--epochs0 300 --d 128` (+ underfit diagnostic in the log: train vs val mse0) |
| 3 | mixture-density head + action-chunk auxiliary loss | `--head mdn --mdn_k 5 --chunk 4` |
| 4 | expert learnability vs optimality | `quality_sweep.py` |
| 5 | temporal / predictive observation features | `--obs_extra` |

## 0. Setup (once)
```bash
# on the instance (any CUDA image with python>=3.10)
pip install torch numpy scipy matplotlib pytest
# copy the v2 folder up:  scp -P <port> -r offline_marl_exp/v2 root@<ip>:~/v2
cd ~/v2
nvidia-smi | head -15            # confirm a GPU; the code also runs on CPU (slow)
```

## 1. Tests (must pass before anything else)
```bash
python -m pytest tests -x -q     # old gates (test_gates.py) + new test_bc2.py
```
If `test_bc2.py` fails, stop and send me the traceback — that is a bug in new code.

## 2. Smoke run (seconds–minutes; numbers meaningless)
```bash
python student.py --name smoke --smoke --head mdn --chunk 4 --obs_extra --rounds 1 --out results_smoke
python quality_sweep.py --variants base weak --n_train 20 --n_val 8 --epochs 1 --n_monitor 8 --out results_smoke/qs
```

## 3. Ablation matrix, cumulative (3 student seeds first; 5 for the final table)
Same data (`data/expert_train.pt`, 600 scenarios), same seeds, k = 6. `A0` is the control: it isolates *budget only*.
```bash
for s in 0 1 2; do
python student.py --name A0 --seed $s --rounds 0 --epochs0 300 --d 128                                   # budget/capacity/EMA
python student.py --name A1 --seed $s --rounds 0 --epochs0 300 --d 128 --head mdn --chunk 4              # + MDN & chunk
python student.py --name A2 --seed $s --rounds 0 --epochs0 300 --d 128 --head mdn --chunk 4 --obs_extra # + temporal features
python student.py --name A3 --seed $s --rounds 5 --scen 256 --epochs0 300 --d 128 --head mdn --chunk 4 --obs_extra   # + DAgger
done
python summarize.py --names A0 A1 A2 A3 --baseline A0
```
Single-factor checks (to attribute gains): `--name B_dagger_only --rounds 5` (no mdn/chunk/extra),
`--name B_mdn_only --head mdn --chunk 4`, `--name B_extra_only --obs_extra` (all with `--rounds 0` unless stated).

Diagnostic for item 2: in `results2/A0/seed0/log.txt` compare `EMA train mse0` and `val mse0`. Train ≈ val and both still
falling ⇒ underfit (more budget helps). Train ≪ val ⇒ data-limited (DAgger / more scenarios help, not epochs).

DAgger cost knobs: `--scen`, `--plan_restarts 4 --plan_rounds 4 --plan_iters 80` (the planner is called at every one of the
20 steps of every round). Cheaper labels: `--plan_restarts 2 --plan_rounds 2 --plan_iters 40`.

## 4. Expert quality sweep (item 4)
```bash
for s in 0 1 2; do python quality_sweep.py --seed $s --epochs 150 --d 128 --head mdn --chunk 4; done
```
Read: does `smooth1/smooth2` (planner pays a higher energy price, smoother trajectories) give a student with a higher
normalised score than `base`, even though its expert return at the true price is lower? Do `weak/weakest` experts give a
*worse* student, and by how much? (No prediction; this is the measurement.)

## 5. Final, held-out evaluation (do this ONCE per config you will report)
`--final` evaluates the monitor-selected best round on fresh seeds 800000..800255:
```bash
for s in 0 1 2 3 4; do python student.py --name A4 --seed $s --rounds 5 --epochs0 300 --d 128 --head mdn --chunk 4 --obs_extra --final; done
python summarize.py
```
Do not tune on the final seeds; choose the config from the monitor numbers (section 3).

## 6. Online phase with the new student (masked vs unmasked critic)
```bash
python online2.py --ckpt "results2/A4/seed{s}/best.pt" --seeds 0 1 2 3 4 --iters 150 --tag A4
# at the very end, once:  ... --tag A4 --final
```
Reuses `mappo.py` unchanged (the MDN student acts as a Gaussian around its mode online). The masked-vs-unmasked contrast is
defined as in v2: the arms differ only in the critic's attention mask.

## Outputs
`results2/<name>/seed<s>/{log.txt,summary.json,round*.pt,best.pt}`, `results2/quality_sweep/sweep_seed<s>.json`,
`results2/online/<tag>/…`. Copy `results2/` back and send it to me; `summarize.py` output is the table I will read.

## Caveats
* Normalised score = (student − stay-put)/(expert − stay-put) with the *planner as expert*; the planner is not a certified optimum.
* The monitor seeds (700000..) are used for round selection, so monitor numbers are optimistic; `--final` numbers are not.
* `--obs_extra` changes the observation size, so its checkpoints are not interchangeable with the old `bc_seed*.pt` actors.
* `--plan_energy` in `student.py` only changes DAgger labels; use `quality_sweep.py` for the energy-price experiment.

## 7. Split radius and global broadcast (v4 additions)
New flags (defaults reproduce everything above): `--k_a R` gives the other-UAV set its own radius (`--k` then governs users only);
`--bcast 1` adds the per-channel count of ALL other UAVs; `--bcast 2` also adds the share of ALL users in 4 distance rings
(<5, 5-10, 10-15, >=15). `bash scripts/run_v4.sh` launches B_uinf, B_ainf, C_b1, C_b2 (3 seeds each, A7 recipe);
compare with `python summarize.py --names A7 A8_kinf B_uinf B_ainf C_b1 C_b2 --baseline A7`.
Reading: B_uinf vs A7 = value of seeing distant USERS; B_ainf vs A7 = value of seeing distant UAVs; C_b1/C_b2 vs A7 = how much of
the gap to A8_kinf a fixed-size broadcast recovers. Tests: `python -m pytest tests -x -q` (now 20 tests).
