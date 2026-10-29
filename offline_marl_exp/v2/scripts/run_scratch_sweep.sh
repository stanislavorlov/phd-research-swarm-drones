#!/usr/bin/env bash
# Scratch-specific PPO sweep, stage 1 (one factor at a time around the default scratch config + one combined config).
# Default scratch config (already run, 3 seeds, tag scratch_4500): lr_actor 3e-4, init_std 0.2, critic_warmup 100, n_envs 32.
#
#   cd ~/v2 && bash scripts/run_scratch_sweep.sh            # 5 configs x seeds 0 1 = 10 runs, 1350 iterations each, 10 in parallel
#   DRY=1 bash scripts/run_scratch_sweep.sh                 # only print the jobs
#   ITERS=1350 P=10 SEEDS="0 1" bash scripts/run_scratch_sweep.sh
#
# Results: results2/online/<tag>/{log,run}_masked_seed<s>.*   stdout: logs_sweep/<tag>_s<s>.out
set -u
ITERS=${ITERS:-1350}
P=${P:-10}
SEEDS=${SEEDS:-"0 1"}
# tag|lr_actor|init_std|critic_warmup
CONFIGS="sw_lr1e-3|1e-3|0.2|100
sw_std0.1|3e-4|0.1|100
sw_std0.4|3e-4|0.4|100
sw_wu15|3e-4|0.2|15
sw_combo|1e-3|0.3|15"

mkdir -p results2/online logs_sweep
jobs_list() {
  for c in $CONFIGS; do
    IFS='|' read -r tag lr std wu <<< "$c"
    for s in $SEEDS; do echo "$tag $lr $std $wu $s"; done
  done
}
if [ "${DRY:-0}" = "1" ]; then jobs_list; exit 0; fi

jobs_list | xargs -P "$P" -L 1 bash -c \
  'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python online2.py --ckpt "results2/A7/seed{s}/best.pt" --seeds $4 --arms masked --scratch \
     --iters '"$ITERS"' --lr_actor $1 --init_std $2 --critic_warmup $3 --final --tag $0 --out results2/online \
     > logs_sweep/$0_s$4.out 2>&1'
echo "sweep finished"
