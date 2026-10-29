#!/usr/bin/env bash
set -u
sweep () {   # $1 = beta value, $2 = tag
  echo "[$(date +%H:%M:%S)] beta=$1" >&2
  for s in 0 1 2 3 4; do
    OMP_NUM_THREADS=1 nohup python3 iql.py --name "Ib$2_weak1" --seed "$s" \
      --train_data data/expert_train_weak1.pt --val_data data/expert_val_weak1.pt \
      --k 6 --k_a -1 --bcast 6 --head mdn --d 128 --mdn_k 5 --chunk 4 \
      --epochs0 300 --critic_epochs 60 --beta "$1" \
      > "results2/Ib$2_weak1_seed$s.out" 2>&1 &
  done
  wait
}
sweep 0    0
sweep 0.25 025
sweep 0.5  05
sweep 1.0  1
