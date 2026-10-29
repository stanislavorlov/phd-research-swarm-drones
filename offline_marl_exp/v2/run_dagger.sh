#!/usr/bin/env bash
set -u
run () {   # name  train  val  restarts rounds iters
  for s in 0 1 2 3 4; do
    OMP_NUM_THREADS=1 nohup python3 student.py --name "$1" --seed "$s" \
      --train_data "$2" --val_data "$3" \
      --k 6 --k_a -1 --bcast 6 --head mdn --d 128 --mdn_k 5 --chunk 4 \
      --epochs0 300 --rounds 8 --scen 256 \
      --plan_restarts "$4" --plan_rounds "$5" --plan_iters "$6" \
      > "results2/$1_seed$s.out" 2>&1 &
  done
  wait
}
run QD_weak1  data/expert_train_weak1.pt data/expert_val_weak1.pt 1 1 10
run QD_weak2  data/expert_train_weak2.pt data/expert_val_weak2.pt 1 2 20
run QD_weak3  data/expert_train_weak3.pt data/expert_val_weak3.pt 2 2 40
run QD_strong data/expert_train.pt       data/expert_val.pt       4 4 80
