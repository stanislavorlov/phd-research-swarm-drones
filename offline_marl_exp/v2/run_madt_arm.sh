#!/bin/bash
set -x
M="--train_data data/expert_train_mix.pt --val_data data/expert_val_mix.pt --k 6 --k_a -1 --bcast 6"
for s in 0 1 2; do
python3 run_madt.py --name M_mix $M --d 128 --epochs 25 --targets 100 120 140 160 --seed $s
python3 student.py --name B_mix $M --rounds 0 --head mdn --d 128 --mdn_k 5 --chunk 4 --epochs0 300 --seed $s
done
