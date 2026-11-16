#!/usr/bin/env bash
# run_long.sh -- 4,500-iteration runs with the bcast = 6 observation, to close the paper-1/paper-2 gap.
# Mirrors paper 1's scratch_4500 / online_A7_long protocol exactly, with paper 2's checkpoints:
#   ft_scratch_4500 : random init (options taken from P_hard), masked critic, 4500 it   -- the open question
#   ft_hard_4500    : P_hard warm start, masked critic, 4500 it                         -- the equal-budget partner
#
#   bash run_long.sh smoke     # ~2 minutes: one tiny run, checks the environment
#   bash run_long.sh           # the six real runs, in the background
#   bash run_long.sh status    # progress
#   bash run_long.sh eval      # after all six finish: frozen test set (n = 1024)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results2/online logs
ONL="--arms masked --lr_actor 3e-4 --critic_warmup 100 --init_std 0.2 --final --out results2/online"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1      # one core per run; six runs need >= 6 physical cores

case "${1:-run}" in
smoke)
  python3 online2.py --ckpt "results2/P_hard/seed{s}/best.pt" --seeds 0 --iters 30 --scratch \
      --arms masked --lr_actor 3e-4 --critic_warmup 15 --init_std 0.2 --out results2/online --tag SMOKE_long
  echo "smoke OK -- now remove it:  rm -r results2/online/SMOKE_long" ;;
run)
  for s in 0 1 2; do
    nohup python3 online2.py --ckpt "results2/P_hard/seed{s}/best.pt" --seeds $s --scratch --iters 4500 $ONL \
        --tag ft_scratch_4500 > logs/ft_scratch_4500_$s.out 2>&1 &
    nohup python3 online2.py --ckpt "results2/P_hard/seed{s}/best.pt" --seeds $s --iters 4500 $ONL \
        --tag ft_hard_4500    > logs/ft_hard_4500_$s.out 2>&1 &
  done
  sleep 5; echo "started $(pgrep -fc 'online2.py') runs"; echo "progress: bash run_long.sh status" ;;
status)
  for t in ft_scratch_4500 ft_hard_4500; do for s in 0 1 2; do
    f=results2/online/$t/log_masked_seed$s.txt
    printf "%-16s s%s  " $t $s; if [ -f "$f" ]; then tail -1 "$f" | cut -c1-110; else echo "(no log yet)"; fi
  done; done
  echo "running: $(pgrep -fc 'online2.py' || true)   finished npz: $(ls results2/online/ft_*_4500/run_masked_seed*.npz 2>/dev/null | wc -l)/6" ;;
eval)
  python3 online_test_eval.py --tags ft_scratch_4500 ft_hard_4500 --seed0 900000 --n 1024 --planner strong++ \
    --ckpt_map "ft_scratch_4500=results2/P_hard/seed{s}/best.pt,scratch" "ft_hard_4500=results2/P_hard/seed{s}/best.pt" \
    --start_names P_hard --out results2/final/online_test_long.json ;;
esac
