#!/usr/bin/env bash
# Split-radius + broadcast experiments (open-next-steps 1 and 2). Run from the v2 folder on the instance:
#     bash scripts/run_v4.sh            # launches all 12 jobs in the background, 6 at a time (P=6 bash ... to change)
# Re-running is safe: student.py refuses a (name, seed) that is finished or already running (PID lock).
set -u
PY=/venv/main/bin/python; [ -x "$PY" ] || PY=$(command -v python3 || command -v python)
P=${P:-6}
COMMON="--rounds 8 --scen 256 --epochs_round 30 --epochs0 300 --d 128 --head mdn --chunk 4"
mkdir -p logs_v4
JOBS=logs_v4/jobs.txt; : > "$JOBS"
for s in 0 1 2; do
  echo "--name B_uinf --seed $s --k 1000 --k_a 6   $COMMON" >> "$JOBS"   # all users visible, UAVs within 6
  echo "--name B_ainf --seed $s --k 6    --k_a 1000 $COMMON" >> "$JOBS"  # users within 6, all UAVs visible
  echo "--name C_b1   --seed $s --k 6 --bcast 1   $COMMON" >> "$JOBS"     # + UAVs-per-channel broadcast
  echo "--name C_b2   --seed $s --k 6 --bcast 2   $COMMON" >> "$JOBS"     # + user distance-ring broadcast
done
nohup sh -c "cat $JOBS | xargs -P $P -I{} sh -c '$PY student.py {} > /dev/null 2>&1'" > logs_v4/xargs.out 2>&1 &
echo "launched $(wc -l < $JOBS) jobs, $P at a time. Progress:  tail -n 2 results2/*/seed*/log.txt   |  pgrep -af student.py | wc -l"
