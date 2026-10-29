#!/usr/bin/env bash
# Usage: scripts/deploy.sh <instance-ip> [ssh-port=22] [remote-dir=~/v2] [user=root]
# Copies this folder to the Vast.ai instance, installs deps, optionally runs the tests.
#   RUN_TESTS=1 scripts/deploy.sh 1.2.3.4 40022
set -euo pipefail
IP="${1:?usage: deploy.sh <ip> [port] [remote-dir] [user]}"
PORT="${2:-22}"; RDIR="${3:-~/v2}"; USER_="${4:-root}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"
echo ">> sync $HERE -> $USER_@$IP:$RDIR"
$SSH "$USER_@$IP" "mkdir -p $RDIR"
if command -v rsync >/dev/null && $SSH "$USER_@$IP" "command -v rsync" >/dev/null 2>&1; then
  rsync -az -e "$SSH" \
    --exclude '__pycache__' --exclude '.pytest_cache' --exclude '/results*' --exclude '.git' \
    "$HERE/" "$USER_@$IP:$RDIR/"
else
  echo "(rsync unavailable, falling back to tar over ssh)"
  tar -C "$HERE" --exclude=__pycache__ --exclude=.pytest_cache --exclude='./results*' -czf - . \
    | $SSH "$USER_@$IP" "tar -C $RDIR -xzf -"
fi
echo ">> install deps"
$SSH "$USER_@$IP" "python3 -c 'import torch,numpy,scipy' 2>/dev/null || pip install torch numpy scipy matplotlib pytest; pip list 2>/dev/null | grep -qi pytest || pip install pytest matplotlib"
$SSH "$USER_@$IP" "cd $RDIR && python3 -c 'import torch; print(\"torch\", torch.__version__, \"cuda:\", torch.cuda.is_available())'; nvidia-smi -L || true"
if [ "${RUN_TESTS:-0}" = "1" ]; then
  echo ">> pytest"; $SSH "$USER_@$IP" "cd $RDIR && python3 -m pytest tests -x -q"
fi
cat <<MSG
>> done. Log in and launch inside tmux so runs survive disconnects:
   ssh -p $PORT $USER_@$IP
   cd $RDIR && tmux new -s marl     # (apt-get install -y tmux if missing)
   then paste commands from RUNBOOK_v3.md
MSG
