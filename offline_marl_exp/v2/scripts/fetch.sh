#!/usr/bin/env bash
# Usage: scripts/fetch.sh <instance-ip> [ssh-port=22] [remote-dir=~/v2] [user=root]
# Pulls results back into ./results2 (and ./results_smoke if present). Safe to re-run; does not delete anything locally.
#   WITH_CKPT=0 scripts/fetch.sh 1.2.3.4 40022     # skip *.pt checkpoints (logs + json only)
set -euo pipefail
IP="${1:?usage: fetch.sh <ip> [port] [remote-dir] [user]}"
PORT="${2:-22}"; RDIR="${3:-~/v2}"; USER_="${4:-root}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"
EXC=""
[ "${WITH_CKPT:-1}" = "0" ] && EXC="--exclude *.pt"
for d in results2 results_smoke; do
  if $SSH "$USER_@$IP" "test -d $RDIR/$d"; then
    echo ">> fetch $d"
    mkdir -p "$HERE/$d"
    rsync -az --progress2 -e "$SSH" $EXC "$USER_@$IP:$RDIR/$d/" "$HERE/$d/"
  fi
done
echo ">> done. Summarise locally:  cd $HERE && python summarize.py"
