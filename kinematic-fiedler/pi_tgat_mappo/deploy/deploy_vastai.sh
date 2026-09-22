#!/usr/bin/env bash
# Sync pi_tgat_mappo/ to a Vast.ai instance, install deps, and run a smoke test.
#
# Usage:
#   ./deploy_vastai.sh <ssh_host> <ssh_port> [remote_workdir]
#
# Example (from the "SSH" button on your Vast.ai instance page, which gives
# you something like: ssh -p 50983 root@154.64.230.50):
#   ./deploy_vastai.sh 154.64.230.50 50983
#
# Run this from YOUR OWN Mac terminal (not through Claude) -- it needs your
# normal internet access and your existing SSH key (already registered with
# your Vast.ai account at https://cloud.vast.ai/manage-keys/).
set -euo pipefail

HOST="${1:?Usage: $0 <ssh_host> <ssh_port> [remote_workdir]}"
PORT="${2:?Usage: $0 <ssh_host> <ssh_port> [remote_workdir]}"
REMOTE_WORKDIR="${3:-/workspace}"
REMOTE_PKG_DIR="$REMOTE_WORKDIR/pi_tgat_mappo"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"   # .../kinematic-fiedler/pi_tgat_mappo

SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new root@$HOST"
RSYNC_SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"

echo "==> Checking remote GPU..."
$SSH "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"

echo "==> Creating $REMOTE_PKG_DIR ..."
$SSH "mkdir -p '$REMOTE_PKG_DIR'"

echo "==> Syncing code (excluding results/ and __pycache__/) ..."
rsync -avz --delete \
  --exclude 'results/' --exclude '__pycache__/' --exclude '*.pyc' \
  -e "$RSYNC_SSH" \
  "$LOCAL_PKG_DIR"/ "root@$HOST:$REMOTE_PKG_DIR"/

echo "==> Confirming remote python3 (this must be the SAME interpreter your"
echo "    interactive SSH session's nohup command will use -- mismatched"
echo "    envs, e.g. a conda env only active in interactive shells, is the"
echo "    most common cause of \"works here, ModuleNotFoundError there\")..."
$SSH "which python3 && python3 --version"

echo "==> Installing dependencies on remote (idempotent -- pip no-ops on"
echo "    anything already satisfied, so this is safe/cheap to re-run)..."
$SSH "python3 -m pip install --upgrade pip -q && \
      python3 -m pip install torch -q && \
      python3 -m pip install torch_geometric numpy matplotlib scipy -q"

echo "==> Verifying CUDA is visible to torch..."
$SSH "python3 -c \"import torch; print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')\""

echo "==> Running --smoke-test on remote (this exercises the whole pipeline)..."
$SSH "cd '$REMOTE_WORKDIR' && python3 -m pi_tgat_mappo.train --smoke-test"

cat <<EOF

==> Smoke test passed. To launch a real run that survives your SSH session
    disconnecting, SSH in and use nohup or tmux, e.g.:

    ssh -p $PORT root@$HOST
    cd $REMOTE_WORKDIR
    nohup python3 -m pi_tgat_mappo.train \\
        --n-min 20 --n-max 50 --max-steps 1000 --iterations 500 \\
        > train.log 2>&1 &
    disown
    tail -f train.log        # Ctrl-C to stop watching (job keeps running)

    Once it's done (or to check progress), fetch results with:
    ./fetch_results.sh $HOST $PORT $REMOTE_WORKDIR

    IMPORTANT: Vast.ai bills by the hour while the instance is running.
    Destroy/stop the instance from https://cloud.vast.ai/instances/ when
    you're done, even if a training run is still queued -- it will not
    stop itself.
EOF
