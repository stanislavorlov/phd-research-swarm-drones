#!/usr/bin/env bash
# Pull pi_tgat_mappo/results/ (plots, checkpoint, history) and train.log back
# from a Vast.ai instance to your local repo.
#
# Usage: ./fetch_results.sh <ssh_host> <ssh_port> [remote_workdir]
set -euo pipefail

HOST="${1:?Usage: $0 <ssh_host> <ssh_port> [remote_workdir]}"
PORT="${2:?Usage: $0 <ssh_host> <ssh_port> [remote_workdir]}"
REMOTE_WORKDIR="${3:-/workspace}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RSYNC_SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"

echo "==> Fetching results/ ..."
rsync -avz -e "$RSYNC_SSH" \
  "root@$HOST:$REMOTE_WORKDIR/pi_tgat_mappo/results/" \
  "$LOCAL_PKG_DIR/results/"

echo "==> Fetching train.log (if present) ..."
rsync -avz -e "$RSYNC_SSH" \
  "root@$HOST:$REMOTE_WORKDIR/train.log" \
  "$LOCAL_PKG_DIR/deploy/train.log" 2>/dev/null || echo "  (no train.log found remotely, skipping)"

echo "==> Done. Results are in $LOCAL_PKG_DIR/results/"
