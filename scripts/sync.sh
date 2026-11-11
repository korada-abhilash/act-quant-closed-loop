#!/usr/bin/env bash
# Copy the repo to the GPU machine (code only; outputs stay remote). Reads cluster.local.env.
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster.local.env
rsync -az --exclude .git --exclude outputs --exclude logs ./ "$REMOTE_HOST:$REMOTE_DIR/"
