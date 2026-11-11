#!/usr/bin/env bash
# Sync, then submit a job script on the GPU machine.
# Usage: [STEPS=.. RUN_NAME=..] scripts/submit.sh jobs/<job>.sbatch [extra sbatch args]
# Job settings (STEPS, RUN_NAME, EVAL_FREQ, SAVE_FREQ, LOG_FREQ, N_EPISODES, CKPT, SEED, METHOD, ACT_ONNX, ACT_NOISE_STD) set locally are forwarded.
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster.local.env
scripts/sync.sh
JOB=$1; shift
FORWARD=""
for v in STEPS RUN_NAME EVAL_FREQ SAVE_FREQ LOG_FREQ N_EPISODES CKPT SEED METHOD ACT_ONNX ACT_NOISE_STD; do
  [ -n "${!v:-}" ] && FORWARD+="$v=${!v} "
done
REMOTE_CMD="$SLURM_SETUP && cd $REMOTE_DIR && mkdir -p logs && $FORWARD CONDA_ROOT=$CONDA_ROOT sbatch --export=ALL -p $SLURM_PARTITION $* $JOB"
ssh "$REMOTE_HOST" "bash -lc $(printf %q "$REMOTE_CMD")"
