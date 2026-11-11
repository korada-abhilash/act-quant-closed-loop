#!/usr/bin/env bash
#SBATCH --job-name=act-eval-ckpt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.log
# Evaluate a trained checkpoint in the ALOHA sim with the same setup as eval_pretrained.sbatch.
# Usage: CKPT=outputs/train/<run>/checkpoints/<step>/pretrained_model scripts/submit.sh jobs/eval_ckpt.sbatch
set -euo pipefail
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate act-quant
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
echo "CKPT=$CKPT SEED=${SEED:-1000}"

lerobot-eval \
  --policy.path="$CKPT" \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=${N_EPISODES:-50} --eval.batch_size=10 \
  --eval.use_async_envs=false \
  --policy.device=cuda --seed=${SEED:-1000} \
  --output_dir=outputs/eval_${RUN_NAME:-ckpt}_seed${SEED:-1000}_${SLURM_JOB_ID}
