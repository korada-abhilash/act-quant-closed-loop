#!/usr/bin/env bash
#SBATCH --job-name=act-closed-loop
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.log
# Closed-loop sim eval of one network variant. The policy runs on CPU for every variant (PyTorch and
# ONNX alike), so numerics are comparable; the GPU is only used for MuJoCo rendering.
# Usage: CKPT=... RUN_NAME=int8_minmax [ACT_ONNX=outputs/onnx/act_int8_minmax.onnx] scripts/submit.sh jobs/eval_closed_loop.sbatch
set -euo pipefail
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate act-quant
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
echo "CKPT=$CKPT ACT_ONNX=${ACT_ONNX:-<pytorch>} ACT_NOISE_STD=${ACT_NOISE_STD:-0} SEED=${SEED:-2000} N_EPISODES=${N_EPISODES:-100}"

python -m act_onnx.eval_closed_loop \
  --policy.path="$CKPT" \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=${N_EPISODES:-100} --eval.batch_size=10 \
  --eval.use_async_envs=false \
  --policy.device=cpu --seed=${SEED:-2000} \
  --output_dir=outputs/closed_loop/${RUN_NAME}_seed${SEED:-2000}_${SLURM_JOB_ID}
