#!/usr/bin/env bash
#SBATCH --job-name=act-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --output=logs/%x-%j.log
# Train ACT from scratch on the ALOHA sim transfer-cube human demos.
# STEPS / RUN_NAME / EVAL_FREQ / SAVE_FREQ can be overridden at submit time.
set -euo pipefail
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate act-quant
# conda-forge FFmpeg needs the env's newer libstdc++; on the GPU nodes the system one wins otherwise.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
# LeRobot 0.6.1 starts async env workers with "forkserver", and those workers never import
# gym_aloha, so periodic eval uses synchronous envs (see --eval.use_async_envs below).
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --steps=${STEPS:-100000} --batch_size=8 \
  --env_eval_freq=${EVAL_FREQ:-20000} --eval.n_episodes=10 --eval.batch_size=10 --eval.use_async_envs=false \
  --save_freq=${SAVE_FREQ:-20000} --log_freq=${LOG_FREQ:-200} \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false \
  --num_workers=6 --seed=1000 \
  --output_dir=outputs/train/${RUN_NAME:-act_transfer_cube_${SLURM_JOB_ID}}
