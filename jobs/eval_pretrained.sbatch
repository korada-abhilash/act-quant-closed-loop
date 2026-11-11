#!/usr/bin/env bash
#SBATCH --job-name=act-eval-pretrained
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.log
# Setup check: evaluate LeRobot's published ACT transfer-cube checkpoint in the ALOHA sim.
set -euo pipefail
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate act-quant
# conda-forge FFmpeg needs the env's newer libstdc++; on the GPU nodes the system one wins otherwise.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
# LeRobot 0.6.1 starts async env workers with "forkserver", and those workers never import
# gym_aloha, so the env id is unregistered there. Synchronous envs avoid that path.
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# The hub checkpoint predates LeRobot's processor pipelines; convert it once.
CKPT=outputs/pretrained_act_migrated
if [ ! -f "$CKPT/config.json" ]; then
  python -m lerobot.processor.migrate_policy_normalization \
    --pretrained-path lerobot/act_aloha_sim_transfer_cube_human --output-dir "$CKPT"
fi

lerobot-eval \
  --policy.path="$CKPT" \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=${N_EPISODES:-50} --eval.batch_size=10 \
  --eval.use_async_envs=false \
  --policy.device=cuda --seed=1000 \
  --output_dir=outputs/eval_pretrained_${SLURM_JOB_ID}
