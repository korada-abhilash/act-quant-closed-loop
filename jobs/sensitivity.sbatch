#!/usr/bin/env bash
#SBATCH --job-name=act-sensitivity
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.log
# Per-layer-group INT8 sensitivity + mixed-precision models, then queue closed-loop evals for the
# mixed models (same partition as this job).
# Usage: CKPT=... [METHOD=percentile] scripts/submit.sh jobs/sensitivity.sbatch
set -euo pipefail
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate act-quant
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

command -v sbatch >/dev/null || { echo "sbatch not on PATH"; exit 1; }
python -m act_onnx.sensitivity --ckpt "$CKPT" --fp32 outputs/onnx/act_fp32.onnx --method ${METHOD:-percentile}

for k in 1 2 3 5 8; do
  RUN_NAME=onnx_mixed_top$k ACT_ONNX=outputs/onnx/act_mixed_top$k.onnx \
    sbatch --export=ALL -p "$SLURM_JOB_PARTITION" jobs/eval_closed_loop.sbatch
done
