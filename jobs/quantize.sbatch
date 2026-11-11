#!/usr/bin/env bash
#SBATCH --job-name=act-quant
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.log
# INT8-quantize the FP32 ONNX model with one calibration method, then check parity against PyTorch.
# Usage: CKPT=... METHOD=minmax|percentile scripts/submit.sh jobs/quantize.sbatch
set -euo pipefail
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate act-quant
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
M=${METHOD:-minmax}

python -m act_onnx.quantize --ckpt "$CKPT" --fp32 outputs/onnx/act_fp32.onnx \
  --out outputs/onnx/act_int8_$M.onnx --method $M
python -m act_onnx.parity --ckpt "$CKPT" --onnx outputs/onnx/act_int8_$M.onnx --out outputs/onnx/parity_int8_$M.json
