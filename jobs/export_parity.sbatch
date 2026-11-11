#!/usr/bin/env bash
#SBATCH --job-name=act-export
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.log
# Export the checkpoint's inference network to ONNX FP32 and check parity against PyTorch (CPU only).
# Usage: CKPT=outputs/train/<run>/checkpoints/<step>/pretrained_model scripts/submit.sh jobs/export_parity.sbatch
set -euo pipefail
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate act-quant
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
mkdir -p outputs/onnx

python -m act_onnx.export --ckpt "$CKPT" --out outputs/onnx/act_fp32.onnx
python -m act_onnx.parity --ckpt "$CKPT" --onnx outputs/onnx/act_fp32.onnx --out outputs/onnx/parity_fp32.json
