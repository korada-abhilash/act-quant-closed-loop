"""INT8 static quantization (QDQ) of the exported ACT network with ONNX Runtime.

Usage: python -m act_onnx.quantize --ckpt <pretrained_model dir> --fp32 outputs/onnx/act_fp32.onnx \
           --out outputs/onnx/act_int8_minmax.onnx --method minmax [--n-calib 200] [--exclude n1,n2]
Calibration frames come from the dataset, disjoint from the frames used by parity.py.
"""

import argparse
from pathlib import Path

import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from act_onnx.common import frame_indices, load_normalized_frames, load_policy

N_FRAMES = 20000
CALIB_OFFSET = 17  # parity uses offset 0 with a step of 40, so these never overlap
# Compute ops get INT8; Relu/MaxPool ride along so Conv-Relu-MaxPool stays in INT8.
OPS = ["Conv", "Gemm", "MatMul", "Relu", "MaxPool"]
METHODS = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile}


class FrameReader(CalibrationDataReader):
    def __init__(self, states, images):
        self.items = iter([{"state": states[i : i + 1], "image": images[i : i + 1]} for i in range(len(states))])

    def get_next(self):
        return next(self.items, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=METHODS, default="minmax")
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--percentile", type=float, default=99.999)
    ap.add_argument("--exclude", default="", help="comma-separated node names to keep in FP32")
    args = ap.parse_args()

    _, pre, _ = load_policy(args.ckpt)
    states, images = load_normalized_frames(pre, frame_indices(N_FRAMES, args.n_calib, CALIB_OFFSET))

    prep = Path(args.out).with_suffix(".prep.onnx")  # per output, so parallel jobs don't collide
    quant_pre_process(args.fp32, str(prep))

    quantize_static(
        str(prep),
        args.out,
        FrameReader(states, images),
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=OPS,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=METHODS[args.method],
        nodes_to_exclude=[n for n in args.exclude.split(",") if n],
        # No CalibMaxIntermediateOutputs: in onnxruntime 1.31 MinMax discards each flushed chunk without
        # computing its range, so calibration ends with "No data is collected".
        extra_options={"CalibPercentile": args.percentile},
    )
    model = onnx.load(args.out)
    n_q = sum(n.op_type == "QuantizeLinear" for n in model.graph.node)
    print(f"wrote {args.out}: {Path(args.out).stat().st_size / 1e6:.1f} MB, {n_q} QuantizeLinear nodes")


if __name__ == "__main__":
    main()
