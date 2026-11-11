"""Export ACT's inference network to ONNX (FP32).

Usage: python -m act_onnx.export --ckpt <pretrained_model dir> --out outputs/onnx/act_fp32.onnx
"""

import argparse
from pathlib import Path

import onnx
import torch

from act_onnx.common import ACTInference, load_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    policy, _, _ = load_policy(args.ckpt)
    net = ACTInference(policy.model).eval()
    state = torch.zeros(1, 14)
    image = torch.zeros(1, 3, 480, 640)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # TorchScript-based exporter: plain ONNX ops that ONNX Runtime's quantizer handles directly.
    torch.onnx.export(
        net,
        (state, image),
        args.out,
        dynamo=False,
        opset_version=args.opset,
        input_names=["state", "image"],
        output_names=["actions"],
        dynamic_axes={"state": {0: "batch"}, "image": {0: "batch"}, "actions": {0: "batch"}},
    )
    model = onnx.load(args.out)
    onnx.checker.check_model(model)
    ops = sorted({n.op_type for n in model.graph.node})
    print(f"wrote {args.out}: {len(model.graph.node)} nodes, {Path(args.out).stat().st_size / 1e6:.1f} MB")
    print("op types:", ", ".join(ops))


if __name__ == "__main__":
    main()
