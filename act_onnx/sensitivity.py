"""Per-layer-group INT8 sensitivity and mixed precision.

1. Calibrate once (ranges cached to JSON), then quantize ONE layer group at a time (rest FP32) and
   measure its action error against PyTorch on held-out frames.
2. Rank groups by that error; build mixed-precision models that keep the top-k groups in FP32 and
   quantize everything else; measure them too.

Usage: python -m act_onnx.sensitivity --ckpt <pretrained_model dir> --fp32 outputs/onnx/act_fp32.onnx
Writes outputs/sensitivity/results.json and outputs/onnx/act_mixed_top{k}.onnx.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from act_onnx.common import ACTInference, frame_indices, load_normalized_frames, load_policy
from act_onnx.parity import N_FRAMES, PARITY_OFFSET, stats
from act_onnx.quantize import CALIB_OFFSET, METHODS, OPS, FrameReader

TOP_KS = [1, 2, 3, 5, 8]


def group_of(name: str) -> str:
    """'/model/encoder/layers.0/linear1/MatMul' -> 'encoder.0.ffn'; backbone by stage; projections by name."""
    p = name.strip("/").split("/")[1:]  # drop leading 'model'
    if p[0] == "backbone":
        return f"backbone.{p[1] if p[1].startswith('layer') else 'stem'}"
    if p[0] in ("encoder", "decoder"):
        sub = "ffn" if p[2] in ("linear1", "linear2", "Relu") else p[2]
        return f"{p[0]}.{p[1].split('.')[1]}.{sub}"
    return p[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--method", choices=METHODS, default="percentile")
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--outdir", default="outputs/sensitivity")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    policy, pre, post = load_policy(args.ckpt)
    net = ACTInference(policy.model).eval()
    states, images = load_normalized_frames(pre, frame_indices(N_FRAMES, args.n_eval, PARITY_OFFSET))
    with torch.no_grad():
        ref = np.concatenate(
            [net(torch.from_numpy(states[i : i + 10]), torch.from_numpy(images[i : i + 10])).numpy()
             for i in range(0, len(states), 10)]
        )

    def unnorm(a):
        return post(torch.from_numpy(a.reshape(-1, a.shape[-1]))).numpy().reshape(a.shape)

    ref_u = unnorm(ref)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))

    def evaluate(path) -> dict:
        sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        y = np.concatenate(
            [sess.run(["actions"], {"state": states[i : i + 10], "image": images[i : i + 10]})[0]
             for i in range(0, len(states), 10)]
        )
        return {"normalized": stats(ref, y), "action_units": stats(ref_u, unnorm(y))}

    prep = out / "act_fp32.prep.onnx"
    if not prep.exists():
        quant_pre_process(args.fp32, str(prep))
    groups: dict[str, list[str]] = {}
    for n in onnx.load(str(prep), load_external_data=False).graph.node:
        if n.op_type in OPS:
            groups.setdefault(group_of(n.name), []).append(n.name)

    cache = out / f"calib_{args.method}.json"
    reader = None
    if not cache.exists():
        cs, ci = load_normalized_frames(pre, frame_indices(N_FRAMES, args.n_calib, CALIB_OFFSET))
        reader = FrameReader(cs, ci)

    def quantize(dst, nodes_to_quantize=None, nodes_to_exclude=None):
        quantize_static(
            str(prep), str(dst), reader if not cache.exists() else None,
            quant_format=QuantFormat.QDQ, op_types_to_quantize=OPS, per_channel=True,
            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
            calibrate_method=METHODS[args.method], calibration_cache_path=str(cache),
            nodes_to_quantize=nodes_to_quantize, nodes_to_exclude=nodes_to_exclude,
        )

    results = {"method": args.method, "n_eval": len(states), "groups": {g: len(v) for g, v in groups.items()}}
    tmp = out / "tmp.onnx"

    quantize(tmp)  # full INT8; also fills the calibration cache
    results["full_int8"] = evaluate(tmp)
    print("full_int8", json.dumps(results["full_int8"]["action_units"]), flush=True)

    only = {}
    for g, members in groups.items():
        quantize(tmp, nodes_to_quantize=members)
        only[g] = evaluate(tmp)
        print(f"only {g:40s} mean_abs={only[g]['action_units']['mean_abs']:.5f} "
              f"max_abs={only[g]['action_units']['max_abs']:.4f}", flush=True)
    results["only_group"] = only
    ranking = sorted(only, key=lambda g: only[g]["action_units"]["mean_abs"], reverse=True)
    results["ranking"] = ranking

    mixed = {}
    for k in TOP_KS:
        keep_fp32 = ranking[:k]
        dst = Path("outputs/onnx") / f"act_mixed_top{k}.onnx"
        quantize(dst, nodes_to_exclude=[n for g in keep_fp32 for n in groups[g]])
        mixed[k] = {"fp32_groups": keep_fp32, "size_mb": dst.stat().st_size / 1e6, **evaluate(dst)}
        print(f"mixed top{k} fp32={keep_fp32} mean_abs={mixed[k]['action_units']['mean_abs']:.5f} "
              f"max_abs={mixed[k]['action_units']['max_abs']:.4f}", flush=True)
    results["mixed"] = mixed
    tmp.unlink(missing_ok=True)

    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", out / "results.json")


if __name__ == "__main__":
    main()
