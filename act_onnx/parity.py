"""Compare an ONNX model's action chunks with the PyTorch policy on held-out dataset frames.

Usage: python -m act_onnx.parity --ckpt <pretrained_model dir> --onnx <model.onnx> [--n 500]
Errors are reported in normalized units (what the network outputs) and in action units
(radians / gripper units, after LeRobot's unnormalizer).
"""

import argparse
import json
import os

import numpy as np
import onnxruntime as ort
import torch

from act_onnx.common import ACTInference, frame_indices, load_normalized_frames, load_policy

N_FRAMES = 20000  # frames in the dataset
PARITY_OFFSET = 0  # calibration uses a different offset, so the sets don't overlap


def stats(ref: np.ndarray, out: np.ndarray) -> dict:
    diff = np.abs(ref - out)
    r, o = ref.reshape(len(ref), -1), out.reshape(len(out), -1)
    cos = (r * o).sum(1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(o, axis=1))
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "cos_min": float(cos.min()),
        "cos_mean": float(cos.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--out", default=None, help="optional JSON file for the results")
    args = ap.parse_args()

    policy, pre, post = load_policy(args.ckpt)
    net = ACTInference(policy.model).eval()
    states, images = load_normalized_frames(pre, frame_indices(N_FRAMES, args.n, PARITY_OFFSET))

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    sess = ort.InferenceSession(args.onnx, opts, providers=["CPUExecutionProvider"])
    ref, out = [], []
    for i in range(0, len(states), args.batch):
        s, im = states[i : i + args.batch], images[i : i + args.batch]
        with torch.no_grad():
            ref.append(net(torch.from_numpy(s), torch.from_numpy(im)).numpy())
        out.append(sess.run(["actions"], {"state": s, "image": im})[0])
    ref, out = np.concatenate(ref), np.concatenate(out)

    def unnorm(a):
        return post(torch.from_numpy(a.reshape(-1, a.shape[-1]))).numpy().reshape(a.shape)

    result = {
        "onnx": args.onnx,
        "n_frames": len(states),
        "normalized": stats(ref, out),
        "action_units": stats(unnorm(ref), unnorm(out)),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
