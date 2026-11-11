"""CPU latency of one ACT inference (one 100-step action chunk, batch 1) with ONNX Runtime.

Needs only onnxruntime + numpy. Models are timed in interleaved rounds so thermal or power-state
drift on a laptop affects every model equally.
Usage: python act_onnx/bench_latency.py --threads 4 --out outputs/latency_4t.json outputs/onnx/act_fp32.onnx ...
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--per-round", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    feeds = {
        "state": rng.standard_normal((1, 14), dtype=np.float32),
        "image": rng.standard_normal((1, 3, 480, 640), dtype=np.float32),
    }
    sessions = {}
    for m in args.models:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = args.threads
        opts.inter_op_num_threads = 1
        sessions[m] = ort.InferenceSession(m, opts, providers=["CPUExecutionProvider"])
        for _ in range(args.warmup):
            sessions[m].run(None, feeds)

    times = {m: [] for m in args.models}
    for r in range(args.rounds):
        for m, sess in sessions.items():
            for _ in range(args.per_round):
                t0 = time.perf_counter()
                sess.run(None, feeds)
                times[m].append((time.perf_counter() - t0) * 1e3)
        print(f"round {r + 1}/{args.rounds} done", flush=True)

    result = {
        "cpu": platform.processor() or platform.machine(),
        "onnxruntime": ort.__version__,
        "threads": args.threads,
        "models": {
            Path(m).stem: {
                "size_mb": Path(m).stat().st_size / 1e6,
                "median_ms": float(np.median(t)),
                "p90_ms": float(np.percentile(t, 90)),
                "n": len(t),
            }
            for m, t in times.items()
        },
    }
    for k, v in result["models"].items():
        print(f"{k:24s} {v['size_mb']:7.1f} MB  median {v['median_ms']:7.1f} ms  p90 {v['p90_ms']:7.1f} ms")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
