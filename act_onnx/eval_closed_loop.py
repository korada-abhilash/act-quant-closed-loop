"""Closed-loop sim eval with the ACT network swapped for an ONNX Runtime session.

Runs LeRobot's own `lerobot-eval` unchanged (env, seeds, processors, action queue, metrics); the only
difference is that `policy.model` is replaced, so FP32 / INT8 / PyTorch runs differ only in the network.

Usage: ACT_ONNX=outputs/onnx/act_int8_minmax.onnx python -m act_onnx.eval_closed_loop <lerobot-eval args>
       Without ACT_ONNX it is plain lerobot-eval (the PyTorch reference).
"""

import os

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

import lerobot.scripts.lerobot_eval as lerobot_eval
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


class OnnxACT(nn.Module):
    def __init__(self, path: str):
        super().__init__()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
        self.sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])

    def forward(self, batch):
        state = batch[OBS_STATE]
        feeds = {
            "state": state.cpu().numpy().astype(np.float32),
            "image": batch[OBS_IMAGES][0].cpu().numpy().astype(np.float32),
        }
        actions = self.sess.run(["actions"], feeds)[0]
        return torch.from_numpy(actions).to(state.device), (None, None)


def main():
    path = os.environ.get("ACT_ONNX")
    if path:
        make_policy = lerobot_eval.make_policy

        def make_onnx_policy(*args, **kwargs):
            policy = make_policy(*args, **kwargs)
            policy.model = OnnxACT(path)
            print(f"policy.model replaced with ONNX Runtime session: {path}")
            return policy

        lerobot_eval.make_policy = make_onnx_policy
    lerobot_eval.main()


if __name__ == "__main__":
    main()
