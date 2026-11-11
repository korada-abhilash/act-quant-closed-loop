"""Closed-loop sim eval with the ACT network swapped for an ONNX Runtime session.

Runs LeRobot's own `lerobot-eval` unchanged (env, seeds, processors, action queue, metrics); the only
difference is that `policy.model` is replaced, so FP32 / INT8 / PyTorch runs differ only in the network.

Usage: ACT_ONNX=outputs/onnx/act_int8_minmax.onnx python -m act_onnx.eval_closed_loop <lerobot-eval args>
       Without ACT_ONNX it is plain lerobot-eval (the PyTorch reference).
       ACT_NOISE_STD=<std> (no ACT_ONNX) adds Gaussian noise to PyTorch's normalized action chunk: a
       control for how many episode outcomes any perturbation of that size flips.
"""

import os

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

import lerobot.scripts.lerobot_eval as lerobot_eval
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


class NoisyACT(nn.Module):
    """Control: the PyTorch network plus Gaussian noise on its (normalized) action chunk."""

    def __init__(self, model: nn.Module, std: float):
        super().__init__()
        self.model, self.std = model, std
        self.gen = torch.Generator().manual_seed(0)

    def forward(self, batch):
        actions, latent = self.model(batch)
        noise = torch.randn(actions.shape, generator=self.gen) * self.std
        return actions + noise.to(actions.device), latent


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
    noise = float(os.environ.get("ACT_NOISE_STD", "0"))
    if path or noise:
        make_policy = lerobot_eval.make_policy

        def make_onnx_policy(*args, **kwargs):
            policy = make_policy(*args, **kwargs)
            if path:
                policy.model = OnnxACT(path)
                print(f"policy.model replaced with ONNX Runtime session: {path}")
            else:
                policy.model = NoisyACT(policy.model, noise)
                print(f"policy.model wrapped with Gaussian action noise, std={noise}")
            return policy

        lerobot_eval.make_policy = make_onnx_policy
    lerobot_eval.main()


if __name__ == "__main__":
    main()
