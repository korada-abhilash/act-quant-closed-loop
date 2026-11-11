"""Shared pieces: the exported network boundary, policy/processor loading and parity data."""

import numpy as np
import torch
from torch import nn

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

DATASET = "lerobot/aloha_sim_transfer_cube_human"
IMAGE_KEY = "observation.images.top"


class ACTInference(nn.Module):
    """ACT's inference path: normalized (state, image) -> normalized action chunk.

    Matches `ACTPolicy.predict_action_chunk` in eval mode, where the VAE encoder is skipped and the
    latent is zero. Normalization and the action queue stay in LeRobot's Python code.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, state: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return self.model({OBS_STATE: state, OBS_IMAGES: [image]})[0]


def load_policy(ckpt: str):
    policy = ACTPolicy.from_pretrained(ckpt).to("cpu").eval()
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=ckpt, preprocessor_overrides={"device_processor": {"device": "cpu"}}
    )
    return policy, pre, post


def frame_indices(n_frames: int, n: int, offset: int) -> list[int]:
    """`n` frames spread evenly over the dataset; different `offset`s give disjoint sets."""
    step = n_frames // n
    return [i * step + offset for i in range(n)]


def load_normalized_frames(pre, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Dataset frames run through the policy's own preprocessor, as in closed-loop eval."""
    ds = LeRobotDataset(DATASET)
    states, images = [], []
    for i in indices:
        item = ds[i]
        obs = pre({OBS_STATE: item[OBS_STATE].unsqueeze(0), IMAGE_KEY: item[IMAGE_KEY].unsqueeze(0)})
        states.append(obs[OBS_STATE].numpy())
        images.append(obs[IMAGE_KEY].numpy())
    return np.concatenate(states).astype(np.float32), np.concatenate(images).astype(np.float32)
