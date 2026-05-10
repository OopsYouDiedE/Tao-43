from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import Dataset

from core_models import ActionConditionedPredictor


class HDF5EpisodeDataset(Dataset):
    def __init__(self, path: str | Path, sequence_len: int) -> None:
        self.path = Path(path)
        self.sequence_len = sequence_len
        with h5py.File(self.path, "r") as h5:
            self.length = int(h5["frames"].shape[0])
            if h5["frames"].shape[1:] != (384, 384, 3):
                raise ValueError("frames must be [Episode_Len, 384, 384, 3].")
            if h5["actions"].shape != (self.length - 1, 7):
                raise ValueError("actions must be [Episode_Len - 1, 7].")
            if h5["states"].shape != (self.length - 1, 7):
                raise ValueError("states must be [Episode_Len - 1, 7].")
        if self.length < sequence_len + 1:
            raise ValueError("Episode is shorter than requested sequence length.")

    def __len__(self) -> int:
        return self.length - self.sequence_len

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        with h5py.File(self.path, "r") as h5:
            frames = h5["frames"][index : index + self.sequence_len + 1]
            actions = h5["actions"][index : index + self.sequence_len]
            states = h5["states"][index : index + self.sequence_len]
        frames = torch.from_numpy(frames.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        return {
            "frames": frames,
            "actions": torch.from_numpy(actions.astype(np.float32)),
            "states": torch.from_numpy(states.astype(np.float32)),
        }


@dataclass(frozen=True)
class TrainerConfig:
    rollout_steps: int = 2
    ema_tau: float = 0.996
    mask_ratio: float = 0.75
    unfreeze_last_blocks: int = 2


def compute_ac_predictor_loss(
    encoder,
    predictor: ActionConditionedPredictor,
    frames: Tensor,
    actions: Tensor,
    states: Tensor,
    rollout_steps: int = 2,
) -> tuple[Tensor, dict[str, float]]:
    if frames.ndim != 5:
        raise ValueError(f"frames must be [B, T+1, 3, 384, 384], got {tuple(frames.shape)}")
    with torch.no_grad():
        z = encoder(frames)
    z_in = z[:, :-1]
    z_target = z[:, 1:]
    pred_tf = predictor(z_in, actions, states)
    loss_tf = F.smooth_l1_loss(pred_tf, z_target)

    rollout_losses = []
    z_window = z[:, :1]
    action_window = actions[:, :1]
    state_window = states[:, :1]
    steps = min(rollout_steps, actions.shape[1])
    for step in range(steps):
        pred = predictor(z_window, action_window, state_window)
        next_z = pred[:, -1]
        rollout_losses.append(F.smooth_l1_loss(next_z, z[:, step + 1]))
        z_window = torch.cat([z_window, next_z[:, None]], dim=1)
        if step + 1 < steps:
            action_window = torch.cat([action_window, actions[:, step + 1 : step + 2]], dim=1)
            state_window = torch.cat([state_window, states[:, step + 1 : step + 2]], dim=1)
    loss_roll = torch.stack(rollout_losses).mean() if rollout_losses else torch.zeros_like(loss_tf)
    total = loss_tf + 0.5 * loss_roll
    return total, {
        "loss_tf": float(loss_tf.detach().cpu()),
        "loss_roll": float(loss_roll.detach().cpu()),
        "loss_total": float(total.detach().cpu()),
    }


class EMAEncoder(nn.Module):
    def __init__(self, online_encoder: nn.Module, tau: float = 0.996) -> None:
        super().__init__()
        self.target = copy.deepcopy(online_encoder).eval()
        self.tau = tau
        for param in self.target.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_encoder: nn.Module) -> None:
        for target_param, online_param in zip(self.target.parameters(), online_encoder.parameters()):
            target_param.data.mul_(self.tau).add_(online_param.data, alpha=1.0 - self.tau)


def apply_block_mask(frames: Tensor, mask_ratio: float = 0.75, patch_size: int = 16) -> tuple[Tensor, Tensor]:
    if frames.ndim != 5:
        raise ValueError("frames must be [B, T, 3, 384, 384].")
    batch, time, channels, height, width = frames.shape
    grid_h = height // patch_size
    grid_w = width // patch_size
    mask = torch.rand(batch, time, grid_h, grid_w, device=frames.device) < mask_ratio
    expanded = mask.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)
    expanded = expanded[:, :, None, :, :].expand(batch, time, channels, height, width)
    masked = frames.clone()
    masked[expanded] = 0.0
    return masked, mask


def set_trainable_last_blocks(encoder: nn.Module, n_blocks: int) -> None:
    if not 2 <= n_blocks <= 8:
        raise ValueError("V-JEPA fine-tuning only allows unfreezing the last 2 to 8 blocks.")
    for param in encoder.parameters():
        param.requires_grad_(False)

    blocks = None
    for attr in ("blocks", "encoder"):
        candidate = getattr(encoder, attr, None)
        if isinstance(candidate, nn.ModuleList):
            blocks = candidate
            break
    if blocks is None and hasattr(encoder, "module"):
        return set_trainable_last_blocks(encoder.module, n_blocks)
    if blocks is None:
        raise AttributeError("Could not find transformer blocks on the official V-JEPA encoder.")

    for block in blocks[-n_blocks:]:
        for param in block.parameters():
            param.requires_grad_(True)


def fine_tune_encoder_smooth_l1(
    online_encoder,
    ema_encoder: EMAEncoder,
    frames: Tensor,
    mask_ratio: float = 0.75,
) -> Tensor:
    masked, mask = apply_block_mask(frames, mask_ratio=mask_ratio)
    online_z = online_encoder(masked)
    with torch.no_grad():
        target_z = ema_encoder.target(frames)
    patch_mask = mask.reshape(mask.shape[0], mask.shape[1], -1).to(online_z.device)
    patch_mask = patch_mask[:, :, : online_z.shape[2]]
    selected_online = online_z[patch_mask]
    selected_target = target_z[patch_mask]
    if selected_online.numel() == 0:
        return F.smooth_l1_loss(online_z, target_z)
    return F.smooth_l1_loss(selected_online, selected_target)
