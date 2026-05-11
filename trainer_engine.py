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
                raise ValueError("frames 必须是 [Episode_Len, 384, 384, 3]。")
            if h5["actions"].shape != (self.length - 1, 7):
                raise ValueError("actions 必须是 [Episode_Len - 1, 7]。")
            if h5["states"].shape != (self.length - 1, 7):
                raise ValueError("states 必须是 [Episode_Len - 1, 7]。")
        if self.length < sequence_len + 1:
            raise ValueError("Episode 长度小于请求的序列长度。")

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


class HDF5WindowDataset(Dataset):
    def __init__(self, path: str | Path, split: str = "train") -> None:
        self.path = Path(path)
        self.split = split
        with h5py.File(self.path, "r") as h5:
            if split not in h5:
                raise ValueError(f"HDF5 数据集没有名为 {split!r} 的 split。")
            group = h5[split]
            frames = group["frames"]
            actions = group["actions"]
            states = group["states"]
            if frames.ndim != 5 or frames.shape[2:] != (384, 384, 3):
                raise ValueError(f"{split}/frames 必须是 [N, T+1, 384, 384, 3]。")
            if actions.shape[:2] != (frames.shape[0], frames.shape[1] - 1) or actions.shape[2] != 7:
                raise ValueError(f"{split}/actions 必须是 [N, T, 7]。")
            if states.shape != actions.shape:
                raise ValueError(f"{split}/states 必须与 actions 形状一致 [N, T, 7]。")
            self.length = int(frames.shape[0])
            self.sequence_len = int(actions.shape[1])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        with h5py.File(self.path, "r") as h5:
            group = h5[self.split]
            frames = group["frames"][index]
            actions = group["actions"][index]
            states = group["states"][index]
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
        raise ValueError(f"frames 必须是 [B, T+1, 3, 384, 384]，实际得到 {tuple(frames.shape)}")
    with torch.no_grad():
        z = encoder(frames)
    z_in = z[:, :-1]
    z_target = z[:, 1:]
    pred_tf = predictor(z_in, actions, states)
    loss_tf = F.smooth_l1_loss(pred_tf, z_target)

    rollout_losses = []
    z_window = z[:, :1]
    steps = min(rollout_steps, actions.shape[1])
    for step in range(steps):
        pred = predictor(z_window, actions[:, :step + 1], states[:, :step + 1])
        next_z = pred[:, -1]
        rollout_losses.append(F.smooth_l1_loss(next_z, z[:, step + 1]))
        z_window = torch.cat([z_window, next_z[:, None]], dim=1)
    loss_roll = torch.stack(rollout_losses).mean() if rollout_losses else torch.zeros_like(loss_tf)
    total = loss_tf + 0.5 * loss_roll
    return total, {
        "loss_tf": float(loss_tf.detach().cpu()),
        "loss_roll": float(loss_roll.detach().cpu()),
        "loss_total": float(total.detach().cpu()),
    }


def compute_ac_predictor_latent_loss(
    predictor: ActionConditionedPredictor,
    z: Tensor,
    actions: Tensor,
    states: Tensor,
    rollout_steps: int = 2,
    delta_loss_weight: float = 2.0,
    focus_loss_weight: float = 4.0,
    action_contrast_weight: float = 0.2,
) -> tuple[Tensor, dict[str, float]]:
    if z.ndim != 4:
        raise ValueError(f"z 必须是 [B, T+1, 576, 768]，实际得到 {tuple(z.shape)}")
    z_in = z[:, :-1]
    z_target = z[:, 1:]
    pred_tf = predictor(z_in, actions, states)
    weights = _latent_focus_weights(z)
    loss_tf = _weighted_smooth_l1(pred_tf, z_target, weights[:, 1:])
    loss_delta = _weighted_smooth_l1(pred_tf - z_in, z_target - z_in, weights[:, 1:])

    rollout_losses = []
    rollout_delta_losses = []
    z_window = z[:, :1]
    steps = min(rollout_steps, actions.shape[1])
    for step in range(steps):
        pred = predictor(z_window, actions[:, :step + 1], states[:, :step + 1])
        next_z = pred[:, -1]
        rollout_losses.append(_weighted_smooth_l1(next_z[:, None], z[:, step + 1 : step + 2], weights[:, step + 1 : step + 2]))
        rollout_delta_losses.append(
            _weighted_smooth_l1(
                (next_z - z_window[:, -1])[:, None],
                (z[:, step + 1] - z_window[:, -1])[:, None],
                weights[:, step + 1 : step + 2],
            )
        )
        z_window = torch.cat([z_window, next_z[:, None]], dim=1)
    loss_roll = torch.stack(rollout_losses).mean() if rollout_losses else torch.zeros_like(loss_tf)
    loss_delta_roll = torch.stack(rollout_delta_losses).mean() if rollout_delta_losses else torch.zeros_like(loss_tf)
    loss_action_contrast = _action_contrastive_loss(predictor, z_in, actions, states, pred_tf.detach(), weights[:, 1:])
    total = loss_tf + 0.5 * loss_roll + delta_loss_weight * (loss_delta + 0.5 * loss_delta_roll) + action_contrast_weight * loss_action_contrast
    return total, {
        "loss_tf": float(loss_tf.detach().cpu()),
        "loss_roll": float(loss_roll.detach().cpu()),
        "loss_delta": float(loss_delta.detach().cpu()),
        "loss_delta_roll": float(loss_delta_roll.detach().cpu()),
        "loss_action_contrast": float(loss_action_contrast.detach().cpu()),
        "loss_total": float(total.detach().cpu()),
    }


def _latent_focus_weights(z: Tensor) -> Tensor:
    diff = (z[:, 1:] - z[:, :-1]).square().mean(dim=-1)
    first = diff[:, :1]
    scores = torch.cat([first, diff], dim=1)
    scores = scores / (scores.mean(dim=-1, keepdim=True) + 1e-6)
    return (1.0 + 4.0 * scores.clamp(max=5.0)).detach()


def _weighted_smooth_l1(pred: Tensor, target: Tensor, weights: Tensor) -> Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def _action_contrastive_loss(
    predictor: ActionConditionedPredictor,
    z_in: Tensor,
    actions: Tensor,
    states: Tensor,
    positive_pred: Tensor,
    weights: Tensor,
) -> Tensor:
    """如果乱序动作的预测结果与真实动作无法区分，则施加惩罚。

    注意：要求 batch_size >= 2 才能产生非零值。当使用 --batch-size 1
    训练时，该项始终为 0；请使用 --grad-accum 保持有效的
    大批量，并设置 --batch-size >= 2 以激活此损失。
    """
    if actions.shape[0] < 2:
        return torch.zeros((), device=z_in.device, dtype=z_in.dtype)
    # 随机排列（绝不为恒等映射），以避免模型学习到固定的平移模式。
    perm = torch.randperm(actions.shape[0], device=actions.device)
    # 防止极端情况下 randperm 返回恒等映射。
    if torch.equal(perm, torch.arange(actions.shape[0], device=actions.device)):
        perm = torch.roll(perm, shifts=1)
    shuffled_actions = actions[perm]
    if torch.allclose(shuffled_actions, actions):
        return torch.zeros((), device=z_in.device, dtype=z_in.dtype)
    negative_pred = predictor(z_in, shuffled_actions, states)
    separation = (
        (negative_pred - positive_pred).square().mean(dim=-1) * weights
    ).sum() / weights.sum().clamp_min(1.0)
    positive_motion = (
        (positive_pred - z_in).square().mean(dim=-1) * weights
    ).sum() / weights.sum().clamp_min(1.0)
    # Hinge 损失：仅在 separation < margin * positive_motion 时施加惩罚。
    margin = 0.05
    return torch.nn.functional.relu(margin * positive_motion.detach() - separation)


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
        raise ValueError("frames 必须是 [B, T, 3, 384, 384]。")
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
        raise ValueError("V-JEPA 微调仅允许解冻最后 2 到 8 个 block。")
    for param in encoder.parameters():
        param.requires_grad_(False)

    if hasattr(encoder, "module"):
        return set_trainable_last_blocks(encoder.module, n_blocks)

    blocks = getattr(encoder, "blocks", None)
    if blocks is None and hasattr(encoder, "encoder"):
        blocks = getattr(encoder.encoder, "blocks", None)
        
    if not isinstance(blocks, nn.ModuleList):
        raise AttributeError("在官方 V-JEPA 编码器上找不到 transformer block。")

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
