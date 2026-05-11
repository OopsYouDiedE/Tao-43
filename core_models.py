from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


PATCHES_PER_FRAME = 576
TOKENS_PER_FRAME = PATCHES_PER_FRAME + 2


@dataclass(frozen=True)
class PredictorConfig:
    visual_dim: int = 1024
    pred_dim: int = 768
    action_dim: int = 7
    state_dim: int = 7
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.0
    max_frames: int = 64  # 可学习帧时间嵌入的上限


class VJEPAEncoder(nn.Module):
    """官方 V-JEPA 2.1 编码器包装器，提供稳定的视频 API。

    公共输入始终为 [B, T, 3, 384, 384]。每一帧都通过显式的单一时间维度
    前向传播到官方模型，匹配 V-JEPA 图像推理路径，并为每帧生成 576 个 ViT-L 补丁 token。
    """

    def __init__(
        self,
        hub_repo: str = "facebookresearch/vjepa2",
        model_name: str = "vjepa2_1_vit_large_384",
        require_official: bool = True,
        pretrained: bool = True,
        checkpoint_base_url: str | None = None,
        patches_per_frame: int = PATCHES_PER_FRAME,
        embed_dim: int = 1024,
        image_size: int = 384,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.hub_repo = hub_repo
        self.model_name = model_name
        self.require_official = require_official
        self.pretrained = pretrained
        self.checkpoint_base_url = checkpoint_base_url
        self.patches_per_frame = patches_per_frame
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.device = torch.device(device or "cpu")
        self.encoder = self._load_official_encoder().to(self.device).eval()

    def _load_official_encoder(self) -> nn.Module:
        if self.model_name == "vjepa2_1_vit_small_384":
            return self._load_official_vjepa21_small()

        try:
            loaded = torch.hub.load(
                self.hub_repo,
                self.model_name,
                pretrained=self.pretrained,
                trust_repo=True,
            )
        except Exception as exc:
            if self.checkpoint_base_url and self._cached_hub_uses_localhost_checkpoint():
                self._patch_cached_hub_checkpoint_base()
                try:
                    loaded = torch.hub.load(
                        self._cached_hub_repo_path(),
                        self.model_name,
                        source="local",
                        pretrained=self.pretrained,
                    )
                    return self._extract_encoder(loaded)
                except Exception as retry_exc:
                    raise RuntimeError(
                        "在重试缓存的官方 hub 仓库后，无法加载官方 V-JEPA 2.1 模型，"
                        f"使用的检查点基础路径为 {self.checkpoint_base_url!r}。"
                    ) from retry_exc
            if self.require_official:
                raise RuntimeError(
                    "无法通过 "
                    f"torch.hub.load({self.hub_repo!r}, {self.model_name!r}) 加载官方 V-JEPA 2.1 模型。"
                    "此验证不允许使用 mock 回退。"
                ) from exc
            raise

        if isinstance(loaded, (tuple, list)):
            encoder = loaded[0]
        else:
            encoder = loaded
        if not isinstance(encoder, nn.Module):
            raise TypeError(f"官方 V-JEPA hub 入口返回了不支持的类型：{type(loaded)!r}")
        return encoder

    def _load_official_vjepa21_small(self) -> nn.Module:
        if self.pretrained:
            raise RuntimeError(
                "vjepa2_1_vit_small_384 是官方 V-JEPA2.1 源码中的架构，"
                "在 hub 中没有公开的预训练检查点。"
            )
        repo_path = Path(self._cached_hub_repo_path())
        if not repo_path.exists():
            torch.hub.load(self.hub_repo, "vjepa2_1_vit_base_384", pretrained=False, trust_repo=True)
        repo_path = Path(self._cached_hub_repo_path())
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        from app.vjepa_2_1.models import vision_transformer as vit_encoder

        encoder = vit_encoder.vit_small(
            patch_size=16,
            img_size=(self.image_size, self.image_size),
            num_frames=64,
            tubelet_size=2,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )
        return encoder

    @staticmethod
    def _extract_encoder(loaded: Any) -> nn.Module:
        if isinstance(loaded, (tuple, list)):
            encoder = loaded[0]
        else:
            encoder = loaded
        if not isinstance(encoder, nn.Module):
            raise TypeError(f"官方 V-JEPA hub 入口返回了不支持的类型：{type(loaded)!r}")
        return encoder

    def _cached_hub_repo_path(self) -> str:
        repo_name = self.hub_repo.replace("/", "_")
        return str(Path(torch.hub.get_dir()) / f"{repo_name}_main")

    def _patch_cached_hub_checkpoint_base(self) -> None:
        repo_path = Path(self._cached_hub_repo_path())
        backbones = repo_path / "src" / "hub" / "backbones.py"
        if not backbones.exists():
            raise FileNotFoundError(f"未找到缓存的官方 hub 源码：{backbones}")
        text = backbones.read_text(encoding="utf-8")
        old = 'VJEPA_BASE_URL = "http://localhost:8300"'
        new = f'VJEPA_BASE_URL = "{self.checkpoint_base_url}"'
        if old in text:
            backbones.write_text(text.replace(old, new), encoding="utf-8")

        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / "vjepa2_1_vitl_dist_vitG_384.pt"
        if checkpoint.exists() and checkpoint.stat().st_size < 1024 * 1024:
            checkpoint.unlink()

    def _cached_hub_uses_localhost_checkpoint(self) -> bool:
        backbones = Path(self._cached_hub_repo_path()) / "src" / "hub" / "backbones.py"
        return backbones.exists() and "localhost:8300" in backbones.read_text(encoding="utf-8")

    @torch.no_grad()
    def forward(self, frames: Tensor) -> Tensor:
        if frames.ndim != 5:
            raise ValueError(f"VJEPAEncoder 期望输入 [B, T, 3, 384, 384]，实际得到 {tuple(frames.shape)}")
        batch, time, channels, height, width = frames.shape
        if channels != 3 or height != self.image_size or width != self.image_size:
            raise ValueError(f"期望 RGB {self.image_size}x{self.image_size} 格式的帧，实际得到 {tuple(frames.shape)}")
        if frames.min().item() < -1e-6 or frames.max().item() > 1.0 + 1e-6:
            raise ValueError("VJEPAEncoder 期望像素值被归一化到 [0.0, 1.0] 范围内。")

        x = frames.to(self.device, non_blocking=True).reshape(batch * time, channels, height, width)
        x = x.unsqueeze(2)  # [B*T, 3, 1, 384, 384]，显式的时间维度。
        z = self.encoder(x)
        z = _unwrap_encoder_output(z)
        if z.ndim != 3:
            raise RuntimeError(
                "官方 V-JEPA 编码器返回 "
                f"{tuple(z.shape)}，期望 [B*T, {self.patches_per_frame}, {self.embed_dim}]。"
            )
        if z.shape[1:] != (self.patches_per_frame, self.embed_dim):
            raise RuntimeError(
                "官方 V-JEPA 编码器输出形状不匹配： "
                f"得到 {tuple(z.shape)}，期望 [B*T, {self.patches_per_frame}, {self.embed_dim}]。"
            )
        return z.reshape(batch, time, self.patches_per_frame, self.embed_dim)


def _unwrap_encoder_output(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    if isinstance(output, dict):
        for key in ("x", "features", "last_hidden_state"):
            value = output.get(key)
            if isinstance(value, Tensor):
                return value
    raise TypeError(f"不支持的 V-JEPA 编码器输出类型：{type(output)!r}")


def _mentions_localhost_checkpoint(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if "localhost:8300" in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


class ActionConditionedPredictor(nn.Module):
    """基于交错的动作、状态和视觉 token 的 Transformer 预测器。"""

    def __init__(self, config: PredictorConfig | None = None) -> None:
        super().__init__()
        self.config = config or PredictorConfig()
        cfg = self.config
        self.visual_proj = nn.Linear(cfg.visual_dim, cfg.pred_dim)
        self.action_proj = nn.Linear(cfg.action_dim, cfg.pred_dim)
        self.state_proj = nn.Linear(cfg.state_dim, cfg.pred_dim)

        # 为视觉 token 提供可学习的时间嵌入（修复：区分
        # 不同时间步下的相同补丁；动作/状态则使用 RoPE）。
        self.frame_temporal_embed = nn.Parameter(
            torch.zeros(cfg.max_frames, cfg.pred_dim)
        )
        nn.init.normal_(self.frame_temporal_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.pred_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.pred_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.output_norm = nn.LayerNorm(cfg.pred_dim)
        self.output_proj = nn.Linear(cfg.pred_dim, cfg.visual_dim)

    def forward(self, visual_tokens: Tensor, actions: Tensor, states: Tensor) -> Tensor:
        seq = self.interleave_tokens(visual_tokens, actions, states)
        mask = self.build_block_causal_mask(
            num_frames=visual_tokens.shape[1],
            device=seq.device,
        )
        encoded = self.blocks(seq, mask=mask)
        encoded = self.output_norm(encoded)
        visual_encoded = self.extract_visual_tokens(encoded, num_frames=visual_tokens.shape[1])
        return self.output_proj(visual_encoded)

    def interleave_tokens(self, visual_tokens: Tensor, actions: Tensor, states: Tensor) -> Tensor:
        self._validate_inputs(visual_tokens, actions, states)
        batch, time, patches, _ = visual_tokens.shape
        if time > self.config.max_frames:
            raise ValueError(f"输入有 {time} 帧，但 max_frames={self.config.max_frames}。")

        visual = self.visual_proj(visual_tokens) + self.frame_temporal_embed[:time].view(1, time, 1, -1)
        action_state = torch.stack([self.action_proj(actions), self.state_proj(states)], dim=2)
        action_state = self._apply_temporal_rope(action_state)
        per_frame = torch.cat([action_state, visual], dim=2)
        return per_frame.reshape(batch, time * (patches + 2), self.config.pred_dim)

    def extract_visual_tokens(self, encoded: Tensor, num_frames: int) -> Tensor:
        batch, seq_len, dim = encoded.shape
        expected = num_frames * TOKENS_PER_FRAME
        if seq_len != expected:
            raise ValueError(f"编码后的序列有 {seq_len} 个 token，期望 {expected} 个。")
        framed = encoded.reshape(batch, num_frames, TOKENS_PER_FRAME, dim)
        return framed[:, :, 2:, :]

    @staticmethod
    def build_block_causal_mask(num_frames: int, device: torch.device | str | None = None) -> Tensor:
        seq_len = num_frames * TOKENS_PER_FRAME
        frame_ids = torch.arange(seq_len, device=device) // TOKENS_PER_FRAME
        return frame_ids.unsqueeze(1) < frame_ids.unsqueeze(0)

    def _apply_temporal_rope(self, tokens: Tensor) -> Tensor:
        batch, time, two_tokens, dim = tokens.shape
        if two_tokens != 2 or dim % 2 != 0:
            raise ValueError("RoPE 期望 [B, T, 2, 偶数维度] 的动作/状态 token。")

        pos = torch.arange(time, device=tokens.device, dtype=tokens.dtype)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=tokens.device, dtype=tokens.dtype) / dim))
        angles = torch.outer(pos, inv_freq)
        cos, sin = angles.cos().view(1, time, 1, -1), angles.sin().view(1, time, 1, -1)
        even, odd = tokens[..., 0::2], tokens[..., 1::2]
        
        rotated = torch.empty_like(tokens)
        rotated[..., 0::2] = even * cos - odd * sin
        rotated[..., 1::2] = even * sin + odd * cos
        return rotated

    def _validate_inputs(self, visual_tokens: Tensor, actions: Tensor, states: Tensor) -> None:
        if visual_tokens.ndim != 4:
            raise ValueError(f"visual_tokens 必须是 [B, T, 576, 1024]，实际得到 {tuple(visual_tokens.shape)}")
        if visual_tokens.shape[2:] != (PATCHES_PER_FRAME, self.config.visual_dim):
            raise ValueError(f"未知的视觉 token 形状：{tuple(visual_tokens.shape)}")
        expected_side = visual_tokens.shape[:2] + (self.config.action_dim,)
        if actions.shape != expected_side:
            raise ValueError(f"actions 必须是 {expected_side}，实际得到 {tuple(actions.shape)}")
        expected_state = visual_tokens.shape[:2] + (self.config.state_dim,)
        if states.shape != expected_state:
            raise ValueError(f"states 必须是 {expected_state}，实际得到 {tuple(states.shape)}")


def mean_patch_cost(predicted: Tensor, goal: Tensor) -> Tensor:
    if predicted.shape[-2:] != goal.shape[-2:]:
        raise ValueError(f"特征形状不同：{tuple(predicted.shape)} 对比 {tuple(goal.shape)}")
    diff = predicted.mean(dim=-2) - goal.mean(dim=-2)
    return diff.square().sum(dim=-1)
