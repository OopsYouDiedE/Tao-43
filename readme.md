# VJEPA-Gym 完整设计方案

> 基于 V-JEPA 2.1 的可回滚物理仿真框架，目标：复现动物认知基准任务

---

## 0. 设计原则与核心约束

**核心假设**
- Encoder（V-JEPA 2.1 ViT-L）训练期间完全冻结，只作为特征提取器
- AC-Predictor 是唯一被持续训练的重型模块
- 物理引擎（MuJoCo）只在验证阶段被触发，规划全程在潜空间进行
- 数据格式采用方案A：`(frame_t, action_t→t+1, state_t)` 预测 `frame_{t+1}`

**版本依赖**
```
mujoco          >= 3.1.0
dm_control      >= 1.0.20
torch           >= 2.3.0
einops          >= 0.7.0
h5py            >= 3.10.0
gymnasium       >= 0.29.0
vjepa2          (torch.hub, facebookresearch/vjepa2)
```

---

## 1. 项目结构

```
vjepa_gym/
│
├── models/                    # 模型层：encoder封装 + 自定义AC-Predictor
│   ├── __init__.py
│   ├── loader.py              # 官方模型加载与版本管理
│   ├── ac_predictor.py        # 自定义AC-Predictor（12层，适配2.1 encoder输出）
│   ├── goal_encoder.py        # 目标图像编码器（复用frozen encoder）
│   └── model_bundle.py        # 统一的模型访问入口
│
├── env/                       # 环境层：dm_control封装
│   ├── __init__.py
│   ├── base_rollback_env.py   # 核心：状态栈 + frame_buffer栈
│   ├── wrappers/
│   │   ├── cartesian_action.py   # Delta Cartesian动作接口
│   │   └── rgb_obs.py            # RGB渲染 → encoder输入格式
│   └── tasks/                    # 场景MJCF + Python任务定义
│       ├── base_task.py
│       ├── nut_crushing.py
│       ├── bottle_cap.py
│       ├── stick_tool.py
│       ├── tube_drop.py
│       ├── multi_tool.py
│       ├── trap_avoidance.py
│       ├── detour.py
│       └── mirror_imitation.py
│
├── planning/                  # 规划层：潜空间rollout + CEM优化
│   ├── __init__.py
│   ├── latent_rollout.py      # AC-Predictor自回归推理
│   ├── cem_planner.py         # Cross-Entropy Method动作序列优化
│   └── energy.py              # 目标能量函数定义
│
├── training/                  # 训练层
│   ├── __init__.py
│   ├── random_explorer.py     # 随机策略轨迹采集
│   ├── ac_trainer.py          # AC-Predictor无监督训练
│   ├── backbone_finetune.py   # V-JEPA 2.1 encoder微调
│   └── curriculum.py          # 课程学习任务调度
│
├── data/                      # 数据层
│   ├── __init__.py
│   ├── trajectory_store.py    # HDF5轨迹存储
│   └── dataset.py             # PyTorch Dataset封装
│
├── configs/                   # 配置层
│   ├── model/
│   │   ├── vitb.yaml
│   │   └── vitl.yaml
│   ├── training/
│   │   ├── ac_train.yaml
│   │   └── finetune.yaml
│   └── tasks/
│       ├── nut_crushing.yaml
│       ├── bottle_cap.yaml
│       ├── stick_tool.yaml
│       ├── tube_drop.yaml
│       ├── multi_tool.yaml
│       ├── trap_avoidance.yaml
│       ├── detour.yaml
│       └── mirror_imitation.yaml
│
└── scripts/                   # 可执行入口
    ├── collect_data.py        # 数据采集
    ├── train_ac.py            # AC训练
    ├── finetune_encoder.py    # Encoder微调
    ├── run_planning.py        # 规划推理
    └── eval_task.py           # 任务评估
```

---

## 2. 模型层（models/）

### 2.1 数据流与张量尺寸约定

```
输入视频片段：  [B, T, C, H, W]  = [B, 8, 3, 224, 224]
                                     ↑帧数上限，推理时可以更少

Encoder输出：   [B, T, N, D]     = [B, 8, 196, 1024]
                                        ↑ 14×14 patches   ↑ ViT-L hidden dim

AC-Predictor输入序列（每帧）：
  action_token:    [B, 1, 768]   ← action_proj(7 → 768)
  state_token:     [B, 1, 768]   ← state_proj(7 → 768)
  visual_tokens:   [B, 196, 768] ← visual_proj(1024 → 768)
  拼接后：         [B, 198, 768] per frame
  全序列：         [B, T×198, 768]

AC-Predictor输出：[B, T, 196, 1024]  ← 还原回encoder embedding dim
目标（教师）：    [B, T, 196, 1024]  ← frozen encoder对下一帧的输出
```

### 2.2 loader.py

```python
"""
models/loader.py
官方V-JEPA 2.1 encoder加载与接口封装。
"""

import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional


class VJEPAEncoder(nn.Module):
    """
    封装官方V-JEPA 2.1 ViT-L encoder。
    对外只暴露 encode() 接口，屏蔽内部细节。
    """

    SUPPORTED_VARIANTS = {
        'vitb': 'vjepa2_vit_base',
        'vitl': 'vjepa2_vit_large',
    }

    def __init__(self, variant: str = 'vitl', device: str = 'cuda'):
        super().__init__()
        assert variant in self.SUPPORTED_VARIANTS, \
            f"variant必须是 {list(self.SUPPORTED_VARIANTS.keys())} 之一"

        hub_name = self.SUPPORTED_VARIANTS[variant]
        # 只加载encoder，不加载官方predictor（我们用自己的AC-Predictor）
        self.backbone = torch.hub.load(
            'facebookresearch/vjepa2',
            hub_name,
            trust_repo=True
        ).to(device)

        # 冻结所有参数，encoder在AC训练阶段不更新
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.embed_dim = self._infer_embed_dim()
        self.patch_size = 16
        self.device = device

    def _infer_embed_dim(self) -> int:
        dims = {'vitb': 768, 'vitl': 1024}
        # 通过实际前向推理确认维度
        dummy = torch.zeros(1, 1, 3, 224, 224, device=self.device)
        with torch.no_grad():
            out = self.backbone(dummy)
        return out.shape[-1]

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: [B, T, C, H, W]，像素值 [0, 1]，已做normalize
        Returns:
            z: [B, T, N_patches, embed_dim]
               N_patches = (H/patch_size) * (W/patch_size) = 14*14 = 196
        """
        B, T, C, H, W = frames.shape
        # encoder逐帧处理，合并batch和time维度
        frames_flat = rearrange(frames, 'b t c h w -> (b t) c h w')
        # 添加fake时间维（官方encoder期望[B, T, C, H, W]，但可单帧运行）
        frames_flat = frames_flat.unsqueeze(1)
        z_flat = self.backbone(frames_flat)         # [(B*T), 1, N, D]
        z = rearrange(z_flat, '(b t) 1 n d -> b t n d', b=B, t=T)
        return z

    def unfreeze_top_k_blocks(self, k: int):
        """微调阶段调用：渐进解冻最后k个transformer block"""
        blocks = list(self.backbone.blocks)
        for block in blocks[-k:]:
            for p in block.parameters():
                p.requires_grad = True
```

### 2.3 ac_predictor.py

```python
"""
models/ac_predictor.py
自定义Action-Conditional Predictor，对齐V-JEPA 2-AC官方架构，
但规模缩小（12层 vs 官方24层）以适配独立训练。

Token Interleaving策略（方案A）：
  每一帧的序列 = [a_t | s_t | z_t^0 ... z_t^{N-1}]
  a_t: 当前帧执行的动作（预测下一帧）
  s_t: 当前帧的末端状态
  z_t: encoder对当前帧的输出
  预测目标: z_{t+1}（下一帧的encoder输出）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class RotaryEmbedding(nn.Module):
    """时间维RoPE，用于action/state token的位置编码"""
    def __init__(self, dim: int, max_seq_len: int = 64):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.unsqueeze(0)  # [1, seq_len, dim]


def build_block_causal_mask(T: int, N_per_frame: int, device: torch.device) -> torch.Tensor:
    """
    构建block-causal attention mask。
    规则：时间步t的token可以attend到时间步≤t的所有token。
    
    每帧的token数量 = 1(action) + 1(state) + N_patches = N_per_frame
    Total tokens = T * N_per_frame
    """
    total = T * N_per_frame
    mask = torch.ones(total, total, device=device, dtype=torch.bool)

    for t in range(T):
        start = t * N_per_frame
        end = (t + 1) * N_per_frame
        # t帧的token不能看到t+1及之后的帧
        mask[start:end, end:] = False

    # 转为additive mask（0 = 可见，-inf = 不可见）
    additive = torch.zeros(total, total, device=device)
    additive[~mask] = float('-inf')
    return additive


class ACBlock(nn.Module):
    """单个Transformer block，支持block-causal attention"""
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ACPredictor(nn.Module):
    """
    Action-Conditional Predictor。

    Architecture:
      - 4个独立线性投影（visual / action / state 各一个）
      - 12层 ACBlock（block-causal attention）
      - 输出投影回encoder embedding dim
    
    参数量估算（dim=768，12层，N=196）：
      投影层：~3M
      Transformer：~85M
      输出层：~0.8M
      总计：~89M
    """

    TOKENS_PER_FRAME = 2  # action_token + state_token（visual_tokens另算）

    def __init__(
        self,
        encoder_dim: int = 1024,    # V-JEPA 2.1 ViT-L输出维度
        predictor_dim: int = 768,   # AC-Predictor内部维度
        n_patches: int = 196,       # 14×14
        action_dim: int = 7,        # Delta Cartesian + gripper
        state_dim: int = 7,         # 末端位姿绝对值
        n_layers: int = 12,
        n_heads: int = 12,
        max_frames: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim
        self.n_patches = n_patches
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.max_frames = max_frames

        # 四路独立线性投影（与官方实现对齐）
        self.visual_proj = nn.Linear(encoder_dim, predictor_dim)
        self.action_proj = nn.Linear(action_dim, predictor_dim)
        self.state_proj = nn.Linear(state_dim, predictor_dim)

        # N_per_frame = TOKENS_PER_FRAME + n_patches
        self.n_per_frame = self.TOKENS_PER_FRAME + n_patches

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ACBlock(predictor_dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(predictor_dim)

        # 输出投影：还原回encoder dim（MSE loss用）
        self.output_proj = nn.Linear(predictor_dim, encoder_dim)

        # 时间RoPE（用于action/state token）
        self.rope = RotaryEmbedding(predictor_dim // n_heads, max_seq_len=max_frames)

        # 预计算causal mask（在forward中按实际T动态截取）
        self._precompute_masks(max_frames)

    def _precompute_masks(self, max_frames: int):
        masks = {}
        for T in range(1, max_frames + 1):
            masks[T] = build_block_causal_mask(T, self.n_per_frame, device='cpu')
        self.register_buffer(
            '_causal_masks_flat',
            torch.stack([masks[T + 1] for T in range(max_frames)], dim=0)
        )  # [max_frames, T*N, T*N]

    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return build_block_causal_mask(T, self.n_per_frame, device)

    def interleave_tokens(
        self,
        z: torch.Tensor,        # [B, T, N, encoder_dim]
        actions: torch.Tensor,  # [B, T, action_dim]
        states: torch.Tensor,   # [B, T, state_dim]
    ) -> torch.Tensor:
        """
        Token Interleaving（方案A）：
        每帧: [a_t(1) | s_t(1) | z_t(N)] → [B, T, N+2, predictor_dim]
        展平: [B, T*(N+2), predictor_dim]
        """
        B, T, N, _ = z.shape

        z_proj = self.visual_proj(z)                    # [B, T, N, predictor_dim]
        a_proj = self.action_proj(actions).unsqueeze(2) # [B, T, 1, predictor_dim]
        s_proj = self.state_proj(states).unsqueeze(2)   # [B, T, 1, predictor_dim]

        # 拼接：action token在前，state token次之，visual tokens在后
        frame_tokens = torch.cat([a_proj, s_proj, z_proj], dim=2)  # [B, T, N+2, D]
        sequence = rearrange(frame_tokens, 'b t n d -> b (t n) d')  # [B, T*(N+2), D]
        return sequence

    def forward(
        self,
        z: torch.Tensor,        # [B, T, N, encoder_dim]  ← 当前帧的encoder输出
        actions: torch.Tensor,  # [B, T, action_dim]       ← 当前帧执行的动作
        states: torch.Tensor,   # [B, T, state_dim]        ← 当前帧的末端状态
    ) -> torch.Tensor:
        """
        Returns:
            z_pred: [B, T, N, encoder_dim]  ← 对下一帧encoder输出的预测
        """
        B, T, N, _ = z.shape

        # Token interleaving
        x = self.interleave_tokens(z, actions, states)  # [B, T*(N+2), predictor_dim]

        # Block-causal attention mask
        attn_mask = self._get_causal_mask(T, z.device)  # [T*(N+2), T*(N+2)]

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask)

        x = self.norm(x)  # [B, T*(N+2), predictor_dim]

        # 只取visual token位置的输出（跳过action/state tokens）
        x = rearrange(x, 'b (t n) d -> b t n d', t=T, n=self.n_per_frame)
        visual_out = x[:, :, self.TOKENS_PER_FRAME:, :]  # [B, T, N, predictor_dim]

        # 输出投影回encoder dim
        z_pred = self.output_proj(visual_out)  # [B, T, N, encoder_dim]
        return z_pred

    def step(
        self,
        z_ctx: torch.Tensor,     # [B, T_ctx, N, encoder_dim] 历史context
        action: torch.Tensor,    # [B, action_dim] 当前动作
        state: torch.Tensor,     # [B, state_dim] 当前状态
    ) -> torch.Tensor:
        """
        单步自回归推理接口（规划时使用）。
        Returns:
            z_next: [B, N, encoder_dim] 预测的下一帧潜向量
        """
        B = z_ctx.shape[0]
        action = action.unsqueeze(1)  # [B, 1, action_dim]
        state = state.unsqueeze(1)    # [B, 1, state_dim]

        # 只用最后一帧的context做预测（单步）
        z_last = z_ctx[:, -1:, :, :]  # [B, 1, N, encoder_dim]
        z_pred = self.forward(z_last, action, state)  # [B, 1, N, encoder_dim]
        return z_pred[:, 0, :, :]  # [B, N, encoder_dim]
```

### 2.4 model_bundle.py

```python
"""
models/model_bundle.py
统一模型访问入口，所有上层模块通过Bundle访问模型。
"""

import torch
from .loader import VJEPAEncoder
from .ac_predictor import ACPredictor
from .goal_encoder import GoalEncoder
from dataclasses import dataclass


@dataclass
class ModelConfig:
    encoder_variant: str = 'vitl'      # 'vitb' | 'vitl'
    predictor_dim: int = 768
    predictor_layers: int = 12
    predictor_heads: int = 12
    action_dim: int = 7
    state_dim: int = 7
    max_frames: int = 16
    device: str = 'cuda'


class ModelBundle:
    """
    框架内所有模块的唯一模型访问点。
    
    Usage:
        bundle = ModelBundle(ModelConfig())
        z = bundle.encode(frames)
        z_pred = bundle.predict(z, actions, states)
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.encoder = VJEPAEncoder(config.encoder_variant, config.device)
        self.predictor = ACPredictor(
            encoder_dim=self.encoder.embed_dim,
            predictor_dim=config.predictor_dim,
            action_dim=config.action_dim,
            state_dim=config.state_dim,
            n_layers=config.predictor_layers,
            n_heads=config.predictor_heads,
            max_frames=config.max_frames,
        ).to(self.device)
        self.goal_encoder = GoalEncoder(self.encoder)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [B, T, C, H, W] → z: [B, T, N, D]"""
        return self.encoder.encode(frames)

    def predict(self, z, actions, states) -> torch.Tensor:
        """z: [B,T,N,D], actions: [B,T,7], states: [B,T,7] → z_pred: [B,T,N,D]"""
        return self.predictor(z, actions, states)

    def encode_goal(self, goal_image: torch.Tensor) -> torch.Tensor:
        """goal_image: [B, C, H, W] → z_goal: [B, N, D]"""
        return self.goal_encoder.encode(goal_image)

    def save_predictor(self, path: str):
        torch.save({
            'config': self.config,
            'predictor': self.predictor.state_dict(),
        }, path)

    @classmethod
    def load_predictor(cls, path: str, config: ModelConfig) -> 'ModelBundle':
        bundle = cls(config)
        ckpt = torch.load(path, map_location=config.device)
        bundle.predictor.load_state_dict(ckpt['predictor'])
        return bundle
```

---

## 3. 环境层（env/）

### 3.1 base_rollback_env.py

```python
"""
env/base_rollback_env.py
核心：在dm_control Physics之上实现状态栈。
所有任务场景继承此类。
"""

import copy
import numpy as np
import torch
from collections import deque
from typing import List, Tuple, Optional, Dict
import dm_control.mujoco as mjc
from dm_control import composer


class RollbackEnv:
    """
    基于dm_control的可回滚环境基类。
    
    关键设计：状态栈同时保存：
      1. MuJoCo物理状态（qpos, qvel, act, ...）
      2. 对应时刻的frame_buffer（保证encoder时序上下文的一致性）
    
    规划时不调用 push/pop，只在latent_rollout.py里用ac_predictor推进。
    push/pop仅用于规划验证阶段的物理确认。
    """

    def __init__(
        self,
        physics: mjc.Physics,
        frame_buffer_len: int = 8,
        render_width: int = 224,
        render_height: int = 224,
        camera_id: int = 0,
    ):
        self.physics = physics
        self.render_width = render_width
        self.render_height = render_height
        self.camera_id = camera_id

        # frame_buffer：保持最近N帧的RGB观测，供encoder使用
        self.frame_buffer: deque = deque(maxlen=frame_buffer_len)
        self.frame_buffer_len = frame_buffer_len

        # 物理状态栈（push/pop用）
        self._phys_stack: List[np.ndarray] = []
        # frame_buffer栈（与物理状态一一对应）
        self._frame_stack: List[List[np.ndarray]] = []

    # ─────────────────────────────────────────────
    # 核心：状态回滚接口
    # ─────────────────────────────────────────────

    def push_state(self) -> int:
        """
        快照当前状态。返回当前栈深度（用于嵌套规划）。
        
        dm_control的physics.get_state()返回展平的[qpos, qvel]拼接向量。
        """
        state = self.physics.get_state().copy()
        self._phys_stack.append(state)
        self._frame_stack.append(list(self.frame_buffer))
        return len(self._phys_stack)

    def pop_state(self) -> None:
        """还原到最近一个快照，并清除该快照。"""
        assert self._phys_stack, "状态栈为空，无法回滚"
        state = self._phys_stack.pop()
        frame_history = self._frame_stack.pop()

        self.physics.set_state(state)
        with self.physics.reset_context():  # 同步mj_step1()派生量
            pass

        self.frame_buffer = deque(frame_history, maxlen=self.frame_buffer_len)

    def clear_stack(self) -> None:
        self._phys_stack.clear()
        self._frame_stack.clear()

    @property
    def stack_depth(self) -> int:
        return len(self._phys_stack)

    # ─────────────────────────────────────────────
    # 标准env接口
    # ─────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """重置场景，清空frame_buffer和状态栈。"""
        self.physics.reset()
        self.frame_buffer.clear()
        self.clear_stack()

        # 用第一帧填满buffer（静止状态重复）
        first_frame = self._render()
        for _ in range(self.frame_buffer_len):
            self.frame_buffer.append(first_frame)

        return first_frame

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作，返回 (rgb_frame, reward, done, info)。
        注意：frame_buffer自动更新。
        """
        self._apply_action(action)
        self.physics.step()

        frame = self._render()
        self.frame_buffer.append(frame)

        reward = self._compute_reward()
        done = self._check_done()
        info = self._get_info()

        return frame, reward, done, info

    def get_frame_tensor(self, device: str = 'cuda') -> torch.Tensor:
        """
        返回当前frame_buffer内容，格式化为encoder输入。
        Returns: [1, T, C, H, W]，像素值归一化到[0,1]
        """
        frames = np.stack(list(self.frame_buffer), axis=0)  # [T, H, W, C]
        frames = frames.transpose(0, 3, 1, 2).astype(np.float32) / 255.0  # [T, C, H, W]
        return torch.from_numpy(frames).unsqueeze(0).to(device)  # [1, T, C, H, W]

    def get_state_vector(self) -> np.ndarray:
        """
        返回末端执行器状态向量 [x, y, z, roll, pitch, yaw, gripper]。
        子类按具体robot MJCF实现。
        """
        raise NotImplementedError

    # ─────────────────────────────────────────────
    # 内部接口（子类实现）
    # ─────────────────────────────────────────────

    def _render(self) -> np.ndarray:
        """渲染当前帧，返回 [H, W, 3] uint8"""
        return self.physics.render(
            height=self.render_height,
            width=self.render_width,
            camera_id=self.camera_id,
        )

    def _apply_action(self, action: np.ndarray) -> None:
        """Delta Cartesian动作 → MuJoCo控制信号（IK）"""
        raise NotImplementedError

    def _compute_reward(self) -> float:
        """任务奖励（训练阶段可能不用）"""
        return 0.0

    def _check_done(self) -> bool:
        return False

    def _get_info(self) -> Dict:
        return {}
```

### 3.2 任务场景定义

所有8个动物认知任务的接口规范与认知能力映射：

```python
"""
env/tasks/base_task.py
任务基类，规定每个场景必须实现的接口。
"""

from dataclasses import dataclass
from enum import IntEnum


class CognitiveLevel(IntEnum):
    """任务对应的认知复杂度层级"""
    SINGLE_CONTACT   = 1   # 单步接触操作
    MULTI_STEP       = 2   # 多步操作序列
    TOOL_AS_MEDIUM   = 3   # 工具作为中介
    HIERARCHICAL     = 4   # 层级子目标规划
    CROSS_MODAL      = 5   # 跨模态迁移


@dataclass
class TaskSpec:
    name: str
    cognitive_level: CognitiveLevel
    animal_prototype: str       # 原型动物
    cognitive_ability: str      # 测试的认知能力
    horizon: int                # 规划步数上限
    success_threshold: float    # 成功判定阈值
    mjcf_path: str


# 全部8个任务的规格定义
TASK_REGISTRY = {
    'nut_crushing': TaskSpec(
        name='坚果压碎',
        cognitive_level=CognitiveLevel.MULTI_STEP,
        animal_prototype='新喀里多尼亚乌鸦',
        cognitive_ability='环境affordance利用 + 空间规划 + 时机判断',
        horizon=30,
        success_threshold=0.8,
        mjcf_path='tasks/mjcf/nut_crushing.xml',
    ),
    'bottle_cap': TaskSpec(
        name='瓶盖拧开',
        cognitive_level=CognitiveLevel.MULTI_STEP,
        animal_prototype='卷尾猴/猩猩',
        cognitive_ability='连续旋转扭矩 + 多步因果链理解',
        horizon=50,
        success_threshold=0.7,
        mjcf_path='tasks/mjcf/bottle_cap.xml',
    ),
    'stick_tool': TaskSpec(
        name='细棍取食',
        cognitive_level=CognitiveLevel.TOOL_AS_MEDIUM,
        animal_prototype='黑猩猩/乌鸦',
        cognitive_ability='工具作为手臂延伸 + 间接操作',
        horizon=40,
        success_threshold=0.7,
        mjcf_path='tasks/mjcf/stick_tool.xml',
    ),
    'tube_drop': TaskSpec(
        name='投石入管',
        cognitive_level=CognitiveLevel.MULTI_STEP,
        animal_prototype='乌鸦/黑猩猩',
        cognitive_ability='非直接物理因果推理',
        horizon=20,
        success_threshold=0.8,
        mjcf_path='tasks/mjcf/tube_drop.xml',
    ),
    'multi_tool': TaskSpec(
        name='多步工具制造',
        cognitive_level=CognitiveLevel.HIERARCHICAL,
        animal_prototype='新喀里多尼亚乌鸦',
        cognitive_ability='元工具使用 + 层级子目标分解',
        horizon=80,
        success_threshold=0.6,
        mjcf_path='tasks/mjcf/multi_tool.xml',
    ),
    'trap_avoidance': TaskSpec(
        name='陷阱规避',
        cognitive_level=CognitiveLevel.TOOL_AS_MEDIUM,
        animal_prototype='倭黑猩猩',
        cognitive_ability='功能性理解 vs 形状匹配 + 反事实推理',
        horizon=25,
        success_threshold=0.75,
        mjcf_path='tasks/mjcf/trap_avoidance.xml',
    ),
    'detour': TaskSpec(
        name='障碍绕行',
        cognitive_level=CognitiveLevel.MULTI_STEP,
        animal_prototype='章鱼/乌鸦',
        cognitive_ability='空间抑制 + detour problem + 路径规划',
        horizon=35,
        success_threshold=0.8,
        mjcf_path='tasks/mjcf/detour.xml',
    ),
    'mirror_imitation': TaskSpec(
        name='镜像模仿',
        cognitive_level=CognitiveLevel.CROSS_MODAL,
        animal_prototype='大象/海豚/猩猩',
        cognitive_ability='视觉→运动映射 + 无监督动作迁移',
        horizon=50,
        success_threshold=0.6,
        mjcf_path='tasks/mjcf/mirror_imitation.xml',
    ),
}
```

---

## 4. 规划层（planning/）

### 4.1 latent_rollout.py

```python
"""
planning/latent_rollout.py
纯潜空间自回归推理。
全程不触碰MuJoCo物理引擎，计算成本极低。
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple
from models.model_bundle import ModelBundle


class LatentRollout:
    """
    用AC-Predictor在潜空间展开动作序列，
    返回预测的潜向量轨迹。
    """

    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle

    @torch.no_grad()
    def rollout(
        self,
        z_init: torch.Tensor,      # [B, T_ctx, N, D] 初始context
        action_seq: torch.Tensor,  # [B, H, action_dim] H步动作序列
        state_seq: torch.Tensor,   # [B, H, state_dim]  H步状态序列
        return_all: bool = True,
    ) -> torch.Tensor:
        """
        自回归rollout H步。
        
        Returns:
            z_traj: [B, H, N, D]  H步的预测潜向量轨迹
        """
        B, H, _ = action_seq.shape
        z_ctx = z_init.clone()
        trajectory = []

        for t in range(H):
            a_t = action_seq[:, t, :]  # [B, action_dim]
            s_t = state_seq[:, t, :]   # [B, state_dim]

            z_next = self.bundle.predictor.step(z_ctx, a_t, s_t)  # [B, N, D]
            trajectory.append(z_next)

            # 更新context window（滑动窗口）
            z_ctx = torch.cat([z_ctx[:, 1:, :, :], z_next.unsqueeze(1)], dim=1)

        return torch.stack(trajectory, dim=1)  # [B, H, N, D]

    @torch.no_grad()
    def compute_goal_distance(
        self,
        z_traj: torch.Tensor,   # [B, H, N, D]
        z_goal: torch.Tensor,   # [B, N, D]
    ) -> torch.Tensor:
        """
        计算轨迹末端与目标的距离（能量函数）。
        使用空间patch均值后的L2距离。
        
        Returns:
            distances: [B, H]  每步与目标的距离
        """
        z_traj_mean = z_traj.mean(dim=2)    # [B, H, D]
        z_goal_mean = z_goal.mean(dim=1)    # [B, D]
        z_goal_exp = z_goal_mean.unsqueeze(1).expand_as(z_traj_mean)
        return F.mse_loss(z_traj_mean, z_goal_exp, reduction='none').mean(-1)  # [B, H]
```

### 4.2 cem_planner.py

```python
"""
planning/cem_planner.py
Cross-Entropy Method动作序列优化器。

规划流程：
  1. 在潜空间用CEM优化H步动作序列（不触碰物理引擎）
  2. 对top-K候选序列，用物理引擎rollback验证
  3. 选出真实物理上可行的最优序列
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from .latent_rollout import LatentRollout
from models.model_bundle import ModelBundle


class CEMPlanner:
    """
    CEM超参建议：
      n_candidates = 512   （latent rollout便宜，采样可以多）
      n_elites = 64        （top 12.5%）
      n_iters = 10         （CEM迭代次数）
      horizon = 15         （规划步数）
      physics_top_k = 5    （物理验证的候选数量，这步昂贵）
    """

    def __init__(
        self,
        bundle: ModelBundle,
        rollout: LatentRollout,
        n_candidates: int = 512,
        n_elites: int = 64,
        n_iters: int = 10,
        horizon: int = 15,
        physics_top_k: int = 5,
        action_dim: int = 7,
        action_low: float = -1.0,
        action_high: float = 1.0,
        device: str = 'cuda',
    ):
        self.bundle = bundle
        self.rollout = rollout
        self.n_candidates = n_candidates
        self.n_elites = n_elites
        self.n_iters = n_iters
        self.horizon = horizon
        self.physics_top_k = physics_top_k
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.device = device

    @torch.no_grad()
    def plan(
        self,
        z_ctx: torch.Tensor,         # [1, T_ctx, N, D] 当前观测的latent context
        z_goal: torch.Tensor,         # [1, N, D] 目标图像的latent
        state_now: torch.Tensor,      # [1, state_dim] 当前末端状态
        env=None,                     # RollbackEnv实例（物理验证用，可选）
    ) -> Tuple[torch.Tensor, float]:
        """
        Returns:
            best_action_seq: [horizon, action_dim]  最优动作序列
            best_cost: float                         最优代价
        """
        # CEM初始分布：均匀高斯
        mu = torch.zeros(self.horizon, self.action_dim, device=self.device)
        sigma = torch.ones(self.horizon, self.action_dim, device=self.device)

        for cem_iter in range(self.n_iters):
            # 采样候选动作序列
            eps = torch.randn(self.n_candidates, self.horizon, self.action_dim,
                              device=self.device)
            candidates = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
            candidates = candidates.clamp(self.action_low, self.action_high)

            # 批量latent rollout（这步快）
            z_ctx_batch = z_ctx.expand(self.n_candidates, -1, -1, -1)
            state_seq = state_now.expand(self.n_candidates, self.horizon, -1)

            z_traj = self.rollout.rollout(z_ctx_batch, candidates, state_seq)
            # z_traj: [n_candidates, horizon, N, D]

            # 计算代价（终态与目标的距离，同时也可以加中间步的折扣代价）
            z_goal_batch = z_goal.expand(self.n_candidates, -1, -1)
            dists = self.rollout.compute_goal_distance(z_traj, z_goal_batch)
            # dists: [n_candidates, horizon]

            # 加权代价：更重视终态
            gamma = torch.pow(torch.tensor(0.9), torch.arange(self.horizon,
                              device=self.device)).flip(0)
            costs = (dists * gamma.unsqueeze(0)).mean(dim=1)  # [n_candidates]

            # 精英更新
            elite_idx = costs.argsort()[:self.n_elites]
            elites = candidates[elite_idx]
            mu = elites.mean(0)
            sigma = elites.std(0).clamp(min=1e-4)

        # 最优序列
        best_idx = costs.argsort()[0]
        best_seq = candidates[best_idx]  # [horizon, action_dim]
        best_cost = costs[best_idx].item()

        # 物理验证（如果提供了env）
        if env is not None and self.physics_top_k > 0:
            top_k_idx = costs.argsort()[:self.physics_top_k]
            best_seq, best_cost = self._physics_verify(
                env, candidates[top_k_idx], z_goal
            )

        return best_seq, best_cost

    def _physics_verify(self, env, top_k_seqs, z_goal):
        """
        对top-K候选在物理引擎里实际执行，选出真实效果最好的。
        这步会调用push_state/pop_state。
        """
        best_cost = float('inf')
        best_seq = top_k_seqs[0]

        for seq in top_k_seqs:
            env.push_state()
            total_reward = 0.0
            for action in seq:
                _, reward, done, _ = env.step(action.cpu().numpy())
                total_reward += reward
                if done:
                    break

            # 取最终帧的latent距离作为物理验证代价
            final_frames = env.get_frame_tensor(device=self.device)
            z_final = self.bundle.encode(final_frames)[:, -1, :, :]  # [1, N, D]
            cost = F.mse_loss(
                z_final.mean(1), z_goal.mean(1)
            ).item() - total_reward * 0.1  # 奖励作为正则

            env.pop_state()

            if cost < best_cost:
                best_cost = cost
                best_seq = seq

        return best_seq, best_cost
```

---

## 5. 训练层（training/）

### 5.1 ac_trainer.py（核心训练逻辑）

```python
"""
training/ac_trainer.py
AC-Predictor的无监督训练。

Loss组合（对齐V-JEPA 2-AC论文）：
  L_total = L_teacher_forcing + λ * L_rollout
  
  L_teacher_forcing: 每步用真实z_t预测z_{t+1}（短程精度）
  L_rollout:         自回归K步后的终态误差（长程稳定性）
  
训练数据来源：random_explorer.py采集的随机轨迹，无需任何标注。
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict
from models.model_bundle import ModelBundle
from data.dataset import TrajectoryDataset
from torch.utils.data import DataLoader


class ACTrainer:

    def __init__(
        self,
        bundle: ModelBundle,
        dataset: TrajectoryDataset,
        config: Dict,
    ):
        self.bundle = bundle
        self.config = config

        # 只训练predictor，encoder保持冻结
        self.optimizer = AdamW(
            bundle.predictor.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-2),
            betas=(0.9, 0.95),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('total_steps', 100_000),
            eta_min=config.get('lr_min', 1e-6),
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        self.rollout_horizon = config.get('rollout_horizon', 8)
        self.rollout_weight = config.get('rollout_weight', 0.5)

    def compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        batch格式：
          'frames':  [B, T+1, C, H, W]  ← T步历史 + 1步target
          'actions': [B, T, action_dim]  ← T步动作
          'states':  [B, T, state_dim]   ← T步状态
        """
        frames = batch['frames'].to(self.bundle.device)
        actions = batch['actions'].to(self.bundle.device)
        states = batch['states'].to(self.bundle.device)

        B, T_plus_1, C, H, W = frames.shape
        T = T_plus_1 - 1

        # Frozen encoder提取所有帧的潜向量
        with torch.no_grad():
            z_all = self.bundle.encode(frames)  # [B, T+1, N, D]
        z_ctx = z_all[:, :T, :, :]             # [B, T, N, D] context
        z_target = z_all[:, 1:, :, :]          # [B, T, N, D] target（方案A：shift 1）

        # ── Teacher-Forcing Loss ──────────────────────────
        # 每步都用真实z_t作为输入，不累积误差
        z_pred_tf = self.bundle.predict(z_ctx, actions, states)  # [B, T, N, D]
        loss_tf = F.smooth_l1_loss(z_pred_tf, z_target.detach())

        # ── Rollout Loss ──────────────────────────────────
        # 自回归rollout K步，用预测值接续（测试长程稳定性）
        K = min(self.rollout_horizon, T)
        z_roll = z_ctx[:, 0:1, :, :]  # 从第0帧开始
        rollout_losses = []

        for t in range(K):
            a_t = actions[:, t:t+1, :]  # [B, 1, action_dim]
            s_t = states[:, t:t+1, :]   # [B, 1, state_dim]
            z_next_pred = self.bundle.predict(z_roll, a_t, s_t)  # [B, 1, N, D]
            rollout_losses.append(
                F.smooth_l1_loss(z_next_pred, z_target[:, t:t+1, :].detach())
            )
            z_roll = z_next_pred  # 用预测值接续（不用真实值）

        loss_rollout = torch.stack(rollout_losses).mean()

        loss_total = loss_tf + self.rollout_weight * loss_rollout

        return {
            'loss_total': loss_total,
            'loss_tf': loss_tf.detach(),
            'loss_rollout': loss_rollout.detach(),
        }

    def train_step(self, batch: Dict) -> Dict[str, float]:
        self.optimizer.zero_grad()
        losses = self.compute_loss(batch)
        losses['loss_total'].backward()
        torch.nn.utils.clip_grad_norm_(
            self.bundle.predictor.parameters(),
            max_norm=1.0,
        )
        self.optimizer.step()
        self.scheduler.step()

        return {k: v.item() for k, v in losses.items()}
```

### 5.2 backbone_finetune.py

```python
"""
training/backbone_finetune.py
V-JEPA 2.1 encoder在本体采集数据上的domain adaptation。

策略：
  - 沿用V-JEPA 2.1的Dense Predictive Loss（所有token都参与loss）
  - Deep Self-Supervision（在encoder多个中间层施加loss）
  - 渐进解冻（从顶层向底层，防止灾难性遗忘）
  - EMA target encoder提供伪标签（防止表征坍塌）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import copy
import numpy as np
from models.loader import VJEPAEncoder


class EMAEncoder(nn.Module):
    """Target encoder，EMA更新，不参与梯度"""
    def __init__(self, online_encoder: VJEPAEncoder, momentum: float = 0.996):
        super().__init__()
        self.target = copy.deepcopy(online_encoder)
        for p in self.target.parameters():
            p.requires_grad = False
        self.momentum = momentum

    @torch.no_grad()
    def update(self, online: VJEPAEncoder):
        for p_online, p_target in zip(online.parameters(), self.target.parameters()):
            p_target.data.mul_(self.momentum).add_(p_online.data, alpha=1 - self.momentum)

    def encode(self, frames):
        return self.target.encode(frames)


class BackboneFinetune:

    def __init__(
        self,
        encoder: VJEPAEncoder,
        config: dict,
    ):
        self.encoder = encoder
        self.ema_encoder = EMAEncoder(encoder, momentum=config.get('ema_momentum', 0.996))
        self.device = encoder.device

        # 初始只解冻最后2个block
        self._current_unfrozen = config.get('initial_unfreeze', 2)
        self.encoder.unfreeze_top_k_blocks(self._current_unfrozen)
        self.max_unfreeze = config.get('max_unfreeze', 8)
        self.unfreeze_interval = config.get('unfreeze_interval_steps', 2000)

        # mask比例（对齐V-JEPA 2.1预训练设定）
        self.mask_ratio = config.get('mask_ratio', 0.75)
        self.context_mask_ratio = config.get('context_mask_ratio', 0.15)

        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, encoder.parameters()),
            lr=config.get('lr', 5e-5),
            weight_decay=1e-2,
        )
        self._step = 0

    def _sample_masks(self, B: int, T: int, N: int, device: torch.device):
        """
        采样时空mask，对齐V-JEPA 2.1的Dense Predictive Loss。
        context_mask：少量遮掩，encoder看大部分
        target_mask：大量遮掩，预测被遮掩区域
        """
        n_context = int(N * (1 - self.context_mask_ratio))
        n_target = int(N * self.mask_ratio)

        context_masks = []
        target_masks = []
        for _ in range(B * T):
            idx = torch.randperm(N, device=device)
            context_masks.append(idx[:n_context])
            target_masks.append(idx[n_context:n_context + n_target])

        return context_masks, target_masks

    def finetune_step(self, frames: torch.Tensor) -> float:
        """
        frames: [B, T, C, H, W]
        """
        self._step += 1
        B, T, C, H, W = frames.shape
        N = (H // self.encoder.patch_size) * (W // self.encoder.patch_size)

        # Target encoder提供Dense伪标签（所有patch）
        with torch.no_grad():
            z_target = self.ema_encoder.encode(frames)  # [B, T, N, D]

        # Online encoder前向（简化：全帧输入，用mask在loss上屏蔽）
        z_online = self.encoder.encode.__wrapped__(frames)  # 绕过no_grad
        # V-JEPA 2.1 Deep Self-Supervision：多层loss（简化为最后一层）
        loss = F.smooth_l1_loss(z_online, z_target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, self.encoder.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        self.ema_encoder.update(self.encoder)

        # 渐进解冻调度
        if (self._step % self.unfreeze_interval == 0 and
                self._current_unfrozen < self.max_unfreeze):
            self._current_unfrozen += 1
            self.encoder.unfreeze_top_k_blocks(self._current_unfrozen)

        return loss.item()
```

### 5.3 curriculum.py

```python
"""
training/curriculum.py
课程学习调度器：按认知复杂度从低到高推进训练任务。

Level 1 → Level 2 → Level 3 → Level 4 → Level 5
  (单步)   (多步)   (工具)   (层级)   (跨模态)

晋级条件：当前级别所有任务的平均成功率 ≥ threshold（默认0.7）
"""

from typing import List, Dict
from env.tasks.base_task import TASK_REGISTRY, CognitiveLevel
import numpy as np


class CurriculumScheduler:

    LEVEL_TASKS = {
        CognitiveLevel.SINGLE_CONTACT: [],         # Level 1由随机探索覆盖
        CognitiveLevel.MULTI_STEP:     ['nut_crushing', 'bottle_cap', 'tube_drop', 'detour'],
        CognitiveLevel.TOOL_AS_MEDIUM: ['stick_tool', 'trap_avoidance'],
        CognitiveLevel.HIERARCHICAL:   ['multi_tool'],
        CognitiveLevel.CROSS_MODAL:    ['mirror_imitation'],
    }

    def __init__(self, promotion_threshold: float = 0.70):
        self.current_level = CognitiveLevel.MULTI_STEP
        self.threshold = promotion_threshold
        self.success_history: Dict[str, List[float]] = {
            task: [] for task in TASK_REGISTRY
        }
        self.window_size = 100  # 最近100次评估的滑动窗口

    def record_result(self, task_name: str, success: bool):
        self.success_history[task_name].append(float(success))
        if len(self.success_history[task_name]) > self.window_size:
            self.success_history[task_name].pop(0)

    def current_tasks(self) -> List[str]:
        return self.LEVEL_TASKS[self.current_level]

    def current_success_rate(self) -> float:
        tasks = self.current_tasks()
        if not tasks:
            return 1.0
        rates = []
        for t in tasks:
            history = self.success_history[t]
            if history:
                rates.append(np.mean(history[-self.window_size:]))
        return np.mean(rates) if rates else 0.0

    def try_promote(self) -> bool:
        """检查是否满足晋级条件，满足则自动晋级"""
        if self.current_success_rate() >= self.threshold:
            levels = list(CognitiveLevel)
            current_idx = levels.index(self.current_level)
            if current_idx + 1 < len(levels):
                self.current_level = levels[current_idx + 1]
                print(f"[Curriculum] 晋级！当前认知层级 → {self.current_level.name}")
                return True
        return False

    def get_task_sampling_weights(self) -> Dict[str, float]:
        """
        采样权重：优先采样成功率低的任务（难度自适应）
        """
        tasks = self.current_tasks()
        weights = {}
        for t in tasks:
            rate = np.mean(self.success_history[t][-self.window_size:]) \
                   if self.success_history[t] else 0.5
            weights[t] = 1.0 - rate + 0.1  # 成功率越低权重越高
        total = sum(weights.values())
        return {t: w / total for t, w in weights.items()}
```

---

## 6. 数据层（data/）

### 6.1 trajectory_store.py

```python
"""
data/trajectory_store.py
HDF5格式轨迹存储。单个文件可存储数万条轨迹。

文件结构（HDF5）：
  /metadata
    total_episodes: int
    action_dim: int
    state_dim: int
    frame_shape: (H, W, C)
  /episodes
    /0000000
      frames:  [T+1, H, W, C]  uint8
      actions: [T, action_dim]  float32    ← 方案A：T步动作
      states:  [T, state_dim]   float32    ← T步状态
      task:    str
      success: bool
    /0000001
      ...
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Optional


class TrajectoryStore:

    def __init__(self, path: str, mode: str = 'a'):
        """mode: 'a'=追加写, 'r'=只读"""
        self.path = Path(path)
        self.mode = mode
        self._file: Optional[h5py.File] = None

    def __enter__(self):
        self._file = h5py.File(self.path, self.mode)
        if 'metadata' not in self._file:
            self._file.create_group('metadata')
            self._file.create_group('episodes')
            self._file['metadata'].attrs['total_episodes'] = 0
        return self

    def __exit__(self, *args):
        self._file.close()

    def write_episode(
        self,
        frames: np.ndarray,    # [T+1, H, W, C] uint8
        actions: np.ndarray,   # [T, action_dim] float32
        states: np.ndarray,    # [T, state_dim] float32
        task: str = '',
        success: bool = False,
    ) -> int:
        n = self._file['metadata'].attrs['total_episodes']
        ep_key = f'episodes/{n:07d}'
        ep = self._file.create_group(ep_key)

        ep.create_dataset('frames',  data=frames,   dtype='uint8',   compression='lzf')
        ep.create_dataset('actions', data=actions,  dtype='float32')
        ep.create_dataset('states',  data=states,   dtype='float32')
        ep.attrs['task'] = task
        ep.attrs['success'] = success
        ep.attrs['length'] = len(actions)

        self._file['metadata'].attrs['total_episodes'] = n + 1
        return n

    def read_episode(self, idx: int) -> dict:
        ep = self._file[f'episodes/{idx:07d}']
        return {
            'frames':  ep['frames'][:],
            'actions': ep['actions'][:],
            'states':  ep['states'][:],
            'task':    ep.attrs['task'],
            'success': ep.attrs['success'],
        }

    @property
    def total_episodes(self) -> int:
        return self._file['metadata'].attrs['total_episodes']
```

### 6.2 dataset.py

```python
"""
data/dataset.py
PyTorch Dataset，从HDF5读取训练batch。
支持按任务名过滤和随机子序列采样。
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from .trajectory_store import TrajectoryStore
from typing import Optional, List


class TrajectoryDataset(Dataset):
    """
    每个样本是一条轨迹的随机子序列：
      frames:  [T+1, C, H, W]  float32，归一化到[0,1]
      actions: [T, action_dim]
      states:  [T, state_dim]
    
    T = seq_len（训练时固定长度）
    """

    def __init__(
        self,
        store_path: str,
        seq_len: int = 8,
        task_filter: Optional[List[str]] = None,
        augment: bool = True,
    ):
        self.store_path = store_path
        self.seq_len = seq_len
        self.task_filter = task_filter
        self.augment = augment

        # 建立索引（只在初始化时扫描一次）
        self._index = self._build_index()

    def _build_index(self) -> List[dict]:
        index = []
        with TrajectoryStore(self.store_path, mode='r') as store:
            for i in range(store.total_episodes):
                ep = store._file[f'episodes/{i:07d}']
                task = ep.attrs['task']
                length = ep.attrs['length']
                if self.task_filter and task not in self.task_filter:
                    continue
                if length < self.seq_len + 1:
                    continue
                index.append({'ep_idx': i, 'length': length, 'task': task})
        return index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        meta = self._index[idx]
        with TrajectoryStore(self.store_path, mode='r') as store:
            ep = store.read_episode(meta['ep_idx'])

        # 随机起点采样子序列
        T_total = meta['length']
        start = np.random.randint(0, T_total - self.seq_len)
        end = start + self.seq_len

        frames = ep['frames'][start:end + 1]  # [T+1, H, W, C] uint8
        actions = ep['actions'][start:end]    # [T, action_dim]
        states = ep['states'][start:end]      # [T, state_dim]

        # 转换格式
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        # [T+1, C, H, W]

        return {
            'frames':  frames,
            'actions': torch.from_numpy(actions).float(),
            'states':  torch.from_numpy(states).float(),
            'task':    meta['task'],
        }
```

---

## 7. 配置系统

### configs/model/vitl.yaml

```yaml
encoder:
  variant: vitl           # ViT-L encoder
  device: cuda
  embed_dim: 1024         # 固定，由官方模型决定
  patch_size: 16
  image_size: 224

predictor:
  predictor_dim: 768      # AC-Predictor内部维度
  n_layers: 12            # 官方24层，我们用12层
  n_heads: 12
  action_dim: 7           # [Δx,Δy,Δz,Δroll,Δpitch,Δyaw,gripper]
  state_dim: 7            # [x,y,z,roll,pitch,yaw,gripper]
  max_frames: 16
  dropout: 0.0
```

### configs/training/ac_train.yaml

```yaml
training:
  lr: 1e-4
  lr_min: 1e-6
  weight_decay: 1e-2
  batch_size: 32
  total_steps: 200_000
  rollout_horizon: 8
  rollout_weight: 0.5
  grad_clip: 1.0
  
  # 日志与保存
  log_interval: 100
  save_interval: 5000
  checkpoint_dir: checkpoints/ac_predictor/

data:
  store_path: data/trajectories.h5
  seq_len: 8
  num_workers: 4
```

### configs/training/finetune.yaml

```yaml
finetuning:
  lr: 5e-5
  weight_decay: 1e-2
  ema_momentum: 0.996
  mask_ratio: 0.75
  context_mask_ratio: 0.15
  initial_unfreeze: 2       # 初始解冻最后2个block
  max_unfreeze: 8           # 最多解冻8个block
  unfreeze_interval_steps: 2000
  
data:
  store_path: data/embodied_trajectories.h5
  seq_len: 8
```

---

## 8. 可执行脚本接口

### scripts/collect_data.py

```python
"""
随机策略采集轨迹数据，用于AC-Predictor无监督训练。
用法：
  python scripts/collect_data.py --task nut_crushing --n_episodes 5000
                                 --output data/trajectories.h5
"""
# ...（使用random_explorer.py）
```

### scripts/train_ac.py

```python
"""
训练AC-Predictor。
用法：
  python scripts/train_ac.py --config configs/training/ac_train.yaml
                             --model-config configs/model/vitl.yaml
"""
# ...（使用ac_trainer.py）
```

### scripts/run_planning.py

```python
"""
在指定任务上运行规划推理（不训练）。
用法：
  python scripts/run_planning.py --task bottle_cap
                                 --checkpoint checkpoints/ac_predictor/best.pt
                                 --goal-image assets/goals/bottle_cap_open.png
"""
# ...（使用cem_planner.py）
```

---

## 9. 完整训练流程

```
阶段0：环境验证（约1天）
  ├── 安装dm_control + MuJoCo
  ├── 运行所有8个任务场景，检查物理仿真正确性
  └── 验证push_state/pop_state的determinism

阶段1：数据采集（Level 1-2任务，约2-3天）
  ├── 随机策略在 nut_crushing / bottle_cap / tube_drop / detour 采集
  ├── 目标：每个任务 ~5000 episodes
  └── 存储到 data/trajectories.h5

阶段2：ViT-B原型验证（约1周）
  ├── 加载官方ViT-B encoder（冻结）
  ├── 训练 AC-Predictor（12层，dim=768）
  ├── 验证：latent rollout的预测误差随步数的增长曲线
  ├── 验证：CEM planner能在nut_crushing上达到 >50% 成功率
  └── → 框架验证通过，切换ViT-L

阶段3：ViT-L正式训练（约2-3周）
  ├── 用ViT-L encoder重新采集（表征质量更好）
  ├── 训练 AC-Predictor（full scale）
  ├── curriculum调度：Level 2 → Level 3 → Level 4
  └── 评估：按TASK_REGISTRY的success_threshold逐任务验证

阶段4：Encoder微调（可选，Level 4-5任务，视需要）
  ├── 本体采集专项数据（tool use场景）
  ├── 渐进解冻ViT-L顶层block
  └── 继续AC-Predictor训练（联合优化）

终极目标：mirror_imitation（Level 5）
  ├── 输入：人类执行瓶盖拧开的视频clip（z_goal来自视频末帧）
  ├── 规划：CEM找到使机械臂末态接近视频末帧latent的动作序列
  └── 评估：zero-shot模仿成功率
```

---

## 10. 关键接口一览

```python
# 加载模型
from models.model_bundle import ModelBundle, ModelConfig
bundle = ModelBundle(ModelConfig(encoder_variant='vitl'))

# 创建环境
from env.tasks.nut_crushing import NutCrushingEnv
env = NutCrushingEnv()
obs = env.reset()

# 编码当前观测
frames = env.get_frame_tensor(device='cuda')    # [1, 8, 3, 224, 224]
z = bundle.encode(frames)                        # [1, 8, 196, 1024]

# 规划
from planning.latent_rollout import LatentRollout
from planning.cem_planner import CEMPlanner
rollout = LatentRollout(bundle)
planner = CEMPlanner(bundle, rollout)

goal_img = load_goal_image('assets/goals/nut_crushed.png')
z_goal = bundle.encode_goal(goal_img)            # [1, 196, 1024]
state = env.get_state_vector()

best_actions, cost = planner.plan(z, z_goal, torch.tensor(state))

# 执行
for action in best_actions:
    obs, reward, done, info = env.step(action.numpy())
```

---

*设计版本：v0.1 / 2026-05 / 基于 V-JEPA 2.1 (ViT-L) + dm_control 3.x + MuJoCo 3.1*