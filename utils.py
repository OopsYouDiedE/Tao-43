"""utils.py — VJEPA-Gym 共享实用工具。

这里的函数之前分散在 main.py 中，现在被两个或多个 run_* 命令使用。
main.py 会从这里导入它所需的所有内容。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from core_models import ActionConditionedPredictor, PredictorConfig, VJEPAEncoder, mean_patch_cost

if TYPE_CHECKING:
    pass  # avoid circular: environment imported only for type hints in callers


# ---------------------------------------------------------------------------
# Config / device / seed
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def select_device(config: dict) -> torch.device:
    prefer_cuda = bool(config.get("device", {}).get("prefer_cuda", True))
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_python_compatible() -> None:
    if sys.version_info[:2] < (3, 10):
        version = ".".join(map(str, sys.version_info[:3]))
        raise RuntimeError(f"验证程序必须在 Python 3.10 或更高版本上运行，当前 Python 版本为 {version}。")


def setup_run(args: argparse.Namespace) -> tuple[dict, torch.device, int]:
    """所有 run_* 命令（除 pipeline-latent 外）共享的通用前置准备。

    替换了重复的 5 行代码块：

        assert_python_compatible()
        config = load_config(args.config)
        set_seed(...)
        device = select_device(config)
        print(f"python/torch/device: ...")

    返回 (config, device, seed)。
    """
    assert_python_compatible()
    config = load_config(args.config)
    seed = int(getattr(args, "seed", None) or config.get("seed", 43))
    set_seed(seed)
    device = select_device(config)
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"device: {device}")
    return config, device, seed


# ---------------------------------------------------------------------------
# Model construction helpers
# ---------------------------------------------------------------------------

def frames_to_tensor(
    frames: list[np.ndarray] | np.ndarray,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"frames 必须是 [T, H, W, 3]，实际得到 {array.shape}")
    array = array.astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).unsqueeze(0)
    return tensor.to(device) if device else tensor


def predictor_config_from_dict(data: dict) -> PredictorConfig:
    allowed = PredictorConfig.__dataclass_fields__.keys()
    return PredictorConfig(**{key: data[key] for key in allowed if key in data})


def build_encoder(
    config: dict,
    require_official: bool,
    device: torch.device,
) -> VJEPAEncoder:
    vjepa_cfg = config["vjepa"]
    if require_official and not bool(vjepa_cfg.get("require_official", True)):
        raise RuntimeError(
            "--require-official-vjepa 要求 configs.yaml 中的 vjepa.require_official=true。"
        )
    encoder = VJEPAEncoder(
        hub_repo=vjepa_cfg["hub_repo"],
        model_name=vjepa_cfg["model_name"],
        require_official=require_official,
        pretrained=bool(vjepa_cfg.get("pretrained", True)),
        checkpoint_base_url=vjepa_cfg.get("checkpoint_base_url"),
        patches_per_frame=int(vjepa_cfg["patches_per_frame"]),
        embed_dim=int(vjepa_cfg["embed_dim"]),
        image_size=int(vjepa_cfg["image_size"]),
        device=device,
    )
    print(
        f"official_vjepa: 已加载 {vjepa_cfg['model_name']} "
        f"pretrained={bool(vjepa_cfg.get('pretrained', True))}"
    )
    return encoder


def arg_or_config(value, config: dict, key: str, default):
    """如果提供了 CLI *value* 则返回它，否则在 *config* 中查找 *key*，再否则返回 *default*。"""
    return value if value is not None else config.get(key, default)


# ---------------------------------------------------------------------------
# Goal-latent construction (shared by oracle_demo, collect_*, eval_ac)
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_goal_latent(
    env,
    encoder,
    device: torch.device,
    target_steps: int,
) -> tuple[torch.Tensor, np.ndarray, dict]:
    """在 push/pop 沙盒中运行一个固定的绕路序列，并返回
    终端隐变量、渲染的目标帧以及一个摘要字典。"""
    env.push_state()
    try:
        actions = make_ground_truth_detour_actions(target_steps)
        summary = env.rollout_sequence(actions, record_frames=False)
        target_frame = env.render().copy()
        goal_z = encoder(env.get_frame_tensor(str(device)))[:, -1].detach()
    finally:
        env.pop_state()
    return goal_z, target_frame, summary


def make_ground_truth_detour_actions(target_steps: int) -> np.ndarray:
    """三阶段脚本化绕路：45% 向右 → 35% 向上 → 20% 对角线。"""
    target_steps = max(target_steps, 12)
    phases = [
        (0.45, np.array([1.0, 0.0], dtype=np.float32)),
        (0.35, np.array([0.0, 1.0], dtype=np.float32)),
        (0.20, np.array([-0.25, 0.2], dtype=np.float32)),
    ]
    actions: list[np.ndarray] = []
    allocated = 0
    for idx, (fraction, xy_action) in enumerate(phases):
        count = (
            int(round(target_steps * fraction))
            if idx < len(phases) - 1
            else target_steps - allocated
        )
        allocated += count
        for _ in range(max(0, count)):
            action = np.zeros(7, dtype=np.float32)
            action[:2] = xy_action
            actions.append(action)
    return np.stack(actions, axis=0)


# ---------------------------------------------------------------------------
# Diagnostic helpers (used in oracle_demo)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_direct_vs_detour(
    env,
    planner,
    goal_z: torch.Tensor,
    horizon: int,
) -> dict[str, float]:
    direct = np.zeros((1, horizon, 7), dtype=np.float32)
    direct[:, :, 0] = 1.0
    detour = np.zeros((1, horizon, 7), dtype=np.float32)
    half = max(1, horizon // 2)
    detour[:, :half, 0] = 1.0
    detour[:, half:, 1] = 1.0
    sequences = np.concatenate([direct, detour], axis=0)
    costs, rewards, xs, ys = planner.evaluate_sequences(env, sequences, goal_z)
    return {
        "direct_cost": float(costs[0]),
        "detour_cost": float(costs[1]),
        "direct_reward": float(rewards[0]),
        "detour_reward": float(rewards[1]),
        "direct_x": float(xs[0]),
        "direct_y": float(ys[0]),
        "detour_x": float(xs[1]),
        "detour_y": float(ys[1]),
    }


# ---------------------------------------------------------------------------
# Candidate-rollout visualization (shared by oracle_demo + eval_ac)
# ---------------------------------------------------------------------------

def collect_candidate_rollouts(
    env,
    sequences: np.ndarray,
    action_repeat: int,
) -> list[list[np.ndarray]]:
    rollouts: list[list[np.ndarray]] = []
    before = env.snapshot_signature()
    for sequence in sequences:
        env.push_state()
        try:
            repeated = (
                np.repeat(sequence, action_repeat, axis=0)
                if action_repeat > 1
                else sequence
            )
            result = env.rollout_sequence(repeated, record_frames=True)
            rollouts.append(result["frames"])
        finally:
            env.pop_state()
        if env.snapshot_signature() != before:
            raise RuntimeError(
                "候选可视化展开（rollout）改变了实时环境状态。"
            )
    return rollouts


def make_candidate_grid_frames(rollouts: list[list[np.ndarray]]) -> list[np.ndarray]:
    if not rollouts:
        return []
    max_len = max(len(frames) for frames in rollouts)
    cols = min(2, len(rollouts))
    rows = int(np.ceil(len(rollouts) / cols))
    blank = np.zeros_like(rollouts[0][0])
    grid_frames: list[np.ndarray] = []
    for frame_idx in range(max_len):
        tiles = [
            rollout[min(frame_idx, len(rollout) - 1)] for rollout in rollouts
        ]
        while len(tiles) < rows * cols:
            tiles.append(blank)
        row_imgs = [
            np.concatenate(tiles[row * cols : row * cols + cols], axis=1)
            for row in range(rows)
        ]
        grid_frames.append(np.concatenate(row_imgs, axis=0))
    return grid_frames


# ---------------------------------------------------------------------------
# Video / report output (shared by oracle_demo + eval_ac)
# ---------------------------------------------------------------------------

def save_run_videos(
    output_dir: Path,
    execution_frames: list[np.ndarray],
    candidate_grid_rollouts: list[list[np.ndarray]] | None,
    target_frame: np.ndarray,
    video_fps: int,
) -> None:
    """将 execution.mp4 和 candidate_grid.mp4 写入 *output_dir*。"""
    imageio.mimsave(
        output_dir / "execution.mp4",
        execution_frames,
        fps=video_fps,
        codec="libx264",
        macro_block_size=16,
    )
    grid_frames = (
        make_candidate_grid_frames(candidate_grid_rollouts)
        if candidate_grid_rollouts
        else [target_frame]
    )
    imageio.mimsave(
        output_dir / "candidate_grid.mp4",
        grid_frames,
        fps=video_fps,
        codec="libx264",
        macro_block_size=16,
    )


def write_metrics_csv(path: Path, metrics: list[dict[str, float]]) -> None:
    if not metrics:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)


def write_summary_html(
    path: Path,
    metrics: list[dict[str, float]],
    initial_x: float,
    initial_reward: float,
    final_x: float,
    final_reward: float,
    goal_x: float,
    target_summary: dict,
) -> None:
    reward_svg = make_svg_chart(metrics, "reward", "Reward")
    cost_svg = make_svg_chart(metrics, "latent_cost", "Latent Cost")
    rows = "\n".join(
        "<tr>"
        f"<td>{int(row['step'])}</td>"
        f"<td>{row['chosen_action0']:+.3f}</td>"
        f"<td>{row['chosen_action1']:+.3f}</td>"
        f"<td>{row['x']:.4f}</td>"
        f"<td>{row['y']:.4f}</td>"
        f"<td>{row['reward']:.4f}</td>"
        f"<td>{row['latent_cost']:.6f}</td>"
        "</tr>"
        for row in metrics
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>VJEPA-Gym Oracle Demo</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    img {{ max-width: 480px; margin-right: 16px; border: 1px solid #d0d7de; }}
    table {{ border-collapse: collapse; margin-top: 16px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 10px; text-align: right; }}
    th {{ background: #f6f8fa; }}
    .charts {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }}
  </style>
</head>
<body>
  <h1>VJEPA-Gym Oracle Demo</h1>
  <p>Initial x={initial_x:.4f}, reward={initial_reward:.4f}; final x={final_x:.4f}, reward={final_reward:.4f}; goal x={goal_x:.4f}.</p>
  <p>Target image x={target_summary['x']:.4f}, y={target_summary['y']:.4f}, reward={target_summary['reward']:.4f}.</p>
  <div>
    <video src="execution.mp4" controls autoplay muted loop width="480"></video>
    <video src="candidate_grid.mp4" controls autoplay muted loop width="480"></video>
  </div>
  <div class="charts">{reward_svg}{cost_svg}</div>
  <table>
    <thead><tr><th>step</th><th>action0</th><th>action1</th><th>x</th><th>y</th><th>reward</th><th>latent_cost</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def make_svg_chart(metrics: list[dict[str, float]], key: str, title: str) -> str:
    width, height = 360, 180
    if not metrics:
        return f"<svg width='{width}' height='{height}'><text x='16' y='24'>{title}: no data</text></svg>"
    values = np.asarray([row[key] for row in metrics], dtype=np.float64)
    vmin = float(values.min())
    vmax = float(values.max())
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1.0
    xs = np.linspace(24, width - 16, num=len(values))
    ys = height - 28 - ((values - vmin) / (vmax - vmin)) * (height - 56)
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return (
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"
        f"<text x='16' y='20' font-size='14'>{title}</text>"
        f"<polyline points='{points}' fill='none' stroke='#0969da' stroke-width='2'/>"
        f"<text x='16' y='{height - 8}' font-size='11'>min={vmin:.4f} max={vmax:.4f}</text>"
        "</svg>"
    )


# ---------------------------------------------------------------------------
# 内存感知参数自动调节
# ---------------------------------------------------------------------------

def _get_free_vram_mb(device: torch.device) -> float | None:
    """返回 CUDA 设备当前可用显存（MB）；非 CUDA 设备返回 None。"""
    if device.type != "cuda":
        return None
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
        return free_bytes / (1024.0 ** 2)
    except Exception:
        return None


def _get_free_ram_mb() -> float:
    """返回系统当前可用内存（MB）；检测失败则返回保守默认值 4096.0。"""
    # 优先 psutil（跨平台、最精确）
    try:
        import psutil  # type: ignore[import]
        return psutil.virtual_memory().available / (1024.0 ** 2)
    except ImportError:
        pass
    # Linux: /proc/meminfo
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as _f:
            for _line in _f:
                if _line.startswith("MemAvailable:"):
                    return float(_line.split()[1]) / 1024.0  # kB → MB
    except OSError:
        pass
    # Windows: GlobalMemoryStatusEx via ctypes
    try:
        import ctypes
        import ctypes.wintypes

        class _MEMSTATEX(ctypes.Structure):
            _fields_ = [
                ("dwLength",                ctypes.c_ulong),
                ("dwMemoryLoad",            ctypes.c_ulong),
                ("ullTotalPhys",            ctypes.c_ulonglong),
                ("ullAvailPhys",            ctypes.c_ulonglong),
                ("ullTotalPageFile",        ctypes.c_ulonglong),
                ("ullAvailPageFile",        ctypes.c_ulonglong),
                ("ullTotalVirtual",         ctypes.c_ulonglong),
                ("ullAvailVirtual",         ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        _stat = _MEMSTATEX()
        _stat.dwLength = ctypes.sizeof(_stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(_stat))  # type: ignore[attr-defined]
        return float(_stat.ullAvailPhys) / (1024.0 ** 2)
    except Exception:
        pass
    return 4096.0  # 保守默认：4 GB


# 训练小批量对梯度质量的边际收益递减（过大批次可能伤收敛）
_BATCH_SIZE_CAP:   int = 32
# 超过此值 GPU 利用率提升有限
_ENCODE_BATCH_CAP: int = 128
# 超过此值 DataLoader worker 调度开销超过 I/O 收益
_NUM_WORKERS_CAP:  int = 8


def _predictor_param_count(cfg: dict | PredictorConfig | None) -> int:
    """粗略估算 ActionConditionedPredictor 参数量，用于显存预算。"""
    if isinstance(cfg, PredictorConfig):
        visual_dim = int(cfg.visual_dim)
        pred_dim = int(cfg.pred_dim)
        action_dim = int(cfg.action_dim)
        state_dim = int(cfg.state_dim)
        num_layers = int(cfg.num_layers)
    else:
        data = cfg or {}
        visual_dim = int(data.get("visual_dim", 768))
        pred_dim = int(data.get("pred_dim", 768))
        action_dim = int(data.get("action_dim", 7))
        state_dim = int(data.get("state_dim", 7))
        num_layers = int(data.get("num_layers", 12))

    projections = (visual_dim + action_dim + state_dim) * pred_dim + 3 * pred_dim
    output = pred_dim * visual_dim + visual_dim
    temporal = 64 * pred_dim
    # MultiheadAttention: qkv + out ~= 4*d*d. FFN: d*4d + 4d*d ~= 8*d*d.
    # Norms and biases are tiny by comparison but included as a small linear term.
    layer = 12 * pred_dim * pred_dim + 16 * pred_dim
    return projections + output + temporal + num_layers * layer


def _estimate_predictor_static_vram_mb(cfg: dict | PredictorConfig | None) -> float:
    # fp32 params + grads + AdamW states ~= 16 bytes/param. Add CUDA/module overhead.
    return max(1024.0, _predictor_param_count(cfg) * 16.0 / (1024.0 ** 2) + 512.0)


def _predictor_cfg_value(cfg: dict | PredictorConfig | None, key: str, default: int) -> int:
    if isinstance(cfg, PredictorConfig):
        return int(getattr(cfg, key))
    return int((cfg or {}).get(key, default))


def _latent_train_sample_vram_mb(
    *,
    predictor_cfg: dict | PredictorConfig | None,
    sequence_len: int,
    rollout_steps: int,
    patches_per_frame: int,
    amp: bool,
    include_action_contrast: bool,
) -> float:
    """估算 latent trainer 单样本峰值显存。

    这里显式计算 Transformer attention 的 O(T^2) 项。loss 里会跑一次
    teacher-forcing、若干次 rollout，batch>=2 时还会额外跑一次 action contrast。
    """
    pred_dim = _predictor_cfg_value(predictor_cfg, "pred_dim", 768)
    num_layers = _predictor_cfg_value(predictor_cfg, "num_layers", 12)
    num_heads = _predictor_cfg_value(predictor_cfg, "num_heads", 12)
    visual_dim = _predictor_cfg_value(predictor_cfg, "visual_dim", 768)
    bytes_per_elem = 2.0 if amp else 4.0

    tokens_per_frame = int(patches_per_frame) + 2
    seq = max(1, int(sequence_len))
    steps = max(0, min(int(rollout_steps), seq))
    token_counts = [seq * tokens_per_frame]
    token_counts.extend((step + 1) * tokens_per_frame for step in range(steps))
    if include_action_contrast:
        token_counts.append(seq * tokens_per_frame)

    sum_tokens = float(sum(token_counts))
    sum_tokens_sq = float(sum(t * t for t in token_counts))

    # Attention score/probability tensors dominate and scale with heads*S^2.
    attn_mb = num_layers * num_heads * sum_tokens_sq * bytes_per_elem / (1024.0 ** 2)
    # QKV, FFN, residual/norm, output/loss tensors scale roughly with S*D.
    hidden_mb = num_layers * sum_tokens * pred_dim * bytes_per_elem * 28.0 / (1024.0 ** 2)
    io_mb = (seq + 1) * patches_per_frame * visual_dim * 4.0 * 6.0 / (1024.0 ** 2)
    # Empirical safety factor for PyTorch TransformerEncoder backward workspaces.
    return (attn_mb * 1.15) + hidden_mb + io_mb + 256.0


def _collector_window_vram_mb(
    *,
    sequence_len: int,
    patches_per_frame: int,
    embed_dim: int,
    amp: bool,
) -> float:
    """估算 collector 每个 window 编码时的推理显存。"""
    bytes_per_elem = 2.0 if amp else 4.0
    frames = max(1, int(sequence_len) + 1)
    tokens = max(1, int(patches_per_frame))
    dim = max(1, int(embed_dim))
    # V-JEPA inference has no backward graph, but hub models still allocate
    # attention/MLP workspaces. Keep this conservative under concurrent training.
    token_mb = frames * tokens * dim * bytes_per_elem * 18.0 / (1024.0 ** 2)
    image_mb = frames * 3 * 384 * 384 * 4.0 * 3.0 / (1024.0 ** 2)
    return max(768.0, token_mb + image_mb)


def probe_safe_memory_params(
    device: torch.device,
    batch_size: int,
    grad_accum: int,
    encode_batch_size: int = 32,
    num_workers: int = 4,
    *,
    workload: str = "generic",
    predictor_cfg: dict | PredictorConfig | None = None,
    sequence_len: int = 4,
    rollout_steps: int = 4,
    patches_per_frame: int = 576,
    embed_dim: int = 768,
    amp: bool = True,
    allow_batch_growth: bool = False,
    model_vram_mb: float | None = None,
    per_sample_vram_mb: float | None = None,
    vram_headroom_mb: float = 2048.0,
    reserved_vram_mb: float = 0.0,
    ram_headroom_mb: float = 1024.0,
    ram_per_worker_mb: float = 512.0,
) -> dict[str, int]:
    """根据可用显存/内存**双向自动调节**参数。

    根据 workload 估算 batch_size / encode_batch_size / num_workers。
    默认只在危险时压低 batch_size，不主动放大；有效批量
    （batch_size × grad_accum）通过 grad_accum 尽量保持不变。

    VRAM 调节逻辑
    -------------
    可用预算 = free_vram − vram_headroom − reserved_vram − model_vram
    safe_bs    = max(1, 预算 ÷ 估算单样本显存)
    optimal_bs = min(safe_bs, BATCH_SIZE_CAP)

    若 optimal_bs ≠ batch_size：
      → grad_accum = max(1, ceil(eff / optimal_bs))  [同步缩放]

    encode_batch_size 按 collector 的每 window 推理显存独立计算并加上界。

    RAM 调节逻辑
    ------------
    safe_workers   = (free_ram − ram_headroom) ÷ ram_per_worker
    optimal_workers = min(safe_workers, cpu_count, NUM_WORKERS_CAP)
    """
    result: dict[str, int] = {
        "batch_size":        batch_size,
        "grad_accum":        grad_accum,
        "encode_batch_size": encode_batch_size,
        "num_workers":       num_workers,
    }
    changed: list[str] = []
    eff = batch_size * grad_accum  # 有效批量（始终不变）

    if model_vram_mb is None:
        if workload in {"latent_trainer", "trainer", "full_trainer"}:
            estimated_model_vram = _estimate_predictor_static_vram_mb(predictor_cfg)
        else:
            estimated_model_vram = 1024.0
    else:
        estimated_model_vram = float(model_vram_mb)

    def _candidate_sample_mb(candidate_bs: int) -> float:
        if per_sample_vram_mb is not None:
            return float(per_sample_vram_mb)
        if workload in {"latent_trainer", "trainer"}:
            return _latent_train_sample_vram_mb(
                predictor_cfg=predictor_cfg,
                sequence_len=sequence_len,
                rollout_steps=rollout_steps,
                patches_per_frame=patches_per_frame,
                amp=amp and device.type == "cuda",
                include_action_contrast=candidate_bs >= 2,
            )
        if workload == "full_trainer":
            # Full trainer additionally runs V-JEPA forward before the predictor.
            return _latent_train_sample_vram_mb(
                predictor_cfg=predictor_cfg,
                sequence_len=sequence_len,
                rollout_steps=rollout_steps,
                patches_per_frame=patches_per_frame,
                amp=amp and device.type == "cuda",
                include_action_contrast=False,
            ) + _collector_window_vram_mb(
                sequence_len=sequence_len,
                patches_per_frame=patches_per_frame,
                embed_dim=embed_dim,
                amp=amp and device.type == "cuda",
            )
        return 400.0

    # ── VRAM ──────────────────────────────────────────────────────────────
    free_vram = _get_free_vram_mb(device)
    if free_vram is not None:
        budget = max(0.0, free_vram - vram_headroom_mb - reserved_vram_mb - estimated_model_vram)
        max_candidate = _BATCH_SIZE_CAP if allow_batch_growth else max(1, int(batch_size))
        optimal_bs = 1
        estimated_sample_mb = _candidate_sample_mb(1)
        for candidate in range(1, max_candidate + 1):
            candidate_sample_mb = _candidate_sample_mb(candidate)
            if candidate * candidate_sample_mb <= budget:
                optimal_bs = candidate
                estimated_sample_mb = candidate_sample_mb
            else:
                break
        if budget < estimated_sample_mb:
            changed.append(
                f"warning: batch_size=1 估算仍可能超过预算 "
                f"(budget={budget:.0f} MB, est_sample={estimated_sample_mb:.0f} MB)"
            )

        if optimal_bs != batch_size:
            new_ga    = max(1, math.ceil(eff / optimal_bs))
            direction = "↑扩展" if optimal_bs > batch_size else "↓压低"
            changed.append(
                f"batch_size {batch_size}→{optimal_bs} {direction}, "
                f"grad_accum {grad_accum}→{new_ga} "
                f"(有效批量={optimal_bs * new_ga}, "
                f"free_vram={free_vram:.0f} MB, budget={budget:.0f} MB, "
                f"est_sample={estimated_sample_mb:.0f} MB)"
            )
            result["batch_size"] = optimal_bs
            result["grad_accum"] = new_ga

        enc_sample_mb = _collector_window_vram_mb(
            sequence_len=sequence_len,
            patches_per_frame=patches_per_frame,
            embed_dim=embed_dim,
            amp=amp and device.type == "cuda",
        )
        safe_enc = max(1, int(budget / enc_sample_mb))
        enc_cap = _ENCODE_BATCH_CAP if allow_batch_growth else max(1, int(encode_batch_size))
        optimal_enc = min(safe_enc, enc_cap)
        if optimal_enc != encode_batch_size:
            direction = "↑扩展" if optimal_enc > encode_batch_size else "↓压低"
            changed.append(
                f"encode_batch_size {encode_batch_size}→{optimal_enc} "
                f"({direction}, est_window={enc_sample_mb:.0f} MB)"
            )
            result["encode_batch_size"] = optimal_enc

    # ── RAM ───────────────────────────────────────────────────────────────
    free_ram        = _get_free_ram_mb()
    usable_ram      = max(0.0, free_ram - ram_headroom_mb)
    safe_workers    = max(0, int(usable_ram / ram_per_worker_mb))
    optimal_workers = min(safe_workers, os.cpu_count() or 1, _NUM_WORKERS_CAP)
    if optimal_workers != num_workers:
        direction = "↑扩展" if optimal_workers > num_workers else "↓压低"
        changed.append(
            f"num_workers {num_workers}→{optimal_workers} "
            f"({direction}, free_ram={free_ram:.0f} MB)"
        )
        result["num_workers"] = optimal_workers

    # ── 报告 ──────────────────────────────────────────────────────────────
    vram_str  = f"{free_vram:.0f} MB" if free_vram is not None else "N/A"
    final_bs  = result["batch_size"]
    final_ga  = result["grad_accum"]
    final_enc = result["encode_batch_size"]
    final_wk  = result["num_workers"]
    status    = "参数已自动调节" if changed else "参数与请求值一致"
    print(
        f"[auto-tune] {status} — "
        f"workload={workload} "
        f"batch={final_bs} accum={final_ga} eff={final_bs * final_ga} "
        f"enc_bs={final_enc} workers={final_wk} "
        f"| free_vram={vram_str} reserved_vram={reserved_vram_mb:.0f} MB "
        f"model_vram={estimated_model_vram:.0f} MB free_ram={free_ram:.0f} MB"
    )
    for msg in changed:
        print(f"  ↳ {msg}")
    return result
