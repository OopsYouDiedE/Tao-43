"""utils.py — VJEPA-Gym 共享实用工具。

这里的函数之前分散在 main.py 中，现在被两个或多个 run_* 命令使用。
main.py 会从这里导入它所需的所有内容。
"""

from __future__ import annotations

import argparse
import csv
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


def assert_python_310() -> None:
    if sys.version_info[:2] != (3, 10):
        version = ".".join(map(str, sys.version_info[:3]))
        raise RuntimeError(f"验证程序必须在 Python 3.10 上运行，当前 Python 版本为 {version}。")


def setup_run(args: argparse.Namespace) -> tuple[dict, torch.device, int]:
    """所有 run_* 命令（除 pipeline-latent 外）共享的通用前置准备。

    替换了重复的 5 行代码块：

        assert_python_310()
        config = load_config(args.config)
        set_seed(...)
        device = select_device(config)
        print(f"python/torch/device: ...")

    返回 (config, device, seed)。
    """
    assert_python_310()
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
