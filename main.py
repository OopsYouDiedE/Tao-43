from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from core_models import ActionConditionedPredictor, PredictorConfig, TOKENS_PER_FRAME, VJEPAEncoder, mean_patch_cost
from environment import make_smoke_env
from latent_planner import LatentCEMPlanner, OracleCEMPlanner, oracle_cem_config_from_dict, cem_config_from_dict
from trainer_engine import compute_ac_predictor_loss


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
        raise RuntimeError(f"Verification must run on Python 3.10, got Python {version}.")


def predictor_config_from_dict(data: dict) -> PredictorConfig:
    allowed = PredictorConfig.__dataclass_fields__.keys()
    return PredictorConfig(**{key: data[key] for key in allowed if key in data})


def build_encoder(config: dict, require_official: bool, device: torch.device) -> VJEPAEncoder:
    vjepa_cfg = config["vjepa"]
    if require_official and not bool(vjepa_cfg.get("require_official", True)):
        raise RuntimeError("--require-official-vjepa requires configs.yaml vjepa.require_official=true.")
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
    print(f"official_vjepa: loaded {vjepa_cfg['model_name']} pretrained={bool(vjepa_cfg.get('pretrained', True))}")
    return encoder


def run_verify(args: argparse.Namespace) -> int:
    assert_python_310()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 43)))
    device = select_device(config)
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"device: {device}")

    vjepa_cfg = config["vjepa"]
    encoder = build_encoder(config, args.require_official_vjepa, device)

    predictor = ActionConditionedPredictor(predictor_config_from_dict(config["predictor"])).to(device).eval()
    env = make_smoke_env(config["environment"])
    obs = env.reset()
    if obs.shape != (384, 384, 3) or obs.dtype != np.uint8:
        raise RuntimeError(f"Unexpected smoke env observation: {obs.shape}, {obs.dtype}")

    state_before = env.physics.get_state().copy()
    frames_before = [frame.copy() for frame in env.frame_buffer]
    env.push_state()
    env.step(env.sample_action(0.2))
    env.pop_state()
    if not np.allclose(env.physics.get_state(), state_before):
        raise RuntimeError("Rollback failed to restore physics state.")
    if any(not np.array_equal(a, b) for a, b in zip(env.frame_buffer, frames_before)):
        raise RuntimeError("Rollback failed to restore frame buffer.")
    print("rollback: ok")

    with torch.no_grad():
        frames = env.get_frame_tensor(str(device))
        z_ctx = encoder(frames)
        expected_z = (
            1,
            env.frame_buffer_len,
            int(vjepa_cfg["patches_per_frame"]),
            int(vjepa_cfg["embed_dim"]),
        )
        if tuple(z_ctx.shape) != expected_z:
            raise RuntimeError(f"Unexpected V-JEPA features: {tuple(z_ctx.shape)}")
        print(f"encoder_output: {tuple(z_ctx.shape)}")

        t_ctx = z_ctx.shape[1]
        actions = torch.zeros(1, t_ctx, 7, device=device)
        states = torch.from_numpy(env.get_state_vector()).to(device).reshape(1, 1, 7).expand(1, t_ctx, 7)
        interleaved = predictor.interleave_tokens(z_ctx, actions, states)
        expected_tokens = t_ctx * TOKENS_PER_FRAME
        if interleaved.shape[1] != expected_tokens:
            raise RuntimeError(f"Interleaved length mismatch: {interleaved.shape[1]} vs {expected_tokens}")
        mask = predictor.build_block_causal_mask(t_ctx, device=device)
        if mask[0, TOKENS_PER_FRAME].item() is not True:
            raise RuntimeError("Attention mask allows a first-frame token to attend to a future frame.")
        if mask[TOKENS_PER_FRAME, 0].item() is not False:
            raise RuntimeError("Attention mask blocks attending to a past frame.")
        if mask[0, 1].item() is not False:
            raise RuntimeError("Attention mask blocks same-frame attention.")
        pred = predictor(z_ctx, actions, states)
        print(f"predictor_output: {tuple(pred.shape)}")

        goal_z = z_ctx[:, -1].clone()
        planner_cfg_data = dict(config["planner"])
        verify_cfg = config.get("verify", {})
        planner_cfg_data.update(
            {
                "horizon": int(verify_cfg.get("planner_horizon", planner_cfg_data["horizon"])),
                "n_candidates": int(verify_cfg.get("planner_candidates", planner_cfg_data["n_candidates"])),
                "n_elites": int(verify_cfg.get("planner_elites", planner_cfg_data["n_elites"])),
                "n_iters": int(verify_cfg.get("planner_iters", planner_cfg_data["n_iters"])),
                "topk_physics": int(verify_cfg.get("planner_topk_physics", planner_cfg_data["topk_physics"])),
            }
        )
        planner = LatentCEMPlanner(
            predictor=predictor,
            encoder=encoder,
            config=cem_config_from_dict(planner_cfg_data),
            device=device,
        )
        top_actions, latent_costs = planner.plan(z_ctx, goal_z, states[:, -1])
        best_seq, physics_cost = planner.verify_with_physics(env, top_actions, goal_z)
        print(f"planner: latent_cost={float(latent_costs[0].cpu()):.6f}, physics_cost={physics_cost:.6f}")
        print(f"planner_first_action: {best_seq[0].numpy().round(4).tolist()}")

        train_frames = frames[:, -2:].clone()
        train_actions = torch.zeros(1, 1, 7, device=device)
        train_states = torch.from_numpy(env.get_state_vector()).to(device).reshape(1, 1, 7)
        loss, metrics = compute_ac_predictor_loss(
            encoder=encoder,
            predictor=predictor,
            frames=train_frames,
            actions=train_actions,
            states=train_states,
            rollout_steps=1,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Trainer smoke loss is not finite: {metrics}")
        print(f"trainer_loss: {metrics}")

    print("verify: ok")
    return 0


def run_oracle_demo(args: argparse.Namespace) -> int:
    assert_python_310()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 43)))
    device = select_device(config)
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"device: {device}")
    encoder = build_encoder(config, args.require_official_vjepa, device)

    demo_cfg = dict(config.get("oracle_demo", {}))
    mpc_steps = _arg_or_config(args.mpc_steps, demo_cfg, "mpc_steps", 20)
    target_steps = _arg_or_config(args.target_steps, demo_cfg, "target_steps", 80)
    video_fps = _arg_or_config(args.video_fps, demo_cfg, "video_fps", 12)
    output_dir = Path(args.output_dir or demo_cfg.get("output_dir", "runs/oracle_demo/latest"))
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    oracle_cfg = oracle_cem_config_from_dict(
        {
            "horizon": _arg_or_config(args.horizon, demo_cfg, "horizon", 8),
            "n_candidates": _arg_or_config(args.candidates, demo_cfg, "n_candidates", 32),
            "n_elites": _arg_or_config(args.elites, demo_cfg, "n_elites", 8),
            "n_iters": _arg_or_config(args.iters, demo_cfg, "n_iters", 3),
            "action_dims": int(demo_cfg.get("action_dims", 2)),
            "action_repeat": _arg_or_config(args.action_repeat, demo_cfg, "action_repeat", 5),
            "distance_cost_weight": float(demo_cfg.get("distance_cost_weight", 1.0)),
            "use_action_priors": bool(demo_cfg.get("use_action_priors", False)),
            "encode_batch_size": _arg_or_config(args.encode_batch_size, demo_cfg, "encode_batch_size", 4),
            "topk_visualize": _arg_or_config(args.topk_visualize, demo_cfg, "topk_visualize", 4),
            "seed": int(config.get("seed", 43)),
        }
    )
    print(f"oracle_cem: {oracle_cfg}")

    env = make_smoke_env(config["environment"])
    env.reset()
    initial_signature = env.snapshot_signature()
    initial_x = env.get_agent_x()
    initial_y = env.get_agent_y()
    initial_reward = env.compute_reward()
    initial_distance = env.distance_to_goal()

    goal_z, target_frame, target_summary = build_goal_latent(env, encoder, device, target_steps)
    if env.snapshot_signature() != initial_signature:
        raise RuntimeError("Goal latent construction failed to restore the initial environment state.")
    print(
        "target: "
        f"x={target_summary['x']:.4f}, y={target_summary['y']:.4f}, reward={target_summary['reward']:.4f}, "
        f"distance={target_summary['distance_to_goal']:.4f}"
    )

    planner = OracleCEMPlanner(encoder=encoder, config=oracle_cfg, device=device)
    diagnostic = evaluate_direct_vs_detour(env, planner, goal_z, oracle_cfg.horizon)
    print(
        "diagnostic latent-only costs: "
        f"direct={diagnostic['direct_cost']:.6f}, detour={diagnostic['detour_cost']:.6f}; "
        f"direct_xy=({diagnostic['direct_x']:.3f},{diagnostic['direct_y']:.3f}), "
        f"detour_xy=({diagnostic['detour_x']:.3f},{diagnostic['detour_y']:.3f})"
    )
    metrics: list[dict[str, float]] = []
    execution_frames = [env.render().copy()]
    candidate_grid_rollouts: list[list[np.ndarray]] | None = None

    if args.execute_full_plan:
        before_plan = env.snapshot_signature()
        result = planner.plan(env, goal_z)
        if env.snapshot_signature() != before_plan:
            raise RuntimeError("Oracle planning changed the live environment state.")
        candidate_grid_rollouts = collect_candidate_rollouts(env, result.top_sequences, oracle_cfg.action_repeat)
        execution_actions = list(result.best_sequence)
    else:
        execution_actions = []

    for step_idx in range(mpc_steps):
        before_plan = env.snapshot_signature()
        if args.execute_full_plan:
            if step_idx >= len(execution_actions):
                break
            action = execution_actions[step_idx]
            result = None
        else:
            result = planner.plan(env, goal_z)
            if env.snapshot_signature() != before_plan:
                raise RuntimeError("Oracle planning changed the live environment state.")
            if candidate_grid_rollouts is None:
                candidate_grid_rollouts = collect_candidate_rollouts(env, result.top_sequences, oracle_cfg.action_repeat)
            action = result.best_sequence[0]
        reward = env.compute_reward()
        done = False
        for _ in range(oracle_cfg.action_repeat):
            frame, reward, done, _ = env.step(action)
            execution_frames.append(frame.copy())
        with torch.no_grad():
            current_z = encoder(env.get_frame_tensor(str(device)))[:, -1]
            latent_cost = float(mean_patch_cost(current_z, goal_z.to(device)).item())
        row = {
            "step": float(step_idx),
            "chosen_action0": float(action[0]),
            "chosen_action1": float(action[1]),
            "x": env.get_agent_x(),
            "y": env.get_agent_y(),
            "reward": float(reward),
            "distance_to_goal": env.distance_to_goal(),
            "latent_cost": latent_cost,
            "planned_terminal_x": result.best_x if result is not None else float("nan"),
            "planned_terminal_y": result.best_y if result is not None else float("nan"),
            "planned_terminal_reward": result.best_reward if result is not None else float("nan"),
            "planned_terminal_cost": result.best_cost if result is not None else float("nan"),
        }
        metrics.append(row)
        print(
            f"step={step_idx:02d} action=({row['chosen_action0']:+.3f},{row['chosen_action1']:+.3f}) "
            f"xy=({row['x']:.4f},{row['y']:.4f}) reward={row['reward']:.4f} "
            f"latent_cost={latent_cost:.6f}"
        )
        if done:
            break

    final_x = env.get_agent_x()
    final_y = env.get_agent_y()
    final_reward = env.compute_reward()
    final_distance = env.distance_to_goal()
    if env.stack_depth != 0:
        raise RuntimeError("Oracle demo leaked rollback stack state.")
    success = final_distance < initial_distance
    if not success:
        print(
            "warning: latent-only run did not move closer to the goal: "
            f"initial_xy=({initial_x:.4f},{initial_y:.4f}), "
            f"final_xy=({final_x:.4f},{final_y:.4f}), "
            f"goal_xy=({env.goal_x:.4f},{env.goal_y:.4f})"
        )

    execution_video = output_dir / "execution.mp4"
    imageio.mimsave(execution_video, execution_frames, fps=video_fps, codec="libx264", macro_block_size=16)
    if candidate_grid_rollouts:
        imageio.mimsave(
            output_dir / "candidate_grid.mp4",
            make_candidate_grid_frames(candidate_grid_rollouts),
            fps=video_fps,
            codec="libx264",
            macro_block_size=16,
        )
    else:
        imageio.mimsave(
            output_dir / "candidate_grid.mp4",
            [target_frame],
            fps=video_fps,
            codec="libx264",
            macro_block_size=16,
        )
    write_metrics_csv(output_dir / "metrics.csv", metrics)
    write_summary_html(
        output_dir / "summary.html",
        metrics=metrics,
        initial_x=initial_x,
        initial_reward=initial_reward,
        final_x=final_x,
        final_reward=final_reward,
        goal_x=env.goal_x,
        target_summary=target_summary,
    )

    print(f"initial: x={initial_x:.4f}, y={initial_y:.4f}, reward={initial_reward:.4f}")
    print(f"final: x={final_x:.4f}, y={final_y:.4f}, reward={final_reward:.4f}")
    print(f"outputs: {output_dir}")
    print(f"oracle-demo: {'ok' if success else 'completed-without-success'}")
    return 0


@torch.no_grad()
def build_goal_latent(env, encoder, device: torch.device, target_steps: int) -> tuple[torch.Tensor, np.ndarray, dict]:
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
    target_steps = max(target_steps, 12)
    phases = [
        (0.45, np.array([1.0, 0.0], dtype=np.float32)),
        (0.35, np.array([0.0, 1.0], dtype=np.float32)),
        (0.20, np.array([-0.25, 0.2], dtype=np.float32)),
    ]
    actions: list[np.ndarray] = []
    allocated = 0
    for idx, (fraction, xy_action) in enumerate(phases):
        count = int(round(target_steps * fraction)) if idx < len(phases) - 1 else target_steps - allocated
        allocated += count
        for _ in range(max(0, count)):
            action = np.zeros(7, dtype=np.float32)
            action[:2] = xy_action
            actions.append(action)
    return np.stack(actions, axis=0)


@torch.no_grad()
def evaluate_direct_vs_detour(env, planner: OracleCEMPlanner, goal_z: torch.Tensor, horizon: int) -> dict[str, float]:
    direct = np.zeros((1, horizon, 7), dtype=np.float32)
    direct[:, :, 0] = 1.0
    direct[:, :, 1] = 0.0
    detour = np.zeros((1, horizon, 7), dtype=np.float32)
    half = max(1, horizon // 2)
    detour[:, :half, 0] = 1.0
    detour[:, :half, 1] = 0.0
    detour[:, half:, 0] = 0.0
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


def collect_candidate_rollouts(env, sequences: np.ndarray, action_repeat: int) -> list[list[np.ndarray]]:
    rollouts: list[list[np.ndarray]] = []
    before = env.snapshot_signature()
    for sequence in sequences:
        env.push_state()
        try:
            repeated = np.repeat(sequence, action_repeat, axis=0) if action_repeat > 1 else sequence
            result = env.rollout_sequence(repeated, record_frames=True)
            rollouts.append(result["frames"])
        finally:
            env.pop_state()
        if env.snapshot_signature() != before:
            raise RuntimeError("Candidate visualization rollout changed the live environment state.")
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
        tiles = []
        for rollout in rollouts:
            tiles.append(rollout[min(frame_idx, len(rollout) - 1)])
        while len(tiles) < rows * cols:
            tiles.append(blank)
        row_imgs = []
        for row in range(rows):
            start = row * cols
            row_imgs.append(np.concatenate(tiles[start : start + cols], axis=1))
        grid_frames.append(np.concatenate(row_imgs, axis=0))
    return grid_frames


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


def _arg_or_config(value, config: dict, key: str, default):
    return value if value is not None else config.get(key, default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VJEPA-Gym six-file prototype.")
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify", help="Run official-model smoke verification.")
    verify.add_argument("--config", default="configs.yaml")
    verify.add_argument("--require-official-vjepa", action="store_true")
    verify.set_defaults(func=run_verify)
    oracle = sub.add_parser("oracle-demo", help="Run simulator-in-the-loop oracle CEM demo.")
    oracle.add_argument("--config", default="configs.yaml")
    oracle.add_argument("--require-official-vjepa", action="store_true")
    oracle.add_argument("--mpc-steps", type=int)
    oracle.add_argument("--horizon", type=int)
    oracle.add_argument("--candidates", type=int)
    oracle.add_argument("--elites", type=int)
    oracle.add_argument("--iters", type=int)
    oracle.add_argument("--action-repeat", type=int)
    oracle.add_argument("--target-steps", type=int)
    oracle.add_argument("--encode-batch-size", type=int)
    oracle.add_argument("--topk-visualize", type=int)
    oracle.add_argument("--video-fps", type=int)
    oracle.add_argument("--output-dir")
    oracle.add_argument("--execute-full-plan", action="store_true")
    oracle.set_defaults(func=run_oracle_demo)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
