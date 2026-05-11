from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


from core_models import ActionConditionedPredictor, TOKENS_PER_FRAME, mean_patch_cost
from environment import make_smoke_env
from latent_planner import LatentCEMPlanner, OracleCEMPlanner, oracle_cem_config_from_dict, cem_config_from_dict
from trainer_engine import HDF5WindowDataset, compute_ac_predictor_latent_loss, compute_ac_predictor_loss
from utils import (
    setup_run,
    build_encoder,
    predictor_config_from_dict,
    frames_to_tensor,
    arg_or_config,
    build_goal_latent,
    evaluate_direct_vs_detour,
    collect_candidate_rollouts,
    make_candidate_grid_frames,
    save_run_videos,
    write_metrics_csv,
    write_summary_html,
    probe_safe_memory_params,
)




# ==============================================================================
# [SECTION 1] EVALUATION & DEMO COMMANDS
# 包含：运行验证 (verify)、完美模拟器基准测试 (oracle-demo)
# ==============================================================================
def run_verify(args: argparse.Namespace) -> int:
    config, device, _ = setup_run(args)
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
    config, device, _ = setup_run(args)
    encoder = build_encoder(config, args.require_official_vjepa, device)

    demo_cfg = dict(config.get("oracle_demo", {}))
    mpc_steps = arg_or_config(args.mpc_steps, demo_cfg, "mpc_steps", 20)
    target_steps = arg_or_config(args.target_steps, demo_cfg, "target_steps", 80)
    video_fps = arg_or_config(args.video_fps, demo_cfg, "video_fps", 12)
    output_dir = Path(args.output_dir or demo_cfg.get("output_dir", "runs/oracle_demo/latest"))
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    oracle_cfg = oracle_cem_config_from_dict(
        {
            "horizon": arg_or_config(args.horizon, demo_cfg, "horizon", 8),
            "n_candidates": arg_or_config(args.candidates, demo_cfg, "n_candidates", 32),
            "n_elites": arg_or_config(args.elites, demo_cfg, "n_elites", 8),
            "n_iters": arg_or_config(args.iters, demo_cfg, "n_iters", 3),
            "action_dims": int(demo_cfg.get("action_dims", 2)),
            "action_repeat": arg_or_config(args.action_repeat, demo_cfg, "action_repeat", 5),
            "distance_cost_weight": float(demo_cfg.get("distance_cost_weight", 1.0)),
            "use_action_priors": bool(demo_cfg.get("use_action_priors", False)),
            "encode_batch_size": arg_or_config(args.encode_batch_size, demo_cfg, "encode_batch_size", 4),
            "topk_visualize": arg_or_config(args.topk_visualize, demo_cfg, "topk_visualize", 4),
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

    save_run_videos(output_dir, execution_frames, candidate_grid_rollouts, target_frame, video_fps)
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


# ==============================================================================
# [SECTION 2] DATA COLLECTION COMMANDS
# 包含：收集 HDF5 离线数据集 (collect-synthetic)、收集潜在分片数据 (collect-latent-shards)
# ==============================================================================
def run_collect_synthetic(args: argparse.Namespace) -> int:
    config, device, seed = setup_run(args)
    rng = np.random.default_rng(seed)

    encoder = build_encoder(config, args.require_official_vjepa, device)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    demo_cfg = dict(config.get("oracle_demo", {}))
    oracle_cfg = oracle_cem_config_from_dict(
        {
            "horizon": max(args.sequence_len, int(args.oracle_horizon or demo_cfg.get("horizon", 8))),
            "n_candidates": int(args.oracle_candidates or demo_cfg.get("n_candidates", 32)),
            "n_elites": int(args.oracle_elites or demo_cfg.get("n_elites", 8)),
            "n_iters": int(args.oracle_iters or demo_cfg.get("n_iters", 3)),
            "action_dims": 2,
            "action_repeat": int(args.action_repeat),
            "distance_cost_weight": float(demo_cfg.get("distance_cost_weight", 0.0)),
            "use_action_priors": bool(demo_cfg.get("use_action_priors", False)),
            "encode_batch_size": int(args.encode_batch_size or demo_cfg.get("encode_batch_size", 4)),
            "topk_visualize": 1,
            "seed": int(args.seed if args.seed is not None else config.get("seed", 43)),
        }
    )
    planner = OracleCEMPlanner(encoder=encoder, config=oracle_cfg, device=device)
    env = make_smoke_env(config["environment"])

    with h5py.File(output, "w") as h5:
        h5.attrs["seed"] = int(args.seed if args.seed is not None else config.get("seed", 43))
        h5.attrs["sequence_len"] = int(args.sequence_len)
        h5.attrs["action_repeat"] = int(args.action_repeat)
        h5.attrs["source_ratios"] = "oracle=0.60,directional=0.25,random=0.15"
        h5.attrs["oracle_config"] = repr(oracle_cfg)
        _collect_synthetic_split(
            h5=h5,
            split="train",
            windows=int(args.train_windows),
            env=env,
            planner=planner,
            encoder=encoder,
            device=device,
            rng=rng,
            sequence_len=int(args.sequence_len),
            action_repeat=int(args.action_repeat),
            target_steps=int(args.target_steps),
        )
        _collect_synthetic_split(
            h5=h5,
            split="val",
            windows=int(args.val_windows),
            env=env,
            planner=planner,
            encoder=encoder,
            device=device,
            rng=rng,
            sequence_len=int(args.sequence_len),
            action_repeat=int(args.action_repeat),
            target_steps=int(args.target_steps),
        )

    print(f"synthetic dataset: {output}")
    print("collect-synthetic: ok")
    return 0


def run_collect_latent_shards(args: argparse.Namespace) -> int:
    config, device, seed = setup_run(args)
    rng = np.random.default_rng(seed)

    encoder = build_encoder(config, args.require_official_vjepa, device).eval()
    for param in encoder.parameters():
        param.requires_grad_(False)

    # 根据当前可用显存/内存自动压低 encode_batch_size，防止编码阶段 OOM。
    _mem = probe_safe_memory_params(
        device=device,
        batch_size=1,
        grad_accum=1,
        encode_batch_size=int(args.encode_batch_size),
    )
    args.encode_batch_size = _mem["encode_batch_size"]

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    demo_cfg = dict(config.get("oracle_demo", {}))
    oracle_cfg = oracle_cem_config_from_dict(
        {
            "horizon": max(int(args.sequence_len), int(args.oracle_horizon or demo_cfg.get("horizon", 8))),
            "n_candidates": int(args.oracle_candidates),
            "n_elites": int(args.oracle_elites),
            "n_iters": int(args.oracle_iters),
            "action_dims": 2,
            "action_repeat": int(args.action_repeat),
            "distance_cost_weight": float(demo_cfg.get("distance_cost_weight", 0.0)),
            "use_action_priors": bool(demo_cfg.get("use_action_priors", False)),
            "encode_batch_size": int(args.encode_batch_size),
            "topk_visualize": 1,
            "seed": seed,
        }
    )
    planner = OracleCEMPlanner(encoder=encoder, config=oracle_cfg, device=device)
    env = make_smoke_env(config["environment"])

    _collect_latent_split(
        split="train",
        windows=int(args.train_windows),
        out_dir=out_dir,
        env=env,
        planner=planner,
        encoder=encoder,
        device=device,
        rng=rng,
        sequence_len=int(args.sequence_len),
        action_repeat=int(args.action_repeat),
        target_steps=int(args.target_steps),
        shard_size=int(args.shard_size),
        encode_batch_size=int(args.encode_batch_size),
        amp=bool(args.amp),
        resume=bool(args.resume),
        seed=seed,
        oracle_cfg=oracle_cfg,
    )
    _collect_latent_split(
        split="val",
        windows=int(args.val_windows),
        out_dir=out_dir,
        env=env,
        planner=planner,
        encoder=encoder,
        device=device,
        rng=rng,
        sequence_len=int(args.sequence_len),
        action_repeat=int(args.action_repeat),
        target_steps=int(args.target_steps),
        shard_size=int(args.shard_size),
        encode_batch_size=int(args.encode_batch_size),
        amp=bool(args.amp),
        resume=bool(args.resume),
        seed=seed,
        oracle_cfg=oracle_cfg,
    )

    print(f"latent shards: {out_dir}")
    print("collect-latent-shards: ok")
    return 0


# ==============================================================================
# [SECTION 3] TRAINING & PIPELINE COMMANDS
# 包含：训练潜在预测器 (train-ac-latent)、端到端流水线调度 (pipeline-latent)
# ==============================================================================
def run_train_ac_latent(args: argparse.Namespace) -> int:
    config, device, seed = setup_run(args)
    rng = np.random.default_rng(seed)

    # 根据当前可用显存/内存自动压低训练参数，防止 OOM。
    # 一次性写回 args，后续所有引用均自动生效。
    _mem = probe_safe_memory_params(
        device=device,
        batch_size=int(args.batch_size),
        grad_accum=int(args.grad_accum),
    )
    args.batch_size = _mem["batch_size"]
    args.grad_accum = _mem["grad_accum"]

    # 初始化 W&B（在子进程中必须显式调用，不会继承 notebook 的 run）。
    _wandb_project = getattr(args, "wandb_project", None)
    if HAS_WANDB and _wandb_project:
        wandb.init(
            project=_wandb_project,
            name=getattr(args, "wandb_run_name", None) or "train-ac-latent",
            config={
                "batch_size":   int(args.batch_size),
                "grad_accum":   int(args.grad_accum),
                "lr":           float(args.lr),
                "weight_decay": float(args.weight_decay),
                "rollout_steps":int(args.rollout_steps),
                "amp":          bool(args.amp),
            },
            resume="allow",
        )
        print(f"[wandb] run 已初始化，project={_wandb_project}")

    predictor = ActionConditionedPredictor(predictor_config_from_dict(config["predictor"])).to(device).train()
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pt"
    metrics_path = output_dir / "latent_metrics.csv"
    history: list[dict[str, float]] = []
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        predictor.load_state_dict(checkpoint["predictor"])
        if "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except ValueError as exc:
                print(f"warning: could not restore optimizer state from {args.resume_checkpoint}: {exc}")
        history = list(checkpoint.get("metrics", []))
        print(f"resumed predictor checkpoint: {Path(args.resume_checkpoint).resolve()}")
        del checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "phase",
                "index",
                "shards",
                "loss",
                "loss_tf",
                "loss_roll",
                "loss_delta",
                "loss_delta_roll",
                "loss_action_contrast",
            ],
        )
        writer.writeheader()

        if args.watch:
            processed: set[Path] = set()
            expected_train = int(args.expected_train_shards or 0)
            expected_val = int(args.expected_val_shards or 0)
            while True:
                train_shards = _list_complete_latent_shards(data_dir, "train")
                new_shards = [path for path in train_shards if path not in processed]
                for shard in new_shards:
                    metrics = _train_latent_shard(
                        predictor=predictor,
                        optimizer=optimizer,
                        scaler=scaler,
                        shard=shard,
                        device=device,
                        batch_size=int(args.batch_size),
                        grad_accum=int(args.grad_accum),
                        rollout_steps=int(args.rollout_steps),
                        use_amp=use_amp,
                        rng=rng,
                        condition_on_state=bool(args.condition_on_state),
                    )
                    processed.add(shard)
                    row = {
                        "phase": "watch",
                        "index": float(len(processed)),
                        "shards": float(len(train_shards)),
                        "loss": metrics["loss_total"],
                        "loss_tf": metrics["loss_tf"],
                        "loss_roll": metrics["loss_roll"],
                        "loss_delta": metrics["loss_delta"],
                        "loss_delta_roll": metrics["loss_delta_roll"],
                        "loss_action_contrast": metrics["loss_action_contrast"],
                    }
                    writer.writerow(row)
                    handle.flush()
                    history.append(row)
                    if HAS_WANDB and wandb.run is not None:
                        wandb.log({f"watch_train/{k}": v for k, v in row.items() if isinstance(v, (int, float))})
                    _save_latent_checkpoint(latest_path, predictor, optimizer, config, args, history)
                    print(
                        f"watch shard={shard.name} loss={row['loss']:.6f} "
                        f"seen={len(processed)}/{expected_train or '?'}"
                    )

                train_done = expected_train > 0 and len(train_shards) >= expected_train and len(processed) >= expected_train
                val_done = expected_val <= 0 or len(_list_complete_latent_shards(data_dir, "val")) >= expected_val
                if train_done and val_done:
                    break
                if not new_shards and args.replay_while_waiting and train_shards:
                    shard = Path(rng.choice(train_shards))
                    metrics = _train_latent_shard(
                        predictor=predictor,
                        optimizer=optimizer,
                        scaler=scaler,
                        shard=shard,
                        device=device,
                        batch_size=int(args.batch_size),
                        grad_accum=int(args.grad_accum),
                        rollout_steps=int(args.rollout_steps),
                        use_amp=use_amp,
                        rng=rng,
                        condition_on_state=bool(args.condition_on_state),
                    )
                    row = {
                        "phase": "replay",
                        "index": float(len(history) + 1),
                        "shards": float(len(train_shards)),
                        "loss": metrics["loss_total"],
                        "loss_tf": metrics["loss_tf"],
                        "loss_roll": metrics["loss_roll"],
                        "loss_delta": metrics["loss_delta"],
                        "loss_delta_roll": metrics["loss_delta_roll"],
                        "loss_action_contrast": metrics["loss_action_contrast"],
                    }
                    writer.writerow(row)
                    handle.flush()
                    history.append(row)
                    _save_latent_checkpoint(latest_path, predictor, optimizer, config, args, history)
                    print(f"replay shard={shard.name} loss={row['loss']:.6f} available={len(train_shards)}")
                    continue
                time.sleep(float(args.poll_seconds))

        for epoch in range(1, int(args.epochs) + 1):
            train_shards = _list_complete_latent_shards(data_dir, "train")
            if not train_shards:
                raise RuntimeError(f"No complete train shards found in {data_dir}.")
            rng.shuffle(train_shards)
            totals = {
                "loss_tf": 0.0,
                "loss_roll": 0.0,
                "loss_delta": 0.0,
                "loss_delta_roll": 0.0,
                "loss_action_contrast": 0.0,
                "loss_total": 0.0,
            }
            for shard in train_shards:
                metrics = _train_latent_shard(
                    predictor=predictor,
                    optimizer=optimizer,
                    scaler=scaler,
                    shard=shard,
                    device=device,
                    batch_size=int(args.batch_size),
                    grad_accum=int(args.grad_accum),
                    rollout_steps=int(args.rollout_steps),
                    use_amp=use_amp,
                    rng=rng,
                    condition_on_state=bool(args.condition_on_state),
                )
                for key in totals:
                    totals[key] += metrics[key]
            train_metrics = {key: value / len(train_shards) for key, value in totals.items()}
            val_shards = _list_complete_latent_shards(data_dir, "val")
            val_metrics = _evaluate_latent_shards(
                predictor=predictor,
                shards=val_shards,
                device=device,
                batch_size=int(args.batch_size),
                rollout_steps=int(args.rollout_steps),
                use_amp=use_amp,
                condition_on_state=bool(args.condition_on_state),
            ) if val_shards else {"loss_tf": float("nan"), "loss_roll": float("nan"), "loss_total": float("nan")}
            row = {
                "phase": "epoch",
                "index": float(epoch),
                "shards": float(len(train_shards)),
                "loss": train_metrics["loss_total"],
                "loss_tf": train_metrics["loss_tf"],
                "loss_roll": train_metrics["loss_roll"],
                "loss_delta": train_metrics["loss_delta"],
                "loss_delta_roll": train_metrics["loss_delta_roll"],
                "loss_action_contrast": train_metrics["loss_action_contrast"],
            }
            writer.writerow(row)
            val_row = {
                "phase": "val",
                "index": float(epoch),
                "shards": float(len(val_shards)),
                "loss": val_metrics["loss_total"],
                "loss_tf": val_metrics["loss_tf"],
                "loss_roll": val_metrics["loss_roll"],
                "loss_delta": val_metrics["loss_delta"],
                "loss_delta_roll": val_metrics["loss_delta_roll"],
                "loss_action_contrast": val_metrics["loss_action_contrast"],
            }
            writer.writerow(val_row)
            handle.flush()
            if HAS_WANDB and wandb.run is not None:
                wandb.log({f"epoch_train/{k}": v for k, v in row.items() if isinstance(v, (int, float))} | {f"epoch_val/{k}": v for k, v in val_row.items() if isinstance(v, (int, float))}, step=epoch)
            history.extend([row, {"phase": "val", "index": float(epoch), **val_metrics}])
            _save_latent_checkpoint(latest_path, predictor, optimizer, config, args, history)
            print(
                f"epoch={epoch:03d} train={train_metrics['loss_total']:.6f} "
                f"val={val_metrics['loss_total']:.6f} shards={len(train_shards)}"
            )

    print(f"checkpoint: {latest_path}")
    print("train-ac-latent: ok")
    if HAS_WANDB and wandb.run is not None:
        wandb.finish()
    return 0


def _stream_subprocess(
    proc: subprocess.Popen,
    log_path: Path,
    prefix: str,
) -> threading.Thread:
    """将子进程 stdout 实时同步写入日志文件 + 当前控制台。

    解决 Colab 中 pipeline-latent 没有任何输出的问题：原先子进程
    stdout 被完全重定向到文件，用户看不到任何内容。
    """
    log_file = log_path.open("w", encoding="utf-8", buffering=1)

    def _reader() -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
                print(f"[{prefix}] {line}", end="", flush=True)
        finally:
            log_file.close()

    t = threading.Thread(target=_reader, daemon=True, name=f"{prefix}-reader")
    t.start()
    return t


def run_pipeline_latent(args: argparse.Namespace) -> int:
    root = Path.cwd().resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    train_shards_expected = math.ceil(int(args.train_windows) / int(args.shard_size))
    val_shards_expected   = math.ceil(int(args.val_windows)   / int(args.shard_size))

    base = [sys.executable, str(Path(__file__).resolve())]
    collector_cmd = base + [
        "collect-latent-shards",
        "--require-official-vjepa",
        "--output-dir",        str(data_dir),
        "--train-windows",     str(args.train_windows),
        "--val-windows",       str(args.val_windows),
        "--shard-size",        str(args.shard_size),
        "--oracle-candidates", str(args.oracle_candidates),
        "--oracle-elites",     str(args.oracle_elites),
        "--oracle-iters",      str(args.oracle_iters),
        "--encode-batch-size", str(args.encode_batch_size),
        "--sequence-len",      str(args.sequence_len),
    ]
    trainer_cmd = base + [
        "train-ac-latent",
        "--data-dir",              str(data_dir),
        "--output-dir",            str(Path(args.output_dir).resolve()),
        "--watch",
        "--expected-train-shards", str(train_shards_expected),
        "--expected-val-shards",   str(val_shards_expected),
        "--replay-while-waiting",
        "--epochs",                str(args.final_epochs),
        "--batch-size",            str(args.batch_size),
        "--grad-accum",            str(args.grad_accum),
        "--rollout-steps",         str(args.rollout_steps),
    ]
    if bool(args.condition_on_state):
        trainer_cmd.append("--condition-on-state")
    # 将 W&B 项目名透传给两个子进程，各自独立初始化自己的 wandb run。
    _wandb_project = getattr(args, "wandb_project", None)
    if _wandb_project:
        collector_cmd += ["--wandb-project", _wandb_project, "--wandb-run-name", "collector"]
        trainer_cmd   += ["--wandb-project", _wandb_project, "--wandb-run-name", "trainer"]

    print(f"[pipeline] 预期分片：train={train_shards_expected}  val={val_shards_expected}")
    collector_log_path = run_dir / "collector.log"
    trainer_log_path   = run_dir / "trainer.log"
    print(f"[pipeline] 日志：{collector_log_path}  /  {trainer_log_path}")

    # 用 PIPE 捕获子进程 stdout，由 _stream_subprocess 线程实时转发到控制台。
    collector = subprocess.Popen(
        collector_cmd, cwd=root,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    trainer = subprocess.Popen(
        trainer_cmd, cwd=root,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    (run_dir / "collector.pid").write_text(str(collector.pid), encoding="utf-8")
    (run_dir / "trainer.pid").write_text(str(trainer.pid),   encoding="utf-8")
    collector_thread = _stream_subprocess(collector, collector_log_path, "collector")
    trainer_thread   = _stream_subprocess(trainer,   trainer_log_path,   "trainer")

    try:
        while True:
            collector_code = collector.poll()
            trainer_code   = trainer.poll()
            if collector_code is not None and collector_code != 0:
                trainer.terminate()
                raise RuntimeError(
                    f"collector 失败，exit={collector_code}；"
                    f"详情见 {collector_log_path}"
                )
            if trainer_code is not None and trainer_code != 0:
                collector.terminate()
                raise RuntimeError(
                    f"trainer 失败，exit={trainer_code}；"
                    f"详情见 {trainer_log_path}"
                )
            if collector_code == 0 and trainer_code == 0:
                break
            # 每 10s 打印一次整体状态（分片数 + GPU 全局占用）。
            # 注意：torch.cuda.memory_allocated() 只显示当前（父）进程的显存，
            # 始终为 0。改用 nvidia-smi 获取所有进程的实际 GPU 占用。
            n_train = len(list(data_dir.glob("train-*.h5")))
            n_val   = len(list(data_dir.glob("val-*.h5")))
            try:
                _smi = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True, text=True, timeout=3,
                )
                if _smi.returncode == 0:
                    _parts = [p.strip() for p in _smi.stdout.strip().split(",")]
                    gpu_str = (
                        f"GPU {_parts[0]}MB/{_parts[1]}MB "
                        f"util={_parts[2]}%"
                    )
                else:
                    gpu_str = "GPU=unavailable"
            except Exception:
                gpu_str = "GPU=unavailable"
            # 子进程存活标志
            c_alive = "✓" if collector.poll() is None else "✗"
            t_alive = "✓" if trainer.poll()   is None else "✗"
            print(
                f"[pipeline] shards train={n_train}/{train_shards_expected} "
                f"val={n_val}/{val_shards_expected}  {gpu_str}  "
                f"collector={c_alive} trainer={t_alive}",
                flush=True,
            )
            time.sleep(10.0)
    finally:
        collector_thread.join(timeout=5.0)
        trainer_thread.join(timeout=5.0)
    print("pipeline-latent: ok")
    return 0


def run_train_ac(args: argparse.Namespace) -> int:
    config, device, _ = setup_run(args)

    encoder = build_encoder(config, args.require_official_vjepa, device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)

    # 根据当前可用显存/内存自动调节训练参数。
    _mem = probe_safe_memory_params(
        device=device,
        batch_size=int(args.batch_size),
        grad_accum=int(args.grad_accum),
        num_workers=min(4, os.cpu_count() or 1),
    )
    args.batch_size = _mem["batch_size"]
    args.grad_accum = _mem["grad_accum"]
    _num_workers = _mem["num_workers"]

    predictor = ActionConditionedPredictor(predictor_config_from_dict(config["predictor"])).to(device).train()
    train_loader = DataLoader(
        HDF5WindowDataset(args.dataset, "train"),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=_num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=2 if _num_workers > 0 else None,
        persistent_workers=_num_workers > 0,
    )
    val_loader = DataLoader(
        HDF5WindowDataset(args.dataset, "val"),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=_num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=2 if _num_workers > 0 else None,
        persistent_workers=_num_workers > 0,
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pt"
    metrics_path = output_dir / "metrics.csv"
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float]] = []

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "train_loss_tf", "train_loss_roll", "val_loss", "val_loss_tf", "val_loss_roll"],
        )
        writer.writeheader()

        for epoch in range(1, int(args.epochs) + 1):
            start = time.time()
            train_metrics = _run_ac_epoch(
                encoder=encoder,
                predictor=predictor,
                loader=train_loader,
                device=device,
                rollout_steps=int(args.rollout_steps),
                optimizer=optimizer,
                scaler=scaler,
                grad_accum=int(args.grad_accum),
                use_amp=use_amp,
            )
            val_metrics = _evaluate_ac_loss(
                encoder=encoder,
                predictor=predictor,
                loader=val_loader,
                device=device,
                rollout_steps=int(args.rollout_steps),
                use_amp=use_amp,
            )
            row = {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss_total"],
                "train_loss_tf": train_metrics["loss_tf"],
                "train_loss_roll": train_metrics["loss_roll"],
                "val_loss": val_metrics["loss_total"],
                "val_loss_tf": val_metrics["loss_tf"],
                "val_loss_roll": val_metrics["loss_roll"],
            }
            history.append(row)
            writer.writerow(row)
            handle.flush()
            torch.save(
                {
                    "predictor": predictor.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "config": {
                        "predictor": config["predictor"],
                        "train_ac": {
                            "dataset": str(Path(args.dataset).resolve()),
                            "epochs": int(args.epochs),
                            "batch_size": int(args.batch_size),
                            "grad_accum": int(args.grad_accum),
                            "lr": float(args.lr),
                            "weight_decay": float(args.weight_decay),
                            "rollout_steps": int(args.rollout_steps),
                            "amp": bool(args.amp),
                        },
                    },
                    "metrics": history,
                },
                latest_path,
            )
            print(
                f"epoch={epoch:03d} train={row['train_loss']:.6f} val={row['val_loss']:.6f} "
                f"time={time.time() - start:.1f}s"
            )

    print(f"checkpoint: {latest_path}")
    print("train-ac: ok")
    return 0


# ==============================================================================
# [SECTION 4] EVALUATION (LATENT PREDICTOR)
# 包含：评估训练好的神经网络预测器效果 (eval-ac)
# ==============================================================================
def run_eval_ac(args: argparse.Namespace) -> int:
    config, device, _ = setup_run(args)

    encoder = build_encoder(config, args.require_official_vjepa, device)
    predictor = ActionConditionedPredictor(predictor_config_from_dict(config["predictor"])).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    predictor.load_state_dict(checkpoint["predictor"])
    print(f"loaded predictor checkpoint: {Path(args.checkpoint).resolve()} epoch={checkpoint.get('epoch', 'unknown')}")
    del checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    demo_cfg = dict(config.get("oracle_demo", {}))
    planner_cfg_data = dict(config["planner"])
    planner_cfg_data.update(
        {
            "horizon": int(args.horizon or planner_cfg_data["horizon"]),
            "n_candidates": int(args.candidates or planner_cfg_data["n_candidates"]),
            "n_elites": int(args.elites or planner_cfg_data["n_elites"]),
            "n_iters": int(args.iters or planner_cfg_data["n_iters"]),
            "topk_physics": int(args.topk_visualize),
            "chunk_size": int(args.chunk_size or min(int(planner_cfg_data.get("chunk_size", 16)), 4)),
        }
    )
    planner = LatentCEMPlanner(
        predictor=predictor,
        encoder=encoder,
        config=cem_config_from_dict(planner_cfg_data),
        device=device,
    )
    action_repeat = int(args.action_repeat or demo_cfg.get("action_repeat", 5))
    video_fps = int(args.video_fps or demo_cfg.get("video_fps", 12))
    env = make_smoke_env(config["environment"])
    env.reset()
    initial_x = env.get_agent_x()
    initial_y = env.get_agent_y()
    initial_reward = env.compute_reward()
    initial_distance = env.distance_to_goal()
    goal_z, target_frame, target_summary = build_goal_latent(env, encoder, device, int(args.target_steps or demo_cfg.get("target_steps", 80)))

    metrics: list[dict[str, float]] = []
    sequence_len = int(getattr(args, "sequence_len", 4))
    execution_frames = [env.render().copy()]
    # 处理数据不足 N 帧的初始情况：通过复制初始帧进行 Padding，
    # 这样既保证送入 Predictor 的序列长度符合预期，也自然地表示“当前速度为 0”。
    macro_context_frames = [execution_frames[0].copy() for _ in range(sequence_len)]
    candidate_grid_rollouts: list[list[np.ndarray]] | None = None
    for step_idx in range(int(args.mpc_steps or demo_cfg.get("mpc_steps", 20))):
        before_plan = env.snapshot_signature()
        with torch.no_grad():
            z_ctx = encoder(frames_to_tensor(macro_context_frames, device))
            state0 = torch.from_numpy(env.get_state_vector()).to(device) if args.condition_on_state else torch.zeros(7, device=device)
            top_actions, latent_costs = planner.plan(z_ctx, goal_z, state0)
            random_baseline = _random_latent_baseline(planner, z_ctx, goal_z, state0, planner.config.horizon)
        if env.snapshot_signature() != before_plan:
            raise RuntimeError("AC planning changed the live environment state.")
        top_sequences = top_actions.detach().cpu().numpy()
        if candidate_grid_rollouts is None:
            candidate_grid_rollouts = collect_candidate_rollouts(env, top_sequences, action_repeat)
        action = top_sequences[0, 0]
        reward = env.compute_reward()
        done = False
        frame, reward, done, _ = env.step_macro(action, action_repeat)
        execution_frames.append(frame.copy())
        # 维持固定长度为 sequence_len 的滑动窗口
        macro_context_frames.append(frame.copy())
        macro_context_frames = macro_context_frames[-sequence_len:]
        with torch.no_grad():
            current_z = encoder(env.get_frame_tensor(str(device)))[:, -1]
            actual_latent_cost = float(mean_patch_cost(current_z, goal_z.to(device)).item())
        row = {
            "step": float(step_idx),
            "chosen_action0": float(action[0]),
            "chosen_action1": float(action[1]),
            "x": env.get_agent_x(),
            "y": env.get_agent_y(),
            "reward": float(reward),
            "distance_to_goal": env.distance_to_goal(),
            "latent_cost": actual_latent_cost,
            "planned_terminal_cost": float(latent_costs[0].detach().cpu()),
            "random_baseline_cost": random_baseline,
        }
        metrics.append(row)
        print(
            f"step={step_idx:02d} action=({row['chosen_action0']:+.3f},{row['chosen_action1']:+.3f}) "
            f"xy=({row['x']:.4f},{row['y']:.4f}) reward={row['reward']:.4f} "
            f"latent_cost={actual_latent_cost:.6f} planned={row['planned_terminal_cost']:.6f} "
            f"random={random_baseline:.6f}"
        )
        if done:
            break

    if env.stack_depth != 0:
        raise RuntimeError("AC eval leaked rollback stack state.")
    final_x = env.get_agent_x()
    final_y = env.get_agent_y()
    final_reward = env.compute_reward()
    final_distance = env.distance_to_goal()
    success = final_distance < initial_distance
    save_run_videos(output_dir, execution_frames, candidate_grid_rollouts, target_frame, video_fps)
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
    print(f"eval-ac: {'ok' if success else 'completed-without-success'}")
    return 0


# ==============================================================================
# [SECTION 5] INTERNAL HELPERS (DATA COLLECTION)
# ==============================================================================
def _collect_synthetic_split(
    h5: h5py.File,
    split: str,
    windows: int,
    env,
    planner: OracleCEMPlanner,
    encoder,
    device: torch.device,
    rng: np.random.Generator,
    sequence_len: int,
    action_repeat: int,
    target_steps: int,
) -> None:
    group = h5.create_group(split)
    frames_ds = group.create_dataset(
        "frames",
        shape=(windows, sequence_len + 1, 384, 384, 3),
        dtype=np.uint8,
        chunks=(1, sequence_len + 1, 384, 384, 3),
        compression="lzf",
    )
    actions_ds = group.create_dataset("actions", shape=(windows, sequence_len, 7), dtype=np.float32)
    states_ds = group.create_dataset("states", shape=(windows, sequence_len, 7), dtype=np.float32)
    sources = _make_source_schedule(windows, rng)

    for idx in range(windows):
        env.reset()
        env.set_agent_xy(_sample_start_xy(rng))
        source = sources[idx]
        if source == "oracle":
            goal_z, _, _ = build_goal_latent(env, encoder, device, target_steps)
            before = env.snapshot_signature()
            result = planner.plan(env, goal_z)
            if env.snapshot_signature() != before:
                raise RuntimeError("Oracle synthetic planning changed the live environment state.")
            actions = result.best_sequence[:sequence_len]
        else:
            actions = _scripted_actions(env, rng, sequence_len, source)
        frames, states, _, summary = _record_macro_window(env, actions, action_repeat)
        if env.stack_depth != 0:
            raise RuntimeError("Synthetic data collection leaked rollback stack state.")
        frames_ds[idx] = frames
        actions_ds[idx] = actions.astype(np.float32)
        states_ds[idx] = states.astype(np.float32)
        if (idx + 1) % max(1, min(100, windows)) == 0 or idx == windows - 1:
            print(
                f"{split}: {idx + 1}/{windows} source={source} "
                f"xy=({summary['x']:.3f},{summary['y']:.3f}) reward={summary['reward']:.3f}"
            )

    group.attrs["sources"] = ",".join(sources)


def _collect_latent_split(
    split: str,
    windows: int,
    out_dir: Path,
    env,
    planner: OracleCEMPlanner,
    encoder,
    device: torch.device,
    rng: np.random.Generator,
    sequence_len: int,
    action_repeat: int,
    target_steps: int,
    shard_size: int,
    encode_batch_size: int,
    amp: bool,
    resume: bool,
    seed: int,
    oracle_cfg,
) -> None:
    sources = _make_source_schedule(windows, rng)
    total_shards = math.ceil(windows / shard_size)
    for shard_idx in range(total_shards):
        start = shard_idx * shard_size
        stop = min(windows, start + shard_size)
        final_path = out_dir / f"{split}-{shard_idx:05d}.h5"
        if resume and _latent_shard_complete(final_path):
            print(f"{split}: 跳过已完成的分片 {shard_idx + 1}/{total_shards}")
            continue
        tmp_path = out_dir / f"{split}-{shard_idx:05d}.tmp.h5"
        if tmp_path.exists():
            tmp_path.unlink()
        count = stop - start
        frames = np.zeros((count, sequence_len + 1, 384, 384, 3), dtype=np.uint8)
        actions = np.zeros((count, sequence_len, 7), dtype=np.float32)
        states = np.zeros((count, sequence_len, 7), dtype=np.float32)
        next_states = np.zeros((count, sequence_len, 7), dtype=np.float32)
        shard_sources = sources[start:stop]
        for local_idx, source in enumerate(shard_sources):
            env.reset()
            env.set_agent_xy(_sample_start_xy(rng))
            if source == "oracle":
                goal_z, _, _ = build_goal_latent(env, encoder, device, target_steps)
                before = env.snapshot_signature()
                result = planner.plan(env, goal_z)
                if env.snapshot_signature() != before:
                    raise RuntimeError("Oracle 隐变量分片规划改变了实时环境状态。")
                action_window = result.best_sequence[:sequence_len]
            else:
                action_window = _scripted_actions(env, rng, sequence_len, source)
            frame_window, state_window, next_state_window, _ = _record_macro_window(env, action_window, action_repeat)
            frames[local_idx] = frame_window
            actions[local_idx] = action_window.astype(np.float32)
            states[local_idx] = state_window.astype(np.float32)
            next_states[local_idx] = next_state_window.astype(np.float32)
            if env.stack_depth != 0:
                raise RuntimeError("隐变量分片收集导致回滚栈状态泄露。")

        z = _encode_frame_windows(
            encoder=encoder,
            frames=frames,
            device=device,
            batch_size=encode_batch_size,
            use_amp=amp and device.type == "cuda",
        )
        with h5py.File(tmp_path, "w") as h5:
            h5.create_dataset("z", data=z, dtype=np.float16, chunks=(1, sequence_len + 1, 576, 768), compression="lzf")
            h5.create_dataset("actions", data=actions, dtype=np.float32)
            h5.create_dataset("states", data=states, dtype=np.float32)
            h5.create_dataset("next_states", data=next_states, dtype=np.float32)
            h5.attrs["complete"] = 1
            h5.attrs["split"] = split
            h5.attrs["seed"] = seed
            h5.attrs["sequence_len"] = sequence_len
            h5.attrs["action_repeat"] = action_repeat
            h5.attrs["sources"] = ",".join(shard_sources)
            h5.attrs["oracle_config"] = repr(oracle_cfg)
        os.replace(tmp_path, final_path)
        del frames, actions, states, next_states, z
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"{split}: shard {shard_idx + 1}/{total_shards} windows={count} path={final_path.name}")
    (out_dir / f"{split}.DONE").write_text(str(windows), encoding="utf-8")


@torch.no_grad()
def _encode_frame_windows(
    encoder,
    frames: np.ndarray,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, frames.shape[0], batch_size):
        batch_np = frames[start : start + batch_size].astype(np.float32) / 255.0
        batch = torch.from_numpy(batch_np).permute(0, 1, 4, 2, 3).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            z = encoder(batch)
        outputs.append(z.detach().to(dtype=torch.float16, device="cpu").numpy())
        del batch, z
    return np.concatenate(outputs, axis=0)


def _latent_shard_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as h5:
            return bool(h5.attrs.get("complete", 0)) and "z" in h5 and "actions" in h5 and "states" in h5
    except OSError:
        return False


def _list_complete_latent_shards(data_dir: Path, split: str) -> list[Path]:
    return [path for path in sorted(data_dir.glob(f"{split}-*.h5")) if _latent_shard_complete(path)]


# ==============================================================================
# [SECTION 6] INTERNAL HELPERS (TRAINING)
# ==============================================================================
def _train_latent_shard(
    predictor: ActionConditionedPredictor,
    optimizer: torch.optim.Optimizer,
    scaler,
    shard: Path,
    device: torch.device,
    batch_size: int,
    grad_accum: int,
    rollout_steps: int,
    use_amp: bool,
    rng: np.random.Generator,
    condition_on_state: bool,
) -> dict[str, float]:
    predictor.train()
    totals = {
        "loss_tf": 0.0,
        "loss_roll": 0.0,
        "loss_delta": 0.0,
        "loss_delta_roll": 0.0,
        "loss_action_contrast": 0.0,
        "loss_total": 0.0,
    }
    count = 0
    optimizer.zero_grad(set_to_none=True)

    # 将整个 shard 预加载到 RAM，关闭文件后再训练。
    # 这消除了训练循环内的所有磁盘 I/O：原先的随机索引会导致 HDF5 非连续小块读，
    # 现在均为内存切片， GPU 干活期间 CPU 无需等待 I/O。
    with h5py.File(shard, "r") as h5:
        n_items = int(h5["z"].shape[0])
        z_all: np.ndarray = h5["z"][:]               # float16 [N, T+1, 576, 768]
        actions_all: np.ndarray = h5["actions"][:].astype(np.float32)
        states_all: np.ndarray = h5["states"][:].astype(np.float32) if condition_on_state else None

    order = rng.permutation(n_items)
    step = 0
    for start in range(0, n_items, batch_size):
        idx = order[start : start + batch_size]
        # 内存切片 + float16→float32，再 non_blocking 传输到 GPU。
        z = torch.from_numpy(z_all[idx].astype(np.float32)).to(device, non_blocking=True)
        actions = torch.from_numpy(actions_all[idx]).to(device, non_blocking=True)
        if condition_on_state and states_all is not None:
            states = torch.from_numpy(states_all[idx]).to(device, non_blocking=True)
        else:
            states = torch.zeros_like(actions)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss, metrics = compute_ac_predictor_latent_loss(
                predictor,
                z,
                actions,
                states,
                rollout_steps=rollout_steps,
            )
            scaled_loss = loss / max(1, grad_accum)
        if not torch.isfinite(loss):
            raise RuntimeError(f"在 {shard} 中出现非有限的隐变量训练损失：{metrics}")
        scaler.scale(scaled_loss).backward()
        step += 1
        if step % max(1, grad_accum) == 0 or start + batch_size >= n_items:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        for key in totals:
            totals[key] += metrics[key]
        count += 1
        del z, actions, states, loss
    # 不在每个 shard 后调用 empty_cache：PyTorch 的 CUDA 内存分配器会自动管理
    # 缓存块，强制清空反而会导致下一个 shard 开始时重新分配碎片内存。
    return {key: value / max(1, count) for key, value in totals.items()}


@torch.no_grad()
def _evaluate_latent_shards(
    predictor: ActionConditionedPredictor,
    shards: list[Path],
    device: torch.device,
    batch_size: int,
    rollout_steps: int,
    use_amp: bool,
    condition_on_state: bool,
) -> dict[str, float]:
    predictor.eval()
    totals = {
        "loss_tf": 0.0,
        "loss_roll": 0.0,
        "loss_delta": 0.0,
        "loss_delta_roll": 0.0,
        "loss_action_contrast": 0.0,
        "loss_total": 0.0,
    }
    count = 0
    for shard in shards:
        # 预加载整个 shard 到 RAM，关闭文件后再推理，消除推理循环中的磁盘 I/O。
        with h5py.File(shard, "r") as h5:
            n_items = int(h5["z"].shape[0])
            z_all: np.ndarray = h5["z"][:]           # float16
            actions_all: np.ndarray = h5["actions"][:].astype(np.float32)
            states_all: np.ndarray = h5["states"][:].astype(np.float32) if condition_on_state else None
        for start in range(0, n_items, batch_size):
            slc = slice(start, min(n_items, start + batch_size))
            z = torch.from_numpy(z_all[slc].astype(np.float32)).to(device, non_blocking=True)
            actions = torch.from_numpy(actions_all[slc]).to(device, non_blocking=True)
            if condition_on_state and states_all is not None:
                states = torch.from_numpy(states_all[slc]).to(device, non_blocking=True)
            else:
                states = torch.zeros_like(actions)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss, metrics = compute_ac_predictor_latent_loss(
                    predictor,
                    z,
                    actions,
                    states,
                    rollout_steps=rollout_steps,
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"在 {shard} 中出现非有限的隐变量验证损失：{metrics}")
            for key in totals:
                totals[key] += metrics[key]
            count += 1
            del z, actions, states, loss
    return {key: value / max(1, count) for key, value in totals.items()}


def _save_latent_checkpoint(
    path: Path,
    predictor: ActionConditionedPredictor,
    optimizer: torch.optim.Optimizer,
    config: dict,
    args: argparse.Namespace,
    history: list[dict[str, float]],
) -> None:
    torch.save(
        {
            "predictor": predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": {
                "predictor": config["predictor"],
                "train_ac_latent": _serializable_args(args),
            },
            "metrics": history,
        },
        path,
    )


# ==============================================================================
# [SECTION 7] INTERNAL HELPERS (UTILITIES)
# ==============================================================================
def _serializable_args(args: argparse.Namespace) -> dict:
    data = {}
    for key, value in vars(args).items():
        if key == "func" or callable(value):
            continue
        if isinstance(value, Path):
            data[key] = str(value)
        elif isinstance(value, (str, int, float, bool, type(None))):
            data[key] = value
        else:
            data[key] = repr(value)
    return data


def _make_source_schedule(windows: int, rng: np.random.Generator) -> list[str]:
    # 彻底移除极度耗时的 Oracle CEM 规划，全部使用脚本/随机探索。
    # 这样 Collector 速度将提升百倍，能立刻喂饱 Trainer 的 GPU。
    directional_count = int(round(windows * 0.50))
    random_count = windows - directional_count
    sources = ["directional"] * directional_count + ["random"] * random_count
    rng.shuffle(sources)
    return sources


def _sample_start_xy(rng: np.random.Generator) -> np.ndarray:
    for _ in range(100):
        xy = np.array([rng.uniform(-0.85, 0.55), rng.uniform(-0.85, 0.55)], dtype=np.float32)
        if not (abs(float(xy[0])) < 0.14 and abs(float(xy[1])) < 0.55):
            return xy
    return np.array([-0.75, -0.55], dtype=np.float32)


def _scripted_actions(env, rng: np.random.Generator, sequence_len: int, source: str) -> np.ndarray:
    actions = np.zeros((sequence_len, 7), dtype=np.float32)
    
    # 针对包含物理惯性（damping/friction）的环境，白噪声的力会相互抵消导致原地抖动。
    # 所以在 random 策略下，我们为整个 window 采样一个统一的基础发力方向：
    if source == "random":
        base_xy = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        # 35% 概率掺杂朝向目标方向的拉力
        if rng.random() < 0.35:
            base_xy += 0.4 * (env.goal_xy - env.get_agent_xy())
            
    for step in range(sequence_len):
        if source == "directional":
            if env.get_agent_x() < 0.25:
                xy = np.array([1.0, 0.15], dtype=np.float32)
            else:
                xy = np.array([0.0, 1.0], dtype=np.float32)
            xy += rng.normal(0.0, 0.15, size=2).astype(np.float32)
        elif source == "random":
            # 基础方向 + 较小的正态分布噪声，确保能克服惯性滚起来
            xy = base_xy + rng.normal(0.0, 0.15, size=2).astype(np.float32)
        else:
            raise ValueError(f"未知的合成数据源：{source}")
        actions[step, :2] = np.clip(xy, -1.0, 1.0)
    return actions


def _record_macro_window(env, actions: np.ndarray, action_repeat: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    sequence_len = int(actions.shape[0])
    frames = np.zeros((sequence_len + 1, 384, 384, 3), dtype=np.uint8)
    states = np.zeros((sequence_len, 7), dtype=np.float32)
    next_states = np.zeros((sequence_len, 7), dtype=np.float32)
    frames[0] = env.render().copy()
    reward = env.compute_reward()
    done = False
    for step, action in enumerate(actions.astype(np.float32)):
        states[step] = env.get_state_vector()
        frame, reward, done, _ = env.step_macro(action, action_repeat)
        next_states[step] = env.get_state_vector()
        frames[step + 1] = frame.copy()
    summary = {
        "x": env.get_agent_x(),
        "y": env.get_agent_y(),
        "reward": float(reward),
        "done": float(done),
        "distance_to_goal": env.distance_to_goal(),
    }
    return frames, states, next_states, summary


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frames = batch["frames"].to(device, non_blocking=True)
    actions = batch["actions"].to(device, non_blocking=True)
    states = batch["states"].to(device, non_blocking=True)
    return frames, actions, states


def _run_ac_epoch(
    encoder,
    predictor: ActionConditionedPredictor,
    loader: DataLoader,
    device: torch.device,
    rollout_steps: int,
    optimizer: torch.optim.Optimizer,
    scaler,
    grad_accum: int,
    use_amp: bool,
) -> dict[str, float]:
    predictor.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss_tf": 0.0, "loss_roll": 0.0, "loss_total": 0.0}
    count = 0
    for step, batch in enumerate(loader, start=1):
        frames, actions, states = _move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss, metrics = compute_ac_predictor_loss(encoder, predictor, frames, actions, states, rollout_steps=rollout_steps)
            scaled_loss = loss / max(1, grad_accum)
        if not torch.isfinite(loss):
            raise RuntimeError(f"AC 训练损失不为有限值：{metrics}")
        scaler.scale(scaled_loss).backward()
        if step % max(1, grad_accum) == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        for key in totals:
            totals[key] += metrics[key]
        count += 1
    return {key: value / max(1, count) for key, value in totals.items()}


@torch.no_grad()
def _evaluate_ac_loss(
    encoder,
    predictor: ActionConditionedPredictor,
    loader: DataLoader,
    device: torch.device,
    rollout_steps: int,
    use_amp: bool,
) -> dict[str, float]:
    predictor.eval()
    totals = {"loss_tf": 0.0, "loss_roll": 0.0, "loss_total": 0.0}
    count = 0
    for batch in loader:
        frames, actions, states = _move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss, metrics = compute_ac_predictor_loss(encoder, predictor, frames, actions, states, rollout_steps=rollout_steps)
        if not torch.isfinite(loss):
            raise RuntimeError(f"AC 验证损失不为有限值：{metrics}")
        for key in totals:
            totals[key] += metrics[key]
        count += 1
    return {key: value / max(1, count) for key, value in totals.items()}


@torch.no_grad()
def _random_latent_baseline(
    planner: LatentCEMPlanner,
    z_ctx: torch.Tensor,
    goal_z: torch.Tensor,
    state0: torch.Tensor,
    horizon: int,
) -> float:
    actions = torch.empty(32, horizon, 7, device=planner.device).uniform_(-1.0, 1.0)
    costs = planner.evaluate_sequences(z_ctx, goal_z, actions, state0)
    return float(costs.min().detach().cpu())



# ==============================================================================
# [SECTION 8] CLI PARSER & MAIN ENTRYPOINT
# ==============================================================================
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
    collect = sub.add_parser("collect-synthetic", help="Collect synthetic HDF5 windows for V-JEPA2-AC training.")
    collect.add_argument("--config", default="configs.yaml")
    collect.add_argument("--require-official-vjepa", action="store_true")
    collect.add_argument("--output", default="data/synthetic_ac_medium.h5")
    collect.add_argument("--train-windows", type=int, default=5000)
    collect.add_argument("--val-windows", type=int, default=500)
    collect.add_argument("--sequence-len", type=int, default=4)
    collect.add_argument("--action-repeat", type=int, default=5)
    collect.add_argument("--target-steps", type=int, default=80)
    collect.add_argument("--oracle-horizon", type=int)
    collect.add_argument("--oracle-candidates", type=int)
    collect.add_argument("--oracle-elites", type=int)
    collect.add_argument("--oracle-iters", type=int)
    collect.add_argument("--encode-batch-size", type=int)
    collect.add_argument("--seed", type=int)
    collect.set_defaults(func=run_collect_synthetic)
    collect_latent = sub.add_parser("collect-latent-shards", help="Collect atomic latent shards for V-JEPA2-AC training.")
    collect_latent.add_argument("--config", default="configs.yaml")
    collect_latent.add_argument("--require-official-vjepa", action="store_true")
    collect_latent.add_argument("--output-dir", default="data/ac_latent_shards")
    collect_latent.add_argument("--train-windows", type=int, default=5000)
    collect_latent.add_argument("--val-windows", type=int, default=500)
    collect_latent.add_argument("--sequence-len", type=int, default=4)
    collect_latent.add_argument("--action-repeat", type=int, default=5)
    collect_latent.add_argument("--target-steps", type=int, default=40)   # 原80，生成goal目标的仿真步数
    collect_latent.add_argument("--shard-size", type=int, default=128)   # 原64，更大shard减少分片开销
    collect_latent.add_argument("--oracle-horizon", type=int)
    collect_latent.add_argument("--oracle-candidates", type=int, default=4)  # 原8，CEM候选数少一半规划速度2×
    collect_latent.add_argument("--oracle-elites", type=int, default=1)      # 原2
    collect_latent.add_argument("--oracle-iters", type=int, default=1)
    collect_latent.add_argument("--encode-batch-size", type=int, default=32)  # 原16，GPU利用率↑
    collect_latent.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    collect_latent.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    collect_latent.add_argument("--seed", type=int)
    collect_latent.add_argument("--wandb-project", default=None, help="W&B 项目名，子进程中显式初始化自己的 run")
    collect_latent.add_argument("--wandb-run-name", default=None, help="W&B run 名称")
    collect_latent.set_defaults(func=run_collect_latent_shards)
    train_latent = sub.add_parser("train-ac-latent", help="Train the AC predictor from cached latent shards.")
    train_latent.add_argument("--config", default="configs.yaml")
    train_latent.add_argument("--data-dir", default="data/ac_latent_shards")
    train_latent.add_argument("--output-dir", default="runs/ac_train")
    train_latent.add_argument("--epochs", type=int, default=10)
    train_latent.add_argument("--batch-size", type=int, default=4)   # 原1，提升GPU每步工作量
    train_latent.add_argument("--grad-accum", type=int, default=2)   # 原8，有效批量不变=8
    train_latent.add_argument("--lr", type=float, default=1e-4)
    train_latent.add_argument("--weight-decay", type=float, default=0.04)
    train_latent.add_argument("--rollout-steps", type=int, default=4)
    train_latent.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    train_latent.add_argument("--watch", action="store_true")
    train_latent.add_argument("--expected-train-shards", type=int)
    train_latent.add_argument("--expected-val-shards", type=int)
    train_latent.add_argument("--poll-seconds", type=float, default=5.0)
    train_latent.add_argument("--replay-while-waiting", action="store_true")
    train_latent.add_argument("--condition-on-state", action=argparse.BooleanOptionalAction, default=False)
    train_latent.add_argument("--resume-checkpoint")
    train_latent.add_argument("--seed", type=int)
    train_latent.add_argument("--wandb-project", default=None, help="W&B 项目名，子进程中显式初始化自己的 run")
    train_latent.add_argument("--wandb-run-name", default=None, help="W&B run 名称")
    train_latent.set_defaults(func=run_train_ac_latent)
    pipe_latent = sub.add_parser("pipeline-latent", help="Run latent shard collection and training concurrently.")
    pipe_latent.add_argument("--run-dir", default="runs/latent_pipeline")
    pipe_latent.add_argument("--data-dir", default="data/ac_latent_shards")
    pipe_latent.add_argument("--output-dir", default="runs/ac_train")
    pipe_latent.add_argument("--train-windows", type=int, default=5000)
    pipe_latent.add_argument("--val-windows", type=int, default=500)
    pipe_latent.add_argument("--shard-size", type=int, default=128)          # 原64
    pipe_latent.add_argument("--oracle-candidates", type=int, default=4)     # 原8，规划速度2×
    pipe_latent.add_argument("--oracle-elites", type=int, default=1)         # 原2
    pipe_latent.add_argument("--oracle-iters", type=int, default=1)
    pipe_latent.add_argument("--sequence-len", type=int, default=4)
    pipe_latent.add_argument("--rollout-steps", type=int, default=4)
    pipe_latent.add_argument("--final-epochs", type=int, default=4)
    pipe_latent.add_argument("--batch-size", type=int, default=4)            # 原1
    pipe_latent.add_argument("--grad-accum", type=int, default=2)            # 原8，有效批量=8不变
    pipe_latent.add_argument("--condition-on-state", action=argparse.BooleanOptionalAction, default=False)
    pipe_latent.add_argument("--wandb-project", default=None, help="W&B 项目名，透传给 collector/trainer 子进程")
    pipe_latent.set_defaults(func=run_pipeline_latent)
    train_ac = sub.add_parser("train-ac", help="Train the action-conditioned latent predictor.")
    train_ac.add_argument("--config", default="configs.yaml")
    train_ac.add_argument("--require-official-vjepa", action="store_true")
    train_ac.add_argument("--dataset", default="data/synthetic_ac_medium.h5")
    train_ac.add_argument("--output-dir", default="runs/ac_train")
    train_ac.add_argument("--epochs", type=int, default=10)
    train_ac.add_argument("--batch-size", type=int, default=1)
    train_ac.add_argument("--grad-accum", type=int, default=8)
    train_ac.add_argument("--lr", type=float, default=1e-4)
    train_ac.add_argument("--weight-decay", type=float, default=0.04)
    train_ac.add_argument("--rollout-steps", type=int, default=2)
    train_ac.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    train_ac.add_argument("--seed", type=int)
    train_ac.set_defaults(func=run_train_ac)
    eval_ac = sub.add_parser("eval-ac", help="Evaluate a trained action-conditioned predictor with latent CEM.")
    eval_ac.add_argument("--config", default="configs.yaml")
    eval_ac.add_argument("--require-official-vjepa", action="store_true")
    eval_ac.add_argument("--checkpoint", default="runs/ac_train/latest.pt")
    eval_ac.add_argument("--output-dir", default="runs/ac_eval/latest")
    eval_ac.add_argument("--mpc-steps", type=int)
    eval_ac.add_argument("--horizon", type=int)
    eval_ac.add_argument("--candidates", type=int)
    eval_ac.add_argument("--elites", type=int)
    eval_ac.add_argument("--iters", type=int)
    eval_ac.add_argument("--chunk-size", type=int)
    eval_ac.add_argument("--action-repeat", type=int)
    eval_ac.add_argument("--target-steps", type=int)
    eval_ac.add_argument("--topk-visualize", type=int, default=4)
    eval_ac.add_argument("--video-fps", type=int)
    eval_ac.add_argument("--seed", type=int)
    eval_ac.add_argument("--sequence-len", type=int, default=4, help="必须与训练时的数据采集长度保持对齐")
    eval_ac.add_argument("--condition-on-state", action=argparse.BooleanOptionalAction, default=False)
    eval_ac.set_defaults(func=run_eval_ac)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
