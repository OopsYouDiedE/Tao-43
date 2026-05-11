from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from core_models import ActionConditionedPredictor, mean_patch_cost
from environment import RollbackMuJoCoEnv


@dataclass(frozen=True)
class CEMConfig:
    horizon: int = 15
    n_candidates: int = 512
    n_elites: int = 64
    n_iters: int = 10
    topk_physics: int = 5
    action_low: float = -1.0
    action_high: float = 1.0
    initial_std: float = 0.6
    min_std: float = 0.05
    chunk_size: int = 16


class LatentCEMPlanner:
    def __init__(
        self,
        predictor: ActionConditionedPredictor,
        encoder,
        config: CEMConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.predictor = predictor.to(device).eval()
        self.encoder = encoder
        self.config = config or CEMConfig()
        self.device = torch.device(device)

    @torch.no_grad()
    def rollout_latent(self, z_ctx: Tensor, actions: Tensor, states: Tensor) -> Tensor:
        z_window = z_ctx.to(self.device)
        actions, states = actions.to(self.device), states.to(self.device)
        batch, ctx_len, patches, visual_dim = z_window.shape
        horizon = actions.shape[1]
        if actions.shape != (batch, horizon, 7) or states.shape != (batch, horizon, 7):
            raise ValueError("actions 和 states 必须是 [B, H, 7] 的形状。")

        preds: list[Tensor] = []
        cond_actions = torch.zeros(batch, ctx_len, 7, device=self.device, dtype=z_window.dtype)
        cond_states = torch.zeros(batch, ctx_len, 7, device=self.device, dtype=z_window.dtype)
        for step in range(horizon):
            cond_actions[:, -1] = actions[:, step]
            cond_states[:, -1] = states[:, step]
            pred_window = self.predictor(z_window, cond_actions, cond_states)
            next_z = pred_window[:, -1]
            preds.append(next_z)
            z_window = torch.cat([z_window[:, 1:], next_z[:, None]], dim=1)
        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def evaluate_sequences(self, z_ctx: Tensor, goal_z: Tensor, actions: Tensor, state0: Tensor) -> Tensor:
        costs: list[Tensor] = []
        state_expanded = state0.to(self.device).view(1, 1, 7).expand(-1, actions.shape[1], 7)
        for start in range(0, actions.shape[0], self.config.chunk_size):
            chunk = actions[start : start + self.config.chunk_size].to(self.device)
            z_batch = z_ctx.to(self.device).expand(chunk.shape[0], -1, -1, -1).contiguous()
            state_batch = state_expanded.expand(chunk.shape[0], -1, -1)
            rollout = self.rollout_latent(z_batch, chunk, state_batch)
            costs.append(mean_patch_cost(rollout[:, -1], goal_z.to(self.device).expand(chunk.shape[0], -1, -1)))
        return torch.cat(costs, dim=0)

    @torch.no_grad()
    def plan(self, z_ctx: Tensor, goal_z: Tensor, state0: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.config
        mean = torch.zeros(cfg.horizon, 7, device=self.device)
        std = torch.full_like(mean, cfg.initial_std)
        best_actions = None
        best_costs = None
        for _ in range(cfg.n_iters):
            samples = mean + std * torch.randn(cfg.n_candidates, cfg.horizon, 7, device=self.device)
            samples = samples.clamp(cfg.action_low, cfg.action_high)
            costs = self.evaluate_sequences(z_ctx, goal_z, samples, state0)
            elite_count = min(cfg.n_elites, cfg.n_candidates)
            elite_idx = torch.topk(costs, k=elite_count, largest=False).indices
            elites = samples[elite_idx]
            mean = elites.mean(dim=0)
            std = elites.std(dim=0, unbiased=False).clamp_min(cfg.min_std)
            best_actions, best_costs = samples, costs
        assert best_actions is not None and best_costs is not None
        topk = min(cfg.topk_physics, cfg.n_candidates)
        top_idx = torch.topk(best_costs, k=topk, largest=False).indices
        return best_actions[top_idx], best_costs[top_idx]

    @torch.no_grad()
    def verify_with_physics(
        self,
        env: RollbackMuJoCoEnv,
        action_sequences: Tensor,
        goal_z: Tensor,
    ) -> tuple[Tensor, float]:
        best_seq: Tensor | None = None
        best_cost = float("inf")
        for seq in action_sequences.detach().cpu():
            env.push_state()
            try:
                for action in seq.numpy().astype(np.float32):
                    env.step(action)
                actual_z = self.encoder(env.get_frame_tensor(str(self.device)))[:, -1]
                cost = float(mean_patch_cost(actual_z, goal_z.to(self.device)).item())
            finally:
                env.pop_state()
            if cost < best_cost:
                best_seq = seq
                best_cost = cost
        if best_seq is None:
            raise RuntimeError("物理验证未接收到任何动作序列。")
        return best_seq, best_cost


def cem_config_from_dict(data: dict) -> CEMConfig:
    allowed = CEMConfig.__dataclass_fields__.keys()
    return CEMConfig(**{key: data[key] for key in allowed if key in data})


@dataclass(frozen=True)
class OracleCEMConfig:
    horizon: int = 8
    n_candidates: int = 32
    n_elites: int = 8
    n_iters: int = 3
    action_dims: int = 2
    action_repeat: int = 5
    distance_cost_weight: float = 0.0
    use_action_priors: bool = False
    action_low: float = -1.0
    action_high: float = 1.0
    initial_std: float = 0.8
    min_std: float = 0.08
    encode_batch_size: int = 4
    topk_visualize: int = 4
    seed: int = 43


@dataclass(frozen=True)
class OracleIterationStat:
    iteration: int
    best_cost: float
    mean_cost: float
    best_reward: float
    best_x: float
    best_y: float
    elite_mean_action0: float
    elite_mean_action1: float
    elite_std_action0: float


@dataclass(frozen=True)
class OraclePlanResult:
    best_sequence: np.ndarray
    best_cost: float
    best_reward: float
    best_x: float
    best_y: float
    top_sequences: np.ndarray
    top_costs: np.ndarray
    stats: list[OracleIterationStat]


class OracleCEMPlanner:
    """用于预动作头阶段的在环模拟器 CEM 规划器。"""

    def __init__(self, encoder, config: OracleCEMConfig | None = None, device: torch.device | str = "cpu") -> None:
        self.encoder = encoder
        self.config = config or OracleCEMConfig()
        self.device = torch.device(device)
        self.rng = np.random.default_rng(self.config.seed)

    @torch.no_grad()
    def plan(self, env: RollbackMuJoCoEnv, goal_z: Tensor) -> OraclePlanResult:
        cfg = self.config
        mean = np.zeros((cfg.horizon, cfg.action_dims), dtype=np.float32)
        std = np.full_like(mean, cfg.initial_std)
        stats: list[OracleIterationStat] = []
        last_sequences: np.ndarray | None = None
        last_costs: np.ndarray | None = None
        last_rewards: np.ndarray | None = None
        last_xs: np.ndarray | None = None
        last_ys: np.ndarray | None = None

        for iteration in range(cfg.n_iters):
            action_samples = self.rng.normal(
                mean,
                std,
                size=(cfg.n_candidates, cfg.horizon, cfg.action_dims),
            ).astype(np.float32)
            action_samples = np.clip(action_samples, cfg.action_low, cfg.action_high)
            if cfg.use_action_priors:
                self._inject_directional_priors(action_samples, env)
            sequences = self._to_7d_actions(action_samples)
            costs, rewards, xs, ys = self.evaluate_sequences(env, sequences, goal_z)

            elite_count = min(cfg.n_elites, cfg.n_candidates)
            elite_idx = np.argsort(costs)[:elite_count]
            elites = action_samples[elite_idx]
            mean = elites.mean(axis=0)
            std = np.maximum(elites.std(axis=0), cfg.min_std)

            best_idx = int(np.argmin(costs))
            stats.append(
                OracleIterationStat(
                    iteration=iteration,
                    best_cost=float(costs[best_idx]),
                    mean_cost=float(costs.mean()),
                    best_reward=float(rewards[best_idx]),
                    best_x=float(xs[best_idx]),
                    best_y=float(ys[best_idx]),
                    elite_mean_action0=float(mean[0, 0]),
                    elite_mean_action1=float(mean[0, 1]) if cfg.action_dims > 1 else 0.0,
                    elite_std_action0=float(std[0, 0]),
                )
            )

            last_sequences = sequences
            last_costs = costs
            last_rewards = rewards
            last_xs = xs
            last_ys = ys

        if last_sequences is None or last_costs is None or last_rewards is None or last_xs is None or last_ys is None:
            raise RuntimeError("Oracle CEM 未能评估任何候选序列。")

        order = np.argsort(last_costs)
        topk = min(cfg.topk_visualize, len(order))
        best_idx = int(order[0])
        return OraclePlanResult(
            best_sequence=last_sequences[best_idx],
            best_cost=float(last_costs[best_idx]),
            best_reward=float(last_rewards[best_idx]),
            best_x=float(last_xs[best_idx]),
            best_y=float(last_ys[best_idx]),
            top_sequences=last_sequences[order[:topk]],
            top_costs=last_costs[order[:topk]],
            stats=stats,
        )

    @torch.no_grad()
    def evaluate_sequences(
        self,
        env: RollbackMuJoCoEnv,
        sequences: np.ndarray,
        goal_z: Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        terminal_tensors: list[Tensor] = []
        rewards: list[float] = []
        xs: list[float] = []
        ys: list[float] = []
        for sequence in sequences:
            before = env.snapshot_signature()
            env.push_state()
            try:
                if self.config.action_repeat > 1:
                    repeated = np.repeat(sequence, self.config.action_repeat, axis=0)
                    result = env.rollout_sequence(repeated, record_frames=False)
                else:
                    result = env.rollout_sequence(sequence, record_frames=False)
                terminal_tensors.append(env.get_frame_tensor(device=None).squeeze(0))
                rewards.append(float(result["reward"]))
                xs.append(float(result["x"]))
                ys.append(float(result["y"]))
            finally:
                env.pop_state()
            after = env.snapshot_signature()
            if before != after:
                raise RuntimeError("在评估 Oracle 候选序列后，回滚签名发生了变化。")
            if env.stack_depth != 0:
                raise RuntimeError("Oracle 候选序列评估导致回滚栈状态泄露。")

        frames = torch.stack(terminal_tensors, dim=0)
        costs: list[Tensor] = []
        for start in range(0, frames.shape[0], self.config.encode_batch_size):
            batch = frames[start : start + self.config.encode_batch_size].to(self.device)
            z_final = self.encoder(batch)[:, -1]
            goal = goal_z.to(self.device).expand(z_final.shape[0], -1, -1)
            costs.append(mean_patch_cost(z_final, goal).detach().cpu())
        latent_costs = torch.cat(costs, dim=0).numpy()
        xy = np.stack([np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)], axis=1)
        task_distances = np.linalg.norm(env.goal_xy.reshape(1, 2) - xy, axis=1)
        combined_costs = latent_costs + self.config.distance_cost_weight * task_distances
        return (
            combined_costs.astype(np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(xs, dtype=np.float32),
            np.asarray(ys, dtype=np.float32),
        )

    def _inject_directional_priors(self, scalar_samples: np.ndarray, env: RollbackMuJoCoEnv) -> None:
        priors = (
            [(0.0, 1.0), (-0.3, 1.0), (0.3, 1.0), (0.0, 0.0)]
            if env.get_agent_x() > 0.25
            else [(None, None), (1.0, 1.0), (1.0, -1.0), (1.0, 0.0), (0.0, 1.0)]
        )
        for i, prior in enumerate(priors):
            if i >= scalar_samples.shape[0]:
                break
            if prior == (None, None):  # 特殊的两阶段先验
                half = max(1, scalar_samples.shape[1] // 2)
                scalar_samples[i, :half, 0], scalar_samples[i, :half, 1] = 1.0, 0.0
                scalar_samples[i, half:, 0], scalar_samples[i, half:, 1] = 0.5, 1.0
            else:
                scalar_samples[i, :, 0], scalar_samples[i, :, 1] = prior

    @staticmethod
    def _to_7d_actions(action_samples: np.ndarray) -> np.ndarray:
        sequences = np.zeros((*action_samples.shape[:2], 7), dtype=np.float32)
        action_dims = min(action_samples.shape[2], 7)
        sequences[:, :, :action_dims] = action_samples[:, :, :action_dims]
        return sequences


def oracle_cem_config_from_dict(data: dict) -> OracleCEMConfig:
    allowed = OracleCEMConfig.__dataclass_fields__.keys()
    return OracleCEMConfig(**{key: data[key] for key in allowed if key in data})
