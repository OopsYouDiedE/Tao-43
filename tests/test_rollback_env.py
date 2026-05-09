from __future__ import annotations

import numpy as np
import pytest

dm_control = pytest.importorskip("dm_control")

from vjepa_gym.env.tasks import SmokeDetourEnv
from vjepa_gym.env.tasks import TASK_REGISTRY


def test_task_registry_exposes_smoke_task() -> None:
    assert "smoke_detour" in TASK_REGISTRY
    assert TASK_REGISTRY["smoke_detour"].horizon == 20


def test_reset_fills_frame_buffer() -> None:
    env = SmokeDetourEnv()
    obs = env.reset()

    assert obs.shape == (384, 384, 3)
    assert obs.dtype == np.uint8
    assert len(env.frame_buffer) == env.frame_buffer_len
    assert env.stack_depth == 0


def test_push_pop_restores_physics_and_frames() -> None:
    env = SmokeDetourEnv()
    env.reset()
    state_before = env.physics.get_state().copy()
    frames_before = [frame.copy() for frame in env.frame_buffer]

    assert env.push_state() == 1
    env.step(env.sample_action(0.5))
    assert env.stack_depth == 1

    env.pop_state()

    assert env.stack_depth == 0
    assert np.allclose(env.physics.get_state(), state_before)
    for restored, expected in zip(env.frame_buffer, frames_before):
        assert np.array_equal(restored, expected)


def test_pop_empty_stack_raises_clear_error() -> None:
    env = SmokeDetourEnv()
    with pytest.raises(RuntimeError, match="Rollback stack is empty"):
        env.pop_state()


def test_get_frame_tensor_shape_cpu() -> None:
    pytest.importorskip("torch")

    env = SmokeDetourEnv()
    env.reset()

    frames = env.get_frame_tensor(device="cpu")

    assert tuple(frames.shape) == (1, 8, 3, 384, 384)
    assert float(frames.min()) >= 0.0
    assert float(frames.max()) <= 1.0


def test_fixed_action_rollout_is_deterministic_after_pop() -> None:
    env = SmokeDetourEnv()
    env.reset()
    actions = [
        env.sample_action(0.2),
        env.sample_action(0.4),
        env.sample_action(-0.1),
        env.sample_action(0.3),
    ]

    env.push_state()
    first = [_step_signature(env, action) for action in actions]
    final_state_first = env.physics.get_state().copy()
    env.pop_state()
    second = [_step_signature(env, action) for action in actions]
    final_state_second = env.physics.get_state().copy()

    assert np.allclose(final_state_first, final_state_second)
    assert first == second


def test_nested_push_pop_is_deterministic() -> None:
    env = SmokeDetourEnv()
    env.reset()

    assert env.push_state() == 1
    env.step(env.sample_action(0.15))
    assert env.push_state() == 2

    nested_first = _step_signature(env, env.sample_action(0.25))
    env.pop_state()
    nested_second = _step_signature(env, env.sample_action(0.25))
    env.pop_state()

    assert env.stack_depth == 0
    assert nested_first == nested_second


def _step_signature(env: SmokeDetourEnv, action: np.ndarray) -> tuple:
    frame, reward, done, info = env.step(action)
    state = tuple(np.round(env.physics.get_state(), decimals=8))
    frame_sum = int(frame.sum())
    state_vector = tuple(np.round(info["state_vector"], decimals=8))
    return state, frame_sum, round(float(reward), 8), bool(done), state_vector
