"""Verify the first VJEPA-Gym environment milestone."""

from __future__ import annotations

import hashlib
import importlib.metadata
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _first_version(*packages: str) -> str:
    for package in packages:
        version = _version(package)
        if version != "not installed":
            return version
    return "not installed"


def _frame_checksum(frame: np.ndarray) -> str:
    return hashlib.sha256(frame.tobytes()).hexdigest()[:16]


def _rollout(env: SmokeDetourEnv, actions: list[np.ndarray]):
    frames = []
    rewards = []
    dones = []
    infos = []
    for action in actions:
        frame, reward, done, info = env.step(action)
        frames.append(frame)
        rewards.append(reward)
        dones.append(done)
        infos.append(info)
    return {
        "state": env.physics.get_state().copy(),
        "frame_checksum": _frame_checksum(frames[-1]),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "dones": np.asarray(dones, dtype=bool),
        "infos": infos,
    }


def verify_determinism() -> bool:
    from vjepa_gym.env.tasks import SmokeDetourEnv

    env = SmokeDetourEnv()
    obs = env.reset()
    if obs.shape != (384, 384, 3):
        raise AssertionError(f"Unexpected render shape: {obs.shape}")

    actions = [
        env.sample_action(0.15),
        env.sample_action(0.35),
        env.sample_action(-0.10),
        env.sample_action(0.20),
    ]

    first_depth = env.push_state()
    nested_depth = env.push_state()
    nested = _rollout(env, [env.sample_action(0.05)])
    env.pop_state()
    repeat_nested = _rollout(env, [env.sample_action(0.05)])
    env.pop_state()

    if first_depth != 1 or nested_depth != 2 or env.stack_depth != 0:
        raise AssertionError("Nested push/pop stack depth is incorrect.")
    if not np.allclose(nested["state"], repeat_nested["state"]):
        raise AssertionError("Nested rollback did not restore deterministic state.")
    if nested["frame_checksum"] != repeat_nested["frame_checksum"]:
        raise AssertionError("Nested rollback did not restore deterministic frame history.")

    env.push_state()
    first = _rollout(env, actions)
    env.pop_state()
    second = _rollout(env, actions)

    state_equal = np.allclose(first["state"], second["state"])
    frames_equal = first["frame_checksum"] == second["frame_checksum"]
    rewards_equal = np.allclose(first["rewards"], second["rewards"])
    dones_equal = np.array_equal(first["dones"], second["dones"])
    return bool(state_equal and frames_equal and rewards_equal and dones_equal)


def main() -> int:
    print("VJEPA-Gym environment verification")
    print(f"mujoco: {_version('mujoco')}")
    print(f"dm_control: {_first_version('dm_control', 'dm-control')}")
    print(f"torch: {_version('torch')}")

    try:
        from vjepa_gym.env.tasks import SmokeDetourEnv
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        print(f"missing dependency: {missing}")
        print("install project dependencies before running environment verification")
        return 2

    env = SmokeDetourEnv()
    obs = env.reset()
    try:
        tensor = env.get_frame_tensor(device="cpu")
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        print(f"missing dependency: {missing}")
        print("install project dependencies before running environment verification")
        return 2
    deterministic = verify_determinism()

    print("task: smoke_detour")
    print(f"render shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"frame tensor shape: {tuple(tensor.shape)}")
    print(f"rollback determinism: {deterministic}")
    return 0 if deterministic else 1


if __name__ == "__main__":
    raise SystemExit(main())
