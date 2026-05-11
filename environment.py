from __future__ import annotations

import copy
import hashlib
from collections import deque
from typing import Deque

import numpy as np
from dm_control import mujoco


SMOKE_DETOUR_XML = """
<mujoco model="vjepa_gym_ball_detour">
  <option timestep="0.02" integrator="Euler"/>
  <visual>
    <global offwidth="384" offheight="384"/>
  </visual>
  <default>
    <joint damping="0.35"/>
    <geom condim="3" friction="0.8 0.1 0.1"/>
  </default>
  <worldbody>
    <light name="key" pos="0 -3 4" dir="0 1 -1"/>
    <camera name="fixed" pos="0 -2.7 2.35" xyaxes="1 0 0 0 0.65 0.76"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.18 0.22 0.24 1"/>
    <geom name="left_bound" type="box" pos="-1.08 0 0.08" size="0.03 1.05 0.08" rgba="0.35 0.35 0.36 1"/>
    <geom name="right_bound" type="box" pos="1.08 0 0.08" size="0.03 1.05 0.08" rgba="0.35 0.35 0.36 1"/>
    <geom name="bottom_bound" type="box" pos="0 -1.08 0.08" size="1.08 0.03 0.08" rgba="0.35 0.35 0.36 1"/>
    <geom name="top_bound" type="box" pos="0 1.08 0.08" size="1.08 0.03 0.08" rgba="0.35 0.35 0.36 1"/>
    <geom name="wall" type="box" pos="0 0 0.12" size="0.07 0.48 0.12" rgba="0.04 0.04 0.05 1"/>
    <body name="agent" pos="-0.75 -0.55 0.08">
      <joint name="slide_x" type="slide" axis="1 0 0" limited="true" range="-0.25 1.65"/>
      <joint name="slide_y" type="slide" axis="0 1 0" limited="true" range="-0.45 1.45"/>
      <geom name="agent_geom" type="sphere" size="0.065" rgba="0.1 0.6 0.95 1"/>
    </body>
    <body name="goal" pos="0.75 0.55 0.08">
      <geom name="goal_geom" type="sphere" size="0.06" rgba="0.2 0.9 0.25 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="slide_x_motor" joint="slide_x" gear="4.0" ctrlrange="-1 1" ctrllimited="true"/>
    <motor name="slide_y_motor" joint="slide_y" gear="4.0" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


class RollbackMuJoCoEnv:
    action_dim = 7
    state_dim = 7
    start_xy = np.array([-0.75, -0.55], dtype=np.float32)
    goal_xy = np.array([0.75, 0.55], dtype=np.float32)
    goal_x = 0.75
    goal_y = 0.55

    def __init__(
        self,
        frame_buffer_len: int = 2,
        render_width: int = 384,
        render_height: int = 384,
        camera_id: int | str = "fixed",
    ) -> None:
        self.physics = mujoco.Physics.from_xml_string(SMOKE_DETOUR_XML)
        self.frame_buffer_len = frame_buffer_len
        self.render_width = render_width
        self.render_height = render_height
        self.camera_id = camera_id
        self.frame_buffer: Deque[np.ndarray] = deque(maxlen=frame_buffer_len)
        self._phys_stack: list[np.ndarray] = []
        self._ctrl_stack: list[np.ndarray] = []
        self._frame_stack: list[Deque[np.ndarray]] = []

    def push_state(self) -> int:
        self._phys_stack.append(self.physics.get_state().copy())
        self._ctrl_stack.append(self.physics.data.ctrl.copy())
        self._frame_stack.append(copy.deepcopy(self.frame_buffer))
        return len(self._phys_stack)

    def pop_state(self) -> None:
        if not self._phys_stack:
            raise RuntimeError("回滚栈为空；无法调用 pop_state()。")
        state = self._phys_stack.pop()
        ctrl = self._ctrl_stack.pop()
        frames = self._frame_stack.pop()
        with self.physics.reset_context():
            self.physics.set_state(state)
            self.physics.data.ctrl[:] = ctrl
        self.physics.forward()
        self.frame_buffer = copy.deepcopy(frames)

    def clear_stack(self) -> None:
        self._phys_stack.clear()
        self._ctrl_stack.clear()
        self._frame_stack.clear()

    @property
    def stack_depth(self) -> int:
        return len(self._phys_stack)

    def reset(self) -> np.ndarray:
        self.physics.reset()
        self.physics.forward()
        self.clear_stack()
        self.frame_buffer.clear()
        frame = self.render()
        for _ in range(self.frame_buffer_len):
            self.frame_buffer.append(frame.copy())
        return frame

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (self.action_dim,):
            raise ValueError(f"期望 7D 动作，实际得到 {action.shape}。")
        action = np.clip(action, -1.0, 1.0)
        self.physics.data.ctrl[0] = float(action[0])
        self.physics.data.ctrl[1] = float(action[1])
        self.physics.step()
        frame = self.render()
        self.frame_buffer.append(frame.copy())
        reward = self.compute_reward()
        done = reward >= 0.95
        return frame, reward, done, {"state_vector": self.get_state_vector()}

    def step_macro(self, action: np.ndarray, repeats: int) -> tuple[np.ndarray, float, bool, dict]:
        reward = self.compute_reward()
        done = False
        frame = self.render()
        for _ in range(max(1, repeats)):
            frame, reward, done, info = self.step(action)
            if done:
                break
        return frame, reward, done, info

    def render(self) -> np.ndarray:
        return self.physics.render(
            height=self.render_height,
            width=self.render_width,
            camera_id=self.camera_id,
        )

    def get_state_vector(self) -> np.ndarray:
        return np.array([
            self.get_agent_x(),
            self.get_agent_y(),
            0.08,
            self.physics.named.data.qvel["slide_x"],
            self.physics.named.data.qvel["slide_y"],
            0.0,
            1.0
        ], dtype=np.float32)

    def get_agent_x(self) -> float:
        return float(self.start_xy[0] + self.physics.named.data.qpos["slide_x"])

    def get_agent_y(self) -> float:
        return float(self.start_xy[1] + self.physics.named.data.qpos["slide_y"])

    def get_agent_xy(self) -> np.ndarray:
        return np.array([self.get_agent_x(), self.get_agent_y()], dtype=np.float32)

    def set_agent_xy(self, xy: np.ndarray) -> None:
        xy = np.asarray(xy, dtype=np.float32).reshape(2)
        qpos = xy - self.start_xy
        with self.physics.reset_context():
            self.physics.named.data.qpos["slide_x"] = float(qpos[0])
            self.physics.named.data.qpos["slide_y"] = float(qpos[1])
            self.physics.named.data.qvel["slide_x"] = 0.0
            self.physics.named.data.qvel["slide_y"] = 0.0
            self.physics.data.ctrl[:] = 0.0
        self.physics.forward()
        self.frame_buffer.clear()
        frame = self.render()
        for _ in range(self.frame_buffer_len):
            self.frame_buffer.append(frame.copy())

    def distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal_xy - self.get_agent_xy()))

    def snapshot_signature(self) -> tuple:
        state = tuple(np.round(self.physics.get_state(), decimals=10))
        frame_hashes = tuple(_frame_checksum(frame) for frame in self.frame_buffer)
        return (
            state,
            tuple(np.round(self.physics.data.ctrl.copy(), decimals=10)),
            frame_hashes,
            round(self.get_agent_x(), 10),
            round(self.get_agent_y(), 10),
            round(self.compute_reward(), 10),
            self.stack_depth,
        )

    def get_frame_tensor(self, device: str | None = None):
        import torch

        if len(self.frame_buffer) != self.frame_buffer_len:
            raise RuntimeError("帧缓冲区未满；请先调用 reset()。")
        frames = np.stack(list(self.frame_buffer), axis=0)
        frames = frames.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        tensor = torch.from_numpy(frames).unsqueeze(0)
        return tensor.to(device) if device else tensor

    def get_macro_context_tensor(self, action_repeat: int, device: str | None = None):
        import torch

        if len(self.frame_buffer) != self.frame_buffer_len:
            raise RuntimeError("帧缓冲区未满；请先调用 reset()。")
        frames = np.stack([self.frame_buffer[0], self.frame_buffer[-1]], axis=0)
        frames = frames.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        tensor = torch.from_numpy(frames).unsqueeze(0)
        return tensor.to(device) if device else tensor

    def sample_action(self, x_delta: float = 0.25, y_delta: float = 0.0) -> np.ndarray:
        return np.array([x_delta, y_delta, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def rollout_sequence(self, actions: np.ndarray, record_frames: bool = False) -> dict:
        frames: list[np.ndarray] = []
        reward = self.compute_reward()
        done = False
        for action in np.asarray(actions, dtype=np.float32):
            frame, reward, done, _ = self.step(action)
            if record_frames:
                frames.append(frame.copy())
        return {
            "x": self.get_agent_x(),
            "y": self.get_agent_y(),
            "reward": float(reward),
            "done": bool(done),
            "distance_to_goal": self.distance_to_goal(),
            "frames": frames,
        }

    def compute_reward(self) -> float:
        max_distance = float(np.linalg.norm(self.goal_xy - self.start_xy))
        return max(0.0, 1.0 - self.distance_to_goal() / max_distance)


def make_smoke_env(config: dict | None = None) -> RollbackMuJoCoEnv:
    config = config or {}
    return RollbackMuJoCoEnv(
        frame_buffer_len=int(config.get("frame_buffer_len", 2)),
        render_width=int(config.get("render_width", 384)),
        render_height=int(config.get("render_height", 384)),
        camera_id=config.get("camera_id", "fixed"),
    )


def _frame_checksum(frame: np.ndarray) -> str:
    return hashlib.sha256(frame.tobytes()).hexdigest()[:16]
