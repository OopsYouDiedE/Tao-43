"""Minimal dm_control task used to verify rollback infrastructure."""

from __future__ import annotations

from typing import Dict

import numpy as np
from dm_control import mujoco

from vjepa_gym.env.base_rollback_env import RollbackEnv


SMOKE_DETOUR_XML = """
<mujoco model="smoke_detour">
  <option timestep="0.02" integrator="Euler"/>
  <visual>
    <global offwidth="384" offheight="384"/>
  </visual>
  <default>
    <geom rgba="0.8 0.2 0.1 1"/>
  </default>
  <worldbody>
    <light name="key" pos="0 -3 3" dir="0 1 -1"/>
    <camera name="fixed" pos="0 -3 1.6" xyaxes="1 0 0 0 0.45 0.89"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.2 0.25 0.28 1"/>
    <geom name="barrier" type="box" pos="0 0 0.2" size="0.08 0.8 0.2" rgba="0.1 0.1 0.1 1"/>
    <body name="agent" pos="-0.8 0 0.08">
      <joint name="slide_x" type="slide" axis="1 0 0" damping="0.1"/>
      <geom name="agent_geom" type="sphere" size="0.08" rgba="0.1 0.6 0.9 1"/>
    </body>
    <body name="goal" pos="0.8 0 0.08">
      <geom name="goal_geom" type="sphere" size="0.06" rgba="0.2 0.9 0.25 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="slide_motor" joint="slide_x" gear="1" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


class SmokeDetourEnv(RollbackEnv):
    """A one-actuator environment for smoke tests and rollback verification."""

    action_dim = 1
    state_dim = 7

    def __init__(
        self,
        frame_buffer_len: int = 8,
        render_width: int = 384,
        render_height: int = 384,
    ) -> None:
        physics = mujoco.Physics.from_xml_string(SMOKE_DETOUR_XML)
        super().__init__(
            physics=physics,
            frame_buffer_len=frame_buffer_len,
            render_width=render_width,
            render_height=render_height,
            camera_id="fixed",
        )

    def sample_action(self, value: float = 0.25) -> np.ndarray:
        return np.array([value], dtype=np.float32)

    def get_state_vector(self) -> np.ndarray:
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[0] = float(self.physics.named.data.qpos["slide_x"])
        state[3] = float(self.physics.named.data.qvel["slide_x"])
        return state

    def _apply_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != self.action_dim:
            raise ValueError(f"Expected action shape ({self.action_dim},), got {action.shape}.")
        self.physics.data.ctrl[0] = float(np.clip(action[0], -1.0, 1.0))

    def _compute_reward(self) -> float:
        x = float(self.physics.named.data.qpos["slide_x"])
        return max(0.0, 1.0 - abs(0.8 - x))

    def _check_done(self) -> bool:
        x = float(self.physics.named.data.qpos["slide_x"])
        return abs(0.8 - x) < 0.05

    def _get_info(self) -> Dict:
        return {
            "task": "smoke_detour",
            "qpos": self.physics.get_state().copy(),
            "state_vector": self.get_state_vector(),
        }
