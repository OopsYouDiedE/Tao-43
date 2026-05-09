"""Rollback-capable base environment built on top of dm_control Physics."""

from __future__ import annotations

from collections import deque
from typing import Deque
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from dm_control import mujoco as mjc


class RollbackEnv:
    """Base class for dm_control environments with physical and visual rollback.

    The stack stores both MuJoCo state and the rolling RGB frame buffer so a
    restored state has the same visual history that the encoder would consume.
    """

    def __init__(
        self,
        physics: mjc.Physics,
        frame_buffer_len: int = 8,
        render_width: int = 384,
        render_height: int = 384,
        camera_id: int | str = 0,
    ) -> None:
        self.physics = physics
        self.render_width = render_width
        self.render_height = render_height
        self.camera_id = camera_id
        self.frame_buffer_len = frame_buffer_len
        self.frame_buffer: Deque[np.ndarray] = deque(maxlen=frame_buffer_len)
        self._phys_stack: List[np.ndarray] = []
        self._frame_stack: List[List[np.ndarray]] = []

    def push_state(self) -> int:
        """Snapshot current physics state and frame history.

        Returns:
            The new stack depth, useful for callers doing nested validation.
        """
        self._phys_stack.append(self.physics.get_state().copy())
        self._frame_stack.append([frame.copy() for frame in self.frame_buffer])
        return len(self._phys_stack)

    def pop_state(self) -> None:
        """Restore the most recent snapshot and remove it from the stack."""
        if not self._phys_stack:
            raise RuntimeError("Rollback stack is empty; cannot pop_state().")

        state = self._phys_stack.pop()
        frame_history = self._frame_stack.pop()

        with self.physics.reset_context():
            self.physics.set_state(state)
        self.physics.forward()

        self.frame_buffer = deque(
            [frame.copy() for frame in frame_history],
            maxlen=self.frame_buffer_len,
        )

    def clear_stack(self) -> None:
        """Drop all saved rollback points."""
        self._phys_stack.clear()
        self._frame_stack.clear()

    @property
    def stack_depth(self) -> int:
        """Current rollback stack depth."""
        return len(self._phys_stack)

    def reset(self) -> np.ndarray:
        """Reset physics and fill the frame buffer with the initial frame."""
        self.physics.reset()
        self.physics.forward()
        self.frame_buffer.clear()
        self.clear_stack()

        first_frame = self._render()
        for _ in range(self.frame_buffer_len):
            self.frame_buffer.append(first_frame.copy())
        return first_frame

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Apply an action, advance physics, render, and update frame history."""
        self._apply_action(action)
        self.physics.step()

        frame = self._render()
        self.frame_buffer.append(frame.copy())

        reward = self._compute_reward()
        done = self._check_done()
        info = self._get_info()
        return frame, reward, done, info

    def get_frame_tensor(self, device: str = "cuda"):
        """Return buffered frames as ``[1, T, C, H, W]`` float tensor in [0, 1]."""
        import torch

        if len(self.frame_buffer) != self.frame_buffer_len:
            raise RuntimeError(
                f"Frame buffer has {len(self.frame_buffer)} frames; "
                f"expected {self.frame_buffer_len}. Did you call reset()?"
            )

        frames = np.stack(list(self.frame_buffer), axis=0)
        frames = frames.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        return torch.from_numpy(frames).unsqueeze(0).to(device)

    def get_state_vector(self) -> np.ndarray:
        """Return task-specific state vector, typically end-effector state."""
        raise NotImplementedError

    def _render(self) -> np.ndarray:
        """Render the current RGB frame as ``[H, W, 3]`` uint8."""
        return self.physics.render(
            height=self.render_height,
            width=self.render_width,
            camera_id=self.camera_id,
        )

    def _apply_action(self, action: np.ndarray) -> None:
        """Apply task-specific action to MuJoCo controls."""
        raise NotImplementedError

    def _compute_reward(self) -> float:
        return 0.0

    def _check_done(self) -> bool:
        return False

    def _get_info(self) -> Dict:
        return {}
