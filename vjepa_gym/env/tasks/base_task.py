"""Task metadata shared by VJEPA-Gym environments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class CognitiveLevel(IntEnum):
    """Cognitive complexity level used by the curriculum."""

    SINGLE_CONTACT = 1
    MULTI_STEP = 2
    TOOL_AS_MEDIUM = 3
    HIERARCHICAL = 4
    CROSS_MODAL = 5


@dataclass(frozen=True)
class TaskSpec:
    name: str
    cognitive_level: CognitiveLevel
    animal_prototype: str
    cognitive_ability: str
    horizon: int
    success_threshold: float
    mjcf_path: str


TASK_REGISTRY: dict[str, TaskSpec] = {
    "smoke_detour": TaskSpec(
        name="Smoke Detour",
        cognitive_level=CognitiveLevel.SINGLE_CONTACT,
        animal_prototype="infrastructure smoke test",
        cognitive_ability="rollback determinism and RGB observation plumbing",
        horizon=20,
        success_threshold=0.95,
        mjcf_path="inline://smoke_detour",
    ),
    "detour": TaskSpec(
        name="障碍绕行",
        cognitive_level=CognitiveLevel.MULTI_STEP,
        animal_prototype="章鱼/乌鸦",
        cognitive_ability="空间抑制 + detour problem + 路径规划",
        horizon=35,
        success_threshold=0.8,
        mjcf_path="tasks/mjcf/detour.xml",
    ),
}
