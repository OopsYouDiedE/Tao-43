"""Task definitions and minimal smoke-test environments."""

from vjepa_gym.env.tasks.base_task import CognitiveLevel
from vjepa_gym.env.tasks.base_task import TASK_REGISTRY
from vjepa_gym.env.tasks.base_task import TaskSpec

__all__ = ["CognitiveLevel", "TaskSpec", "TASK_REGISTRY", "SmokeDetourEnv"]


def __getattr__(name: str):
    if name == "SmokeDetourEnv":
        from vjepa_gym.env.tasks.smoke_detour import SmokeDetourEnv

        return SmokeDetourEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
