"""Environment utilities for VJEPA-Gym."""

__all__ = ["RollbackEnv"]


def __getattr__(name: str):
    if name == "RollbackEnv":
        from vjepa_gym.env.base_rollback_env import RollbackEnv

        return RollbackEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
