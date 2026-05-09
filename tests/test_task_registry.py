from vjepa_gym.env.tasks.base_task import CognitiveLevel
from vjepa_gym.env.tasks.base_task import TASK_REGISTRY


def test_smoke_detour_registry_metadata() -> None:
    spec = TASK_REGISTRY["smoke_detour"]

    assert spec.cognitive_level is CognitiveLevel.SINGLE_CONTACT
    assert spec.horizon == 20
    assert spec.mjcf_path == "inline://smoke_detour"
