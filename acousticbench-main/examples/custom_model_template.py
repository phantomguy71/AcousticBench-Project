"""Template for a team model. Copy to user_models/my_model.py and edit."""

from __future__ import annotations

from echobench.models_api import BaseModel, ModelResult


class MyModel(BaseModel):
    """Replace this stub with real inference. Keep the signature."""

    model_id = "my-model"
    display_name = "My model"
    task = "room"  # room | classes | t60 | distance | events
    blurb = "Describe the checkpoint and its training data here."

    def __init__(self, threshold: float = 0.5, **_kw: object) -> None:
        self.threshold = float(threshold)

    def startup(self) -> None:
        return None

    def infer(self, waveform, sample_rate: int) -> ModelResult:
        # Example: always vote for the first class. Replace with real code.
        _ = (waveform, sample_rate, self.threshold)
        return ModelResult(tag="office", confidences={"office": 1.0})
