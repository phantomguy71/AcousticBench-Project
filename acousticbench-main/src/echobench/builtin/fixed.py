"""Always-vote baseline to confirm the metrics react."""

from __future__ import annotations

from echobench.models_api import BaseModel, ModelResult


class FixedTagModel(BaseModel):
    model_id = "fixed-tag"
    display_name = "Fixed-tag baseline"
    task = "classes"
    blurb = "Predicts one configured tag for every clip; checks that scoring moves."

    def __init__(self, tag: str = "noise", **_kw: object) -> None:
        clean = str(tag).strip()
        if not clean:
            raise ValueError("Fixed-tag baseline needs a non-empty tag.")
        self.tag = clean

    def infer(self, waveform, sample_rate: int) -> ModelResult:  # noqa: ANN001
        _ = (waveform, sample_rate)
        return ModelResult(tag=self.tag, confidences={self.tag: 1.0})
