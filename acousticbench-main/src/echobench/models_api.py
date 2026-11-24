"""Model contract. Every model sees mono audio at the run's target rate."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelResult:
    """What a model returns for one clip.

    tag: predicted class name (room / sound class). For regression tasks the
      numeric answer lives in `number` and `tag` is a formatted copy.
    number: seconds for t60, meters for distance, else None.
    happenings: event list for the `events` task: dicts with
      {label, onset_s, offset_s}.
    """

    tag: str = ""
    confidences: dict[str, float] | None = None
    number: float | None = None
    happenings: list[dict] | None = None
    detail: dict | None = field(default=None)


class BaseModel:
    model_id = "base"
    display_name = "Base model"
    task = "room"  # room | classes | t60 | distance | events
    blurb = ""

    def startup(self) -> None:
        return None

    def describe(self) -> dict:
        return {
            "id": self.model_id,
            "name": self.display_name,
            "task": self.task,
            "description": self.blurb,
        }

    def infer(self, waveform, sample_rate: int) -> ModelResult:  # noqa: ANN001
        raise NotImplementedError
