"""Range from the direct-sound arrival (classical, untrained).

Estimate = onset-sample / rate * 343 m/s (20 C convention). The label is the
straight-line source-to-microphone distance in the shoebox plan.
"""

from __future__ import annotations

import numpy as np

from echobench.models_api import BaseModel, ModelResult

SPEED_M_S = 343.0
ONSET_RATIO = 0.05


class OnsetRangeModel(BaseModel):
    model_id = "onset-range"
    display_name = "Onset range baseline"
    task = "distance"
    blurb = "First arrival over 5% of peak times 343 m/s. Classical baseline."

    def describe(self) -> dict:
        info = super().describe()
        info.update({"unit": "meters", "speed_m_s": SPEED_M_S, "onset_ratio": ONSET_RATIO})
        return info

    def infer(self, waveform, sample_rate: int) -> ModelResult:
        meters = arrival_range(waveform, sample_rate)
        return ModelResult(tag=f"{meters:.6f}", number=meters, detail={"unit": "meters"})


def arrival_range(waveform, sample_rate: int, ratio: float = ONSET_RATIO) -> float:
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")
    env = np.abs(np.asarray(waveform, dtype=np.float64).ravel())
    if env.size == 0:
        raise ValueError("Empty clip.")
    peak = float(env.max())
    if peak <= 0:
        raise ValueError("Silent clip.")
    first = int(np.argmax(env >= ratio * peak))
    return float(first) / float(sample_rate) * SPEED_M_S
