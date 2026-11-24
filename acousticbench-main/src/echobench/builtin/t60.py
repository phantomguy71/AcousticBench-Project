"""Schroeder-decay T60 estimate (classical baseline, not a neural net).

Method: locate the direct peak, integrate energy backwards, fit a line to the
log-decay between two levels, extrapolate to 60 dB. Reference: M. R. Schroeder,
"New Method of Measuring Reverberation Time", JASA, 1965.
"""

from __future__ import annotations

import numpy as np

from echobench.models_api import BaseModel, ModelResult


class SchroederT60(BaseModel):
    model_id = "schroeder-t60"
    display_name = "Schroeder decay T60"
    task = "t60"
    blurb = (
        "Classical T60 from Schroeder backward integration with a linear decay fit. "
        "Sanity-check baseline so the harness runs before sponsor weights arrive."
    )

    def __init__(self, top_db: float = -6.0, bottom_db: float = -32.0, **_kw: object) -> None:
        if bottom_db >= top_db:
            raise ValueError("bottom_db must be more negative than top_db.")
        self.top_db = float(top_db)
        self.bottom_db = float(bottom_db)

    def describe(self) -> dict:
        info = super().describe()
        info.update({"top_db": self.top_db, "bottom_db": self.bottom_db, "unit": "seconds", "citation": "Schroeder, JASA 1965"})
        return info

    def infer(self, waveform, sample_rate: int) -> ModelResult:
        seconds = decay_t60(waveform, sample_rate, self.top_db, self.bottom_db)
        return ModelResult(tag=f"{seconds:.6f}", number=seconds, detail={"path": "schroeder", "unit": "seconds"})


def decay_t60(waveform, sample_rate: int, top_db: float = -6.0, bottom_db: float = -32.0) -> float:
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")
    sig = np.asarray(waveform, dtype=np.float64).ravel()
    if sig.size < sample_rate // 10:
        raise ValueError("Clip too short for a T60 fit.")
    peak = int(np.argmax(np.abs(sig)))
    tail = sig[peak:]
    energy = np.cumsum((tail**2)[::-1])[::-1]
    if energy[0] <= 0:
        raise ValueError("Impulse has no energy.")
    curve = 10.0 * np.log10(energy / energy[0] + 1e-12)
    times = np.arange(curve.size, dtype=np.float64) / float(sample_rate)
    sel = (curve <= top_db) & (curve >= bottom_db)
    if int(sel.sum()) < 8:
        sel = (curve <= -6.0) & (curve >= -28.0)
    if int(sel.sum()) < 5:
        raise ValueError("Decay curve too short for a line fit.")
    slope = float(np.polyfit(times[sel], curve[sel], 1)[0])
    if slope >= -1e-6:
        raise ValueError("Decay slope is not negative.")
    return float(-60.0 / slope)
