"""Energy-onset event detector (untrained baseline for the `events` task).

Finds bursts above an adaptive RMS floor, merges close neighbours, drops
flicker-short spans. Returns happenings as {label, onset_s, offset_s}.
Labels everything 'knock' — timing, not naming, is what this baseline tests.
"""

from __future__ import annotations

import numpy as np

from echobench.models_api import BaseModel, ModelResult


class EnergyEventDetector(BaseModel):
    model_id = "energy-events"
    display_name = "Energy event detector"
    task = "events"
    blurb = "Adaptive energyGate; merges gaps <120ms, drops spans <80ms. Untrained timing baseline."

    def __init__(self, floor_db: float = -26.0, merge_s: float = 0.15, min_s: float = 0.10, **_kw: object) -> None:
        self.floor_db = float(floor_db)
        self.merge_s = float(merge_s)
        self.min_s = float(min_s)

    def infer(self, waveform, sample_rate: int) -> ModelResult:
        spans = detect_spans(waveform, sample_rate, self.floor_db, self.merge_s, self.min_s)
        happenings = [{"label": "knock", "onset_s": a, "offset_s": b} for a, b in spans]
        return ModelResult(tag="knock" if happenings else "silence", happenings=happenings, detail={"spans": len(spans)})


def detect_spans(waveform, sample_rate: int, floor_db: float = -26.0, merge_s: float = 0.15, min_s: float = 0.10) -> list[tuple[float, float]]:
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")
    sig = np.asarray(waveform, dtype=np.float64).ravel()
    if sig.size == 0:
        raise ValueError("Empty clip.")
    win = max(1, sample_rate // 100)
    n = int(np.ceil(sig.size / win))
    pad = np.zeros(n * win)
    pad[: sig.size] = sig
    rms = np.sqrt((pad.reshape(n, win) ** 2).mean(axis=1) + 1e-12)
    peak = float(rms.max())
    if peak <= 0:
        return []
    thresh = peak * (10.0 ** (floor_db / 20.0))
    hot = rms >= thresh
    spans: list[tuple[float, float]] = []
    i = 0
    while i < n:
        if not hot[i]:
            i += 1
            continue
        j = i
        while j < n and hot[j]:
            j += 1
        spans.append((i * win / sample_rate, min(j * win / sample_rate, sig.size / sample_rate)))
        i = j
    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a - merged[-1][1] <= merge_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= min_s]
