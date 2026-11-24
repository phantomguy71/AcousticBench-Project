"""Room-family vote from a Schroeder T60.

Bands sit in the gaps of the synthetic room plan (office at or below 0.47 s,
meeting from 0.58 s, lecture from 0.95 s): below 0.54 s -> office, below
0.84 s -> meeting, else lecture. Fixed before seeing ACE scores; no lobby
output.
"""

from __future__ import annotations

from echobench.builtin.t60 import decay_t60
from echobench.models_api import BaseModel, ModelResult

OFFICE_BELOW_S = 0.54
MEETING_BELOW_S = 0.84


class DecayRoomRule(BaseModel):
    model_id = "decay-room-rule"
    display_name = "Decay room rule"
    task = "room"
    blurb = "Maps a Schroeder T60 through fixed bands to office/meeting/lecture. Untrained; no lobby class."

    def describe(self) -> dict:
        info = super().describe()
        info.update({"bands_s": [OFFICE_BELOW_S, MEETING_BELOW_S], "outputs": ["office", "meeting", "lecture"]})
        return info

    def infer(self, waveform, sample_rate: int) -> ModelResult:
        t60 = decay_t60(waveform, sample_rate)
        tag = "office" if t60 < OFFICE_BELOW_S else ("meeting" if t60 < MEETING_BELOW_S else "lecture")
        return ModelResult(tag=tag, detail={"t60_s": t60, "bands_s": [OFFICE_BELOW_S, MEETING_BELOW_S]})
