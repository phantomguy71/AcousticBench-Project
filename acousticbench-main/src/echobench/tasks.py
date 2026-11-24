"""Task catalogue: which metrics and columns each task needs."""

from __future__ import annotations

TASKS = {
    "room": {"label": "Room-size classification", "needs": "tag", "unit": ""},
    "classes": {"label": "Sound-class classification", "needs": "tag", "unit": ""},
    "t60": {"label": "Reverberation-time estimation", "needs": "target", "unit": "seconds"},
    "distance": {"label": "Source-to-microphone distance", "needs": "target", "unit": "meters"},
    "events": {"label": "Sound-event detection", "needs": "happenings", "unit": "spans"},
}

CLASS_TASKS = {"room", "classes"}
NUMBER_TASKS = {"t60", "distance"}


def check_task(name: str) -> str:
    if name not in TASKS:
        raise ValueError(f"Unknown task '{name}'. Expected one of {sorted(TASKS)}.")
    return name


def task_of_model(model) -> str:  # noqa: ANN001
    return check_task(getattr(model, "task", "room"))
