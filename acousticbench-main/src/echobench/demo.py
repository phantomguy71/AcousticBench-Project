"""One-command demo: score every built-in baseline and rewrite the reports."""

from __future__ import annotations

from pathlib import Path

from echobench.comparison import compare_pair
from echobench.engine import run_one
from echobench.reporting import compare_markdown, draft_markdown, run_markdown
from echobench.service import queue_eval
from echobench.storage import Store

SIM_ROOMS = "synth-rooms-dev"
REAL_ACE = "ace-single-rir"


def run_demo(root: Path, store: Store) -> dict[str, str]:
    for need in (SIM_ROOMS, REAL_ACE):
        try:
            store.fetch_dataset(need)
        except KeyError as exc:
            raise ValueError(f"Dataset '{need}' missing. Build synth rooms and import ACE first.") from exc
    t60_sim = _go(store, root, "schroeder-t60", SIM_ROOMS, "Synth rooms; Sabine reference.", 30)
    t60_real = _go(store, root, "schroeder-t60", REAL_ACE, "ACE single-channel; published fullband T60.", 30)
    room_sim = _go(store, root, "decay-room-rule", SIM_ROOMS, "Room family from the shoebox plan.", 30)
    room_real = _go(store, root, "decay-room-rule", REAL_ACE, "Room family from the ACE room name.", 30)
    span = _go(store, root, "onset-range", SIM_ROOMS, "Straight-line shoebox distance.", 30)
    events = _go(store, root, "energy-events", "demo-events", "Demo knock clips; onset tolerance 0.2 s.", 10)
    t60_c = compare_pair(t60_sim, t60_real)
    room_c = compare_pair(room_sim, room_real)
    out = root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "t60": out / "t60-synth-vs-ace.md",
        "room": out / "room-synth-vs-ace.md",
        "range": out / "range-synth.md",
        "events": out / "events-demo.md",
        "draft": out / "draft-report.md",
    }
    files["t60"].write_text(compare_markdown(t60_c, t60_sim, t60_real), encoding="utf-8")
    files["room"].write_text(compare_markdown(room_c, room_sim, room_real), encoding="utf-8")
    files["range"].write_text(run_markdown(span), encoding="utf-8")
    files["events"].write_text(run_markdown(events), encoding="utf-8")
    files["draft"].write_text(draft_markdown(t60_comp=t60_c, t60_sim=t60_sim, t60_real=t60_real, room_comp=room_c,
                                             room_sim=room_sim, room_real=room_real, range_run=span, events_run=events),
                              encoding="utf-8")
    return {k: str(v) for k, v in files.items()}


def _go(store: Store, root: Path, model: str, dataset: str, notes: str, cap: float) -> dict:
    queued = queue_eval(store, root, model_id=model, dataset_id=dataset,
                        raw={"sample_rate": 16000, "max_seconds": cap, "notes": notes})
    done = run_one(store, queued["id"], root)
    if done.get("status") != "completed":
        raise RuntimeError(done.get("error") or f"{model} on {dataset} failed.")
    return done
