"""Dataset records: class folders or manifest.csv, normalized to Clip rows."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from echobench.audio_io import SUPPORTED_SUFFIXES

ALIASES = {"reference": "target", "room": "tag"}


@dataclass
class Clip:
    rel: str
    abspath: Path
    tag: str
    target: float | None
    group: str | None
    split: str | None
    domain: str | None
    happenings: list[dict] | None = None
    target_t60: float | None = None
    target_distance: float | None = None


def discover(root: Path, layout: str) -> list[Clip]:
    if layout == "class_folders":
        return _from_folders(root)
    if layout == "csv":
        return _from_manifest(root)
    raise ValueError("layout must be class_folders or csv")


def summarize(clips: list[Clip]) -> dict:
    tags: dict[str, int] = {}
    for c in clips:
        tags[c.tag] = tags.get(c.tag, 0) + 1
    return {"n_items": len(clips), "tags": tags}


def _from_folders(root: Path) -> list[Clip]:
    if not root.is_dir():
        raise ValueError(f"Dataset folder missing: {root}")
    out: list[Clip] = []
    for tag_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for wav in sorted(tag_dir.rglob("*")):
            if wav.is_file() and wav.suffix.lower() in SUPPORTED_SUFFIXES:
                side = wav.with_suffix(".events.json")
                happenings = None
                if side.is_file():
                    try:
                        import json as _json

                        happenings = _json.loads(side.read_text(encoding="utf-8"))
                    except Exception:
                        happenings = None
                out.append(Clip(str(wav.relative_to(root).as_posix()), wav, tag_dir.name, None, None, None, None, happenings))
    _need(out, root)
    out.sort(key=lambda c: c.rel)
    return out


def _from_manifest(root: Path) -> list[Clip]:
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        raise ValueError(f"manifest.csv missing in {root}")
    out: list[Clip] = []
    with manifest.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            cols = {k.strip().lower(): (v or "").strip() for k, v in (row or {}).items() if k}
            for old, new in ALIASES.items():
                if old in cols and new not in cols:
                    cols[new] = cols[old]
            rel = cols.get("path") or cols.get("file") or ""
            if not rel:
                continue
            cand = (root / rel) if not Path(rel).is_absolute() else Path(rel)
            if not cand.is_file():
                # tolerate moved trees: try resolving against root by name
                alt = root / Path(rel).name
                cand = alt if alt.is_file() else cand
            tag = cols.get("label") or cols.get("tag") or ""
            tgt = _num(cols.get("target") or cols.get("value"))
            tgt_t60 = _num(cols.get("target_t60") or cols.get("t60_s") or cols.get("t60_seconds") or cols.get("t60"))
            tgt_dist = _num(cols.get("target_distance") or cols.get("distance_m") or cols.get("distance"))

            group = cols.get("condition") or cols.get("room") or None
            split = (cols.get("split") or None) or None
            domain = (cols.get("domain") or None) or None
            happenings = None
            if cols.get("events_json"):
                try:
                    happenings = json.loads(cols["events_json"])
                except Exception:
                    happenings = None
            events_path = cols.get("events") or ""
            if happenings is None and events_path:
                side = (root / events_path) if not Path(events_path).is_absolute() else Path(events_path)
                if side.is_file():
                    try:
                        happenings = json.loads(side.read_text(encoding="utf-8"))
                    except Exception:
                        happenings = None
            out.append(Clip(rel, cand, tag, tgt, group, split, domain, happenings, tgt_t60, tgt_dist))
    # attach sidecar .events.json files when present (synth + SED sets)
    for c in out:
        if c.happenings is None:
            side = c.abspath.with_suffix(".events.json")
            if side.is_file():
                try:
                    c.happenings = json.loads(side.read_text(encoding="utf-8"))
                except Exception:
                    c.happenings = None
    _need(out, root)
    out.sort(key=lambda c: c.rel)
    return out


def _num(text: str | None) -> float | None:
    if text is None or text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _need(clips: list[Clip], root: Path) -> None:
    if not clips:
        raise ValueError(f"No audio items found under {root}.")
