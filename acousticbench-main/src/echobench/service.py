"""Validation + queueing shared by the CLI and the web server."""

from __future__ import annotations

import platform
import sqlite3
import sys
from pathlib import Path

from echobench import __version__
from echobench.datasets import Clip, discover, summarize
from echobench.plugins import build, find_entry, list_entries, load_class
from echobench.models_api import BaseModel
from echobench.storage import Store, fresh_id

DOMAINS = {"simulated", "real"}
LAYOUTS = {"class_folders", "csv"}


def clean_config(raw: dict | None) -> dict:
    raw = dict(raw or {})
    rate = int(raw.get("sample_rate") or 16000)
    if rate < 8000 or rate > 48000:
        raise ValueError("sample_rate must be within 8000..48000.")
    split = str(raw.get("split") or "all").strip().lower()
    if split in {"valid", "validation"}:
        split = "val"
    if split not in {"all", "train", "val", "test", "dev", "corpus"}:
        raise ValueError("split must be all, train, val, test, dev, or corpus.")
    cap = raw.get("max_items")
    cap = None if cap in ("", None) else int(cap)
    if cap is not None and cap < 1:
        raise ValueError("max_items must be >= 1.")
    secs = raw.get("max_seconds", 30)
    secs = None if secs in ("", None) else float(secs)
    if secs is not None and (secs <= 0 or secs > 120):
        raise ValueError("max_seconds must be within (0, 120].")
    seed = raw.get("seed", 11)
    seed = None if seed in ("", None) else int(seed)
    params = raw.get("model_params") or {}
    if not isinstance(params, dict):
        raise ValueError("model_params must be a mapping.")
    tol = float(raw.get("event_tolerance_s") or 0.25)
    docker_image = str(raw.get("docker_image") or "").strip()
    isolation = "docker" if docker_image else "direct"
    return {
        "sample_rate": rate, "split": split, "max_items": cap, "max_seconds": secs,
        "seed": seed, "notes": str(raw.get("notes") or "").strip(), "model_params": params,
        "paired_group": raw.get("paired_group"), "event_tolerance_s": tol,
        "docker_image": docker_image, "isolation": isolation,
    }


def clips_for(dataset: dict, config: dict) -> list[Clip]:
    clips = discover(Path(dataset["root_path"]), dataset["layout"])
    want = (config or {}).get("split") or "all"
    if want != "all":
        clips = [c for c in clips if (c.split or "").lower() in _aliases(want)]
    cap = (config or {}).get("max_items")
    return clips[: int(cap)] if cap else clips


def inventory(dataset: dict) -> dict:
    payload = dict(dataset)
    try:
        clips = discover(Path(dataset["root_path"]), dataset["layout"])
        payload["inventory"] = summarize(clips)
        payload["inventory_error"] = None
    except Exception as exc:
        payload["inventory"] = None
        payload["inventory_error"] = str(exc)
    return payload


def register_dataset(store: Store, fields: dict, *, allow_flags: bool = False) -> dict:
    name = str(fields.get("name") or "").strip()
    domain = str(fields.get("domain") or "").strip().lower()
    layout = str(fields.get("layout") or "").strip()
    where = Path(str(fields.get("root_path") or "")).expanduser()
    note = str(fields.get("description") or "").strip()
    if not name:
        raise ValueError("Dataset name is required.")
    if domain not in DOMAINS:
        raise ValueError("domain must be simulated or real.")
    if layout not in LAYOUTS:
        raise ValueError("layout must be class_folders or csv.")
    if not where.exists():
        raise ValueError(f"Dataset path does not exist: {where}")
    discover(where, layout)
    taken = {d["id"] for d in store.all_datasets()}
    key = str(fields.get("id") or "").strip() or fresh_id(name, taken)
    rec = {"id": key, "name": name, "domain": domain, "root_path": str(where.resolve()),
           "layout": layout, "description": note,
           "is_example": bool(fields.get("is_example")) if allow_flags else False,
           "synthetic_stand_in": bool(fields.get("synthetic_stand_in")) if allow_flags else False,
           "metadata": fields.get("metadata") or {}}
    if not allow_flags:
        rec["metadata"] = {**rec["metadata"], "origin": "user"}
    try:
        return store.save_dataset(rec)
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"A dataset named '{name}' already exists.") from exc


def register_model(store: Store, root: Path, fields: dict) -> dict:
    key = str(fields.get("id") or "").strip()
    title = str(fields.get("name") or key).strip()
    dotted = str(fields.get("qualified_name") or "").strip()
    blurb = str(fields.get("description") or "").strip()
    params = fields.get("params") or {}
    if not key or not title:
        raise ValueError("Model id and name are required.")
    if not isinstance(params, dict):
        raise ValueError("params must be a mapping.")
    known = {e.key for e in list_entries(root, store.extra_models())}
    if key in known and key not in store.extra_models_dict():
        raise ValueError(f"Model id '{key}' is taken by a built-in or config model.")
    cls = load_class(dotted)
    if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        raise ValueError(f"{dotted} must extend BaseModel.")
    try:
        inst = cls(**params)
    except TypeError as exc:
        raise ValueError(f"Cannot construct {cls.__name__}: {exc}") from exc
    if not isinstance(inst, BaseModel):
        raise ValueError("Model must extend BaseModel.")
    return store.save_model({"id": key, "name": title, "qualified_name": dotted, "description": blurb, "params": params})


def queue_eval(store: Store, root: Path, *, model_id: str, dataset_id: str, raw: dict | None) -> dict:
    cfg = clean_config(raw)
    try:
        ds = store.fetch_dataset(dataset_id)
    except KeyError as exc:
        raise ValueError(exc.args[0] if exc.args else "Unknown dataset.") from exc
    ent = find_entry(root, model_id, store.extra_models())
    model = build(ent, cfg.pop("model_params"))
    notes = cfg.pop("notes")
    cfg["task"] = getattr(model, "task", "room")
    cfg["dataset_flags"] = {"is_example": ds["is_example"], "synthetic_stand_in": ds["synthetic_stand_in"],
                            "synthetic": bool((ds.get("metadata") or {}).get("synthetic"))}
    prove = {"echobench": __version__, "python": sys.version.split()[0], "platform": platform.platform()}
    return store.start_run({"model_id": ent.key, "model_name": ent.title, "model_path": ent.dotted,
                            "model_params": dict(getattr(model, "__dict__", {})),
                            "dataset_id": ds["id"], "dataset_name": ds["name"], "dataset_domain": ds["domain"],
                            "config": cfg, "notes": notes, "provenance": prove})


def _aliases(split: str) -> set[str]:
    return {"val", "valid", "validation"} if split == "val" else {split}
