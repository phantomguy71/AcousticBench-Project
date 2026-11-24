"""Execute one queued run: load clips, call the model, score, persist."""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

from echobench.audio_io import read_mono_wav, to_rate
from echobench.datasets import Clip
from echobench.models_api import ModelResult
from echobench.paths import runs_root
from echobench.plugins import build, find_entry
from echobench.scoring import calibration_from_flags, classify_metrics, detect_metrics, regress_metrics
from echobench.service import clips_for
from echobench.storage import Store
from echobench.tasks import NUMBER_TASKS, task_of_model
from echobench.util import safe_json, seed_everything

log = logging.getLogger(__name__)


def run_one(store: Store, rid: str, root: Path) -> dict:
    run = store.fetch_run(rid)
    store.flag_running(rid)
    try:
        summary, pred_path = _score(store, run, root)
    except Exception as exc:  # noqa: BLE001
        log.exception("run %s failed", rid)
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        return store.close_run(rid, status="failed", error=str(msg))
    return store.close_run(rid, status="completed", summary=safe_json(summary), predictions_path=str(pred_path))


def _score(store: Store, run: dict, root: Path) -> tuple[dict, Path]:
    # Midterm §Expected Evaluation Algorithm, step 1: read + validate config.
    ds = store.fetch_dataset(run["dataset_id"])
    cfg = run.get("config") or {}
    seed_everything(cfg.get("seed"))
    # Step 2: load the dataset into the standardized Clip rows.
    clips = clips_for(ds, cfg)
    if not clips:
        raise RuntimeError(f"No clips matched split '{cfg.get('split')}'.")
    # Step 3 happens per clip below (validate labels/targets/spans).
    # Step 4: invoke the model through the plugin contract.
    model = build(find_entry(root, run["model_id"], store.extra_models()), run.get("model_params") or {})
    if hasattr(model, "startup"):
        model.startup()
    task = task_of_model(model)
    # Isolation is recorded, never assumed: 'direct' unless queued with a docker image.
    isolation = (run.get("config") or {}).get("isolation", "direct")
    rate = int(cfg.get("sample_rate") or 16000)
    cap_s = cfg.get("max_seconds")
    tol = float(cfg.get("event_tolerance_s") or 0.25)

    rows: list[dict] = []
    tags_t, tags_p, confs, lats, groups = [], [], [], [], []
    nums_t, nums_p = [], []
    ev_t: list[list[tuple[float, float]]] = []
    ev_p: list[list[tuple[float, float]]] = []
    item_scores: list[float] = []
    per_pred, per_true = [], []
    dom_mix = False
    for clip in clips:
        want_target = _want_target(clip, task)
        rec = {"path": clip.rel, "true_tag": clip.tag, "pred_tag": "", "true_value": _f(want_target),
               "pred_value": "", "group": clip.group or "", "confidence": "", "latency_ms": "", "rate": "", "error": ""}
        if clip.domain and clip.domain != ds["domain"]:
            dom_mix = True
        try:
            wav, src = read_mono_wav(clip.abspath)
            wav = to_rate(wav, src, rate)
            if cap_s:
                wav = wav[: int(float(cap_s) * rate)]
            if wav.size == 0:
                raise ValueError("Clip empty after load.")
            t0 = time.perf_counter()
            out = model.infer(wav, rate)
            dt = (time.perf_counter() - t0) * 1000.0
            parsed = _read_out(out, task)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = str(exc)
            rows.append(rec)
            continue
        rec.update({"pred_tag": parsed["tag"], "pred_value": _f(parsed["number"]),
                    "confidence": _f(parsed["confidence"]), "latency_ms": f"{dt:.3f}", "rate": str(src)})
        rows.append(rec)
        tags_t.append(clip.tag)
        tags_p.append(parsed["tag"])
        confs.append(parsed["confidence"])
        lats.append(dt)
        groups.append(clip.group)
        per_pred.append(parsed["tag"])
        per_true.append(clip.tag)
        if task in NUMBER_TASKS:
            if want_target is None:
                raise RuntimeError(f"Clip {clip.rel} has no numeric target for task '{task}'.")
            nums_t.append(float(want_target))
            nums_p.append(float(parsed["number"]))
            item_scores.append(-abs(float(parsed["number"]) - float(want_target)))
        elif task == "events":
            want = [(float(e.get("onset_s", 0.0)), float(e.get("offset_s", 0.0))) for e in (clip.happenings or [])]
            got = [(float(e.get("onset_s", 0.0)), float(e.get("offset_s", 0.0))) for e in (parsed["events"] or [])]
            ev_t.append(want)
            ev_p.append(got)
            item_scores.append(_event_clip_f1(want, got, tol))
        else:
            item_scores.append(1.0 if parsed["tag"] == clip.tag else 0.0)

    ok_rows = [r for r in rows if not r["error"]]
    if task in NUMBER_TASKS and not nums_t:
        raise RuntimeError(f"No clips scored. First error: {_first(rows)}")
    if task not in NUMBER_TASKS and task != "events" and not ok_rows:
        raise RuntimeError(f"No clips scored. First error: {_first(rows)}")
    # Step 5: task-appropriate metrics.
    if task in NUMBER_TASKS:
        summary = regress_metrics(nums_t, nums_p, lats, groups, task=task, unit="seconds" if task == "t60" else "meters")
    elif task == "events":
        summary = detect_metrics(ev_t, ev_p, tol, lats)
    else:
        summary = classify_metrics(tags_t, tags_p, confs, lats)
        summary["task"] = task
    # Steps 6-7: persist scores + provenance; the report layer renders step 7.
    # Step 8 (regressions) lives in reporting.regression_alerts over run history.
    summary["n_items"] = len(clips)
    summary["n_errors"] = len(clips) - len(ok_rows)
    summary["item_scores"] = item_scores
    summary["correctness"] = {"pred": per_pred, "true": per_true}
    summary["model_info"] = model.describe() if hasattr(model, "describe") else {}
    summary["isolation"] = isolation
    warns = []
    if dom_mix:
        warns.append("Some files carry a domain tag that differs from the dataset domain.")
    if summary.get("n_errors"):
        warns.append(f"{summary['n_errors']} clip(s) failed to score; see predictions CSV.")
    if task in {"room", "classes"} and all(c is None for c in confs):
        warns.append("Model returned no confidences, so calibration (ECE) is unavailable.")
    summary["warnings"] = warns
    dest = runs_root(root) / f"{run['id']}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    _ = calibration_from_flags  # keep import used for API stability
    return summary, dest


def _want_target(clip: Clip, task: str):
    if task == "t60":
        return clip.target_t60 if clip.target_t60 is not None else clip.target
    if task == "distance":
        return clip.target_distance if clip.target_distance is not None else clip.target
    return clip.target


def _read_out(out: ModelResult, task: str) -> dict:
    conf = None
    if out.confidences:
        try:
            conf = float(out.confidences.get(out.tag, max(out.confidences.values())))
        except (ValueError, TypeError):
            conf = None
    if task in NUMBER_TASKS:
        if out.number is None:
            raise ValueError(f"Task '{task}' needs ModelResult.number.")
        return {"tag": out.tag or f"{float(out.number):.6f}", "number": float(out.number), "confidence": conf, "events": []}
    if task == "events":
        return {"tag": out.tag, "number": None, "confidence": conf, "events": out.happenings or []}
    if not out.tag:
        raise ValueError("Classification result needs ModelResult.tag.")
    return {"tag": out.tag, "number": None, "confidence": conf, "events": []}


def _event_clip_f1(want, got, tol: float) -> float:
    used = [False] * len(want)
    hit = 0
    for p_on, _ in got:
        for i, (t_on, _t_off) in enumerate(want):
            if not used[i] and abs(p_on - t_on) <= tol:
                used[i] = True
                hit += 1
                break
    denom = len(want) + len(got)
    return (2 * hit / denom) if denom else 1.0


def _first(rows: list[dict]) -> str:
    for r in rows:
        if r["error"]:
            return r["error"]
    return "unknown error"


def _f(value) -> str:  # noqa: ANN001
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _clip_lookup(clips: list[Clip]) -> dict:  # pragma: no cover - helper
    return {c.rel: c for c in clips}
