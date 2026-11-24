"""Markdown / HTML / JSON reports, leaderboard, and regression checks."""

from __future__ import annotations

import html
import json

from echobench import ADVISOR, PROJECT_LABEL, SYNTH_CAVEAT, TEAM


def _fmt(v) -> str:  # noqa: ANN001
    if isinstance(v, bool) or v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def run_markdown(run: dict) -> str:
    s = run.get("summary") or {}
    parts = [f"# {PROJECT_LABEL} run report", "",
             f"- Team: {', '.join(TEAM)}", f"- Advisor: {ADVISOR}",
             f"- Run `{run['id']}` · status {run['status']}", f"- Created {run.get('created_at', '')}",
             f"- Model {run.get('model_name')} (`{run.get('model_id')}`)",
             f"- Dataset {run.get('dataset_name')} (`{run.get('dataset_domain')}`)",
             f"- Notes: {run.get('notes') or '—'}", ""]
    if _demo(run):
        parts += ["## Demo-data caveat", "", SYNTH_CAVEAT, ""]
    if run.get("error"):
        parts += ["## Error", "", run["error"], ""]
    if s:
        parts += ["## Metrics", ""]
        for k in ("accuracy", "macro_f1", "weighted_f1", "mae", "rmse", "median_ae", "event_f1", "precision", "recall",
                  "mean_latency_ms", "p50_latency_ms", "ece", "r2"):
            if s.get(k) is not None:
                parts.append(f"- {k}: {_fmt(s[k])}")
        parts.append("")
        if s.get("by_condition"):
            parts += ["## By condition", "", "| Group | n | MAE | RMSE |", "| --- | ---: | ---: | ---: |"]
            for g, row in sorted(s["by_condition"].items()):
                parts.append(f"| {g} | {row.get('n')} | {_fmt(row.get('mae'))} | {_fmt(row.get('rmse'))} |")
            parts.append("")
        if s.get("per_class"):
            parts += ["## Per class", "", "| Class | P | R | F1 | n |", "| --- | ---: | ---: | ---: | ---: |"]
            for c, row in sorted(s["per_class"].items()):
                parts.append(f"| {c} | {_fmt(row.get('precision'))} | {_fmt(row.get('recall'))} | {_fmt(row.get('f1'))} | {row.get('support')} |")
            parts.append("")
        if s.get("warnings"):
            parts += ["## Warnings", ""] + [f"- {w}" for w in s["warnings"]] + [""]
    parts += ["## Provenance", "", "```json", json.dumps(run.get("provenance") or {}, indent=2), "```", ""]
    return "\n".join(parts)


def compare_markdown(comp: dict, sim: dict, real: dict) -> str:
    parts = [f"# {PROJECT_LABEL} sim-to-real comparison", "",
             f"- Team: {', '.join(TEAM)}", f"- Advisor: {ADVISOR}",
             f"- {comp.get('gap_definition')}",
             f"- Simulated `{sim['id']}` {sim.get('dataset_name')} · {sim.get('model_name')}",
             f"- Real `{real['id']}` {real.get('dataset_name')} · {real.get('model_name')}", ""]
    if comp.get("warnings"):
        parts += ["## Warnings", ""] + [f"- {w}" for w in comp["warnings"]] + [""]
    parts += ["## Metrics", "", "| Metric | Simulated | Real | Gap |", "| --- | ---: | ---: | ---: |"]
    for m in comp.get("metrics") or []:
        parts.append(f"| {m['label']} | {_fmt(m['simulated'])} | {_fmt(m['real'])} | {_fmt(m['gap'])} |")
    parts.append("")
    sig = comp.get("significance") or {}
    boot = sig.get("bootstrap") or {}
    parts += ["## Significance", "",
              f"- Bootstrap gap CI (95%, {boot.get('rounds', 0)} resamples): gap {_fmt(boot.get('gap'))} "
              f"[{_fmt(boot.get('ci_low'))}, {_fmt(boot.get('ci_high'))}]",
              f"- Sign-test p: {_fmt(sig.get('sign_p'))} (None means too few paired clips)", ""]
    if comp.get("per_class_f1"):
        parts += ["## Per-class F1", "", "| Class | Sim | Real | Gap |", "| --- | ---: | ---: | ---: |"]
        for row in comp["per_class_f1"]:
            parts.append(f"| {row['label']} | {_fmt(row.get('simulated_f1'))} | {_fmt(row.get('real_f1'))} | {_fmt(row.get('gap'))} |")
        parts.append("")
    for side_name, run in (("Simulated", sim), ("Real", real)):
        by = (run.get("summary") or {}).get("by_condition") or {}
        if by:
            parts += [f"## By condition ({side_name})", "", "| Group | n | MAE | RMSE |", "| --- | ---: | ---: | ---: |"]
            for g, row in sorted(by.items()):
                parts.append(f"| {g} | {row.get('n')} | {_fmt(row.get('mae'))} | {_fmt(row.get('rmse'))} |")
            parts.append("")
    return "\n".join(parts)


def run_html(run: dict) -> str:
    body = html.escape(run_markdown(run)).replace("\n", "<br>")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(run['id'])}</title></head><body><h1>Run {html.escape(run['id'])}</h1><p>{body}</p></body></html>"


def compare_html(comp: dict, sim: dict, real: dict) -> str:
    body = html.escape(compare_markdown(comp, sim, real)).replace("\n", "<br>")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>comparison</title></head><body><h1>Comparison</h1><p>{body}</p></body></html>"


def leaderboard_rows(runs: list[dict]) -> list[dict]:
    """Best completed run per (task, dataset): higher accuracy/F1 wins, lower error wins."""
    best: dict[tuple, dict] = {}
    for run in runs:
        if run.get("status") != "completed":
            continue
        s = run.get("summary") or {}
        task = s.get("task") or "room"
        key = (task, run.get("dataset_id"))
        score = _sort_key(task, s)
        if score is None:
            continue
        if key not in best or score > best[key]["_score"]:
            best[key] = {"task": task, "dataset_id": run.get("dataset_id"), "dataset_name": run.get("dataset_name"),
                         "run_id": run["id"], "model_id": run.get("model_id"), "score": s, "_score": score,
                         "created_at": run.get("created_at")}
    rows = list(best.values())
    for r in rows:
        r.pop("_score", None)
    return sorted(rows, key=lambda r: (r["task"], r["dataset_id"] or ""))


def _sort_key(task: str, s: dict) -> float | None:
    if task in {"room", "classes"}:
        return float(s.get("macro_f1", 0.0))
    if task in {"t60", "distance"}:
        mae = s.get("mae")
        return -float(mae) if mae is not None else None
    if task == "events":
        return float(s.get("event_f1", 0.0))
    return None


def regression_alerts(runs: list[dict], *, tol_f1: float = 0.03, tol_mae: float = 0.05) -> list[dict]:
    """Flag the newest completed run per (task, dataset) when it regresses vs the prior best."""
    by_key: dict[tuple, list[dict]] = {}
    for run in runs:
        if run.get("status") != "completed":
            continue
        s = run.get("summary") or {}
        by_key.setdefault((s.get("task") or "room", run.get("dataset_id")), []).append(run)
    alerts = []
    for key, group in by_key.items():
        group = sorted(group, key=lambda r: r.get("created_at") or "")
        if len(group) < 2:
            continue
        prev, cur = group[-2], group[-1]
        ps, cs = prev.get("summary") or {}, cur.get("summary") or {}
        task = cs.get("task") or "room"
        if task in {"room", "classes"}:
            drop = float(ps.get("macro_f1", 0.0)) - float(cs.get("macro_f1", 0.0))
            if drop > tol_f1:
                alerts.append({"task": task, "dataset_id": key[1], "from_run": prev["id"], "to_run": cur["id"], "drop_f1": drop})
        elif task in {"t60", "distance"}:
            rise = float(cs.get("mae", 0.0)) - float(ps.get("mae", 0.0))
            if rise > tol_mae:
                alerts.append({"task": task, "dataset_id": key[1], "from_run": prev["id"], "to_run": cur["id"], "rise_mae": rise})
        elif task == "events":
            drop = float(ps.get("event_f1", 0.0)) - float(cs.get("event_f1", 0.0))
            if drop > tol_f1:
                alerts.append({"task": task, "dataset_id": key[1], "from_run": prev["id"], "to_run": cur["id"], "drop_f1": drop})
    return alerts


def draft_markdown(*, t60_comp, t60_sim, t60_real, room_comp, room_sim, room_real, range_run, events_run) -> str:
    lines = [f"# {PROJECT_LABEL} — working draft (team report)", "",
             f"Team {', '.join(TEAM)} · Advisor {ADVISOR}", "",
             "This draft is generated from completed harness runs. Demo synthetic scores are not findings.", "",
             f"## T60: simulated MAE {_fmt((t60_sim.get('summary') or {}).get('mae'))}, "
             f"real MAE {_fmt((t60_real.get('summary') or {}).get('mae'))}",
             f"## Room rule: simulated F1 {_fmt((room_sim.get('summary') or {}).get('macro_f1'))}, "
             f"real F1 {_fmt((room_real.get('summary') or {}).get('macro_f1'))}",
             f"## Range (synth only): MAE {_fmt((range_run.get('summary') or {}).get('mae'))} m",
             f"## Events (demo): F1 {_fmt((events_run.get('summary') or {}).get('event_f1'))}", "",
             "Full tables live in the sibling comparison reports.", ""]
    _ = (t60_comp, room_comp)
    return "\n".join(lines)


def _demo(run: dict) -> bool:
    flags = (run.get("config") or {}).get("dataset_flags") or {}
    return bool(flags.get("is_example") or flags.get("synthetic"))
