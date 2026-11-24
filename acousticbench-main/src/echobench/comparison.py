"""Sim-vs-real comparison: aggregate gaps (real − simulated) + significance."""

from __future__ import annotations

from echobench.scoring import gap_significance

CLASS_ROWS = (
    ("accuracy", "Accuracy", "higher"),
    ("macro_f1", "Macro F1", "higher"),
    ("macro_precision", "Macro precision", "higher"),
    ("macro_recall", "Macro recall", "higher"),
    ("weighted_f1", "Weighted F1", "higher"),
    ("ece", "Calibration error", "lower"),
    ("mean_latency_ms", "Mean latency (ms)", "lower"),
)
NUMBER_ROWS = (
    ("mae", "Mean absolute error", "lower"),
    ("rmse", "Root mean squared error", "lower"),
    ("median_ae", "Median absolute error", "lower"),
    ("mean_latency_ms", "Mean latency (ms)", "lower"),
)
EVENT_ROWS = (
    ("event_f1", "Event F1", "higher"),
    ("precision", "Event precision", "higher"),
    ("recall", "Event recall", "higher"),
    ("mean_latency_ms", "Mean latency (ms)", "lower"),
)
NUMBER_TASKS = {"t60", "distance"}

GAP_LINE = "gap = real − simulated"


def compare_pair(sim: dict, real: dict) -> dict:
    if sim.get("status") != "completed" or real.get("status") != "completed":
        raise ValueError("Both runs must be completed first.")
    s_sum, r_sum = sim.get("summary") or {}, real.get("summary") or {}
    s_task, r_task = s_sum.get("task") or "room", r_sum.get("task") or "room"
    if s_task != r_task:
        raise ValueError(f"Cannot compare '{s_task}' with '{r_task}'; use one task.")
    rows = NUMBER_ROWS if s_task in NUMBER_TASKS else (EVENT_ROWS if s_task == "events" else CLASS_ROWS)
    metrics = []
    for key, title, better in rows:
        a, b = s_sum.get(key), r_sum.get(key)
        gap = (float(b) - float(a)) if _num(a) and _num(b) else None
        metrics.append({"key": key, "label": title, "better": better, "simulated": a, "real": b, "gap": gap})
    per_class = []
    if "per_class" in s_sum or "per_class" in r_sum:
        s_pc, r_pc = s_sum.get("per_class") or {}, r_sum.get("per_class") or {}
        for tag in sorted(set(s_pc) & set(r_pc)):
            a, b = s_pc[tag].get("f1"), r_pc[tag].get("f1")
            per_class.append({"label": tag, "simulated_f1": a, "real_f1": b,
                              "gap": (float(b) - float(a)) if _num(a) and _num(b) else None,
                              "simulated_support": s_pc[tag].get("support"), "real_support": r_pc[tag].get("support")})
    sig = gap_significance(s_sum.get("item_scores") or [], r_sum.get("item_scores") or [],
                           s_sum.get("accuracy") or s_sum.get("event_f1") or 0.0,
                           r_sum.get("accuracy") or r_sum.get("event_f1") or 0.0)
    return {"sim_run_id": sim["id"], "real_run_id": real["id"], "model_id": sim.get("model_id"),
            "task": s_task, "metrics": metrics, "per_class_f1": per_class,
            "warnings": _warns(sim, real, s_sum, r_sum), "gap_definition": GAP_LINE, "significance": sig}


def _num(v) -> bool:  # noqa: ANN001
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _warns(sim: dict, real: dict, s_sum: dict, r_sum: dict) -> list[str]:
    out = []
    for run, side in ((sim, "simulated"), (real, "real")):
        flags = (run.get("config") or {}).get("dataset_flags") or {}
        if flags.get("is_example") or flags.get("synthetic"):
            out.append(f"The {side} run uses demo synthetic audio; not study results.")
    if sim.get("dataset_domain") != "simulated":
        out.append(f"Run chosen as simulated is registered '{sim.get('dataset_domain')}'.")
    if real.get("dataset_domain") != "real":
        out.append(f"Run chosen as real is registered '{real.get('dataset_domain')}'.")
    if sim.get("model_id") != real.get("model_id"):
        out.append("Different models: the gap mixes model change with domain change.")
    if (sim.get("config") or {}).get("sample_rate") != (real.get("config") or {}).get("sample_rate"):
        out.append("Different target sample rates.")
    for side, summ in (("simulated", s_sum), ("real", "real")):
        _ = side
    for side, summ in (("simulated", s_sum), ("real", r_sum)):
        if (summ.get("n_evaluated") or 0) < 10:
            out.append(f"The {side} run scored fewer than 10 clips; treat its gap as unstable.")
    return out
