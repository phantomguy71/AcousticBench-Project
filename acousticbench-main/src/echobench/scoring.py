"""Task metrics plus lightweight significance helpers (numpy only)."""

from __future__ import annotations

import numpy as np

from echobench.util import paired_bootstrap_gap, paired_sign_test


def classify_metrics(truth: list[str], pred: list[str], confs=None, lat=None) -> dict:
    if len(truth) != len(pred):
        raise ValueError("Truth and predictions must align.")
    if not truth:
        raise ValueError("Need at least one scored clip.")
    cats = sorted(set(truth) | set(pred))
    pos = {c: i for i, c in enumerate(cats)}
    grid = np.zeros((len(cats), len(cats)), dtype=int)
    for t, p in zip(truth, pred):
        grid[pos[t], pos[p]] += 1
    per, f1s, ps, rs, wsum, sup = {}, [], [], [], 0.0, 0
    for c in cats:
        i = pos[c]
        tp, sup_c, out_c = int(grid[i, i]), int(grid[i, :].sum()), int(grid[:, i].sum())
        prec, rec = (tp / out_c if out_c else 0.0), (tp / sup_c if sup_c else 0.0)
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        per[c] = {"precision": prec, "recall": rec, "f1": f1, "support": sup_c}
        ps.append(prec)
        rs.append(rec)
        f1s.append(f1)
        wsum += f1 * sup_c
        sup += sup_c
    hits = int(np.trace(grid))
    total = int(grid.sum())
    clean_conf: list[float | None] = []
    for value in confs or [None] * len(pred):
        try:
            num = float(value) if value is not None else None
        except (TypeError, ValueError):
            num = None
        clean_conf.append(num if (num is not None and 0.0 <= num <= 1.0) else None)
    flags = [t == p for t, p in zip(truth, pred)]
    ece, n_ece = calibration_from_flags(clean_conf, flags)
    lat = _latency(lat or [])
    return {
        "task": "classification",
        "n_evaluated": total,
        "n_correct": hits,
        "accuracy": hits / total if total else 0.0,
        "macro_precision": float(np.mean(ps)),
        "macro_recall": float(np.mean(rs)),
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": (wsum / sup) if sup else 0.0,
        "labels": cats,
        "per_class": per,
        "confusion_matrix": {"labels": cats, "counts": grid.tolist()},
        "mean_latency_ms": lat["mean"],
        "p50_latency_ms": lat["p50"],
        "p95_latency_ms": lat["p95"],
        "ece": ece,
        "ece_count": n_ece,
    }


def regress_metrics(truth: list[float], pred: list[float], lat=None, groups=None, task="t60", unit="") -> dict:
    if len(truth) != len(pred):
        raise ValueError("Targets and predictions must align.")
    if not truth:
        raise ValueError("Need at least one scored clip.")
    t = np.asarray(truth, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    err = p - t
    lat = _latency(lat or [])
    by: dict[str, dict] = {}
    if groups:
        bucket: dict[str, list[int]] = {}
        for i, g in enumerate(groups):
            if g:
                bucket.setdefault(g, []).append(i)
        for g in sorted(bucket):
            idx = np.asarray(bucket[g])
            e = err[idx]
            by[g] = {"n": int(idx.size), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e**2)))}
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((t - t.mean()) ** 2)) if t.size > 1 else 0.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {
        "task": task,
        "unit": unit,
        "n_evaluated": int(t.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "median_ae": float(np.median(np.abs(err))),
        "r2": r2,
        "mean_latency_ms": lat["mean"],
        "p50_latency_ms": lat["p50"],
        "p95_latency_ms": lat["p95"],
        "by_condition": by,
    }


def detect_metrics(truth_spans: list[list[tuple[float, float]]], pred_spans: list[list[tuple[float, float]]], tol: float = 0.22, lat=None) -> dict:
    """Onset-tolerant event scoring: a predicted span counts when its onset lands
    within `tol` seconds of an unmatched true onset with the same tag."""
    if len(truth_spans) != len(pred_spans):
        raise ValueError("Event lists must align per clip.")
    tp = fp = fn = 0
    per_clip_f1: list[float] = []
    for t_spans, p_spans in zip(truth_spans, pred_spans):
        used = [False] * len(t_spans)
        hit = 0
        for p_on, _p_off in p_spans:
            best, bi = None, -1
            for i, (t_on, _t_off) in enumerate(t_spans):
                if used[i]:
                    continue
                gap = abs(p_on - t_on)
                if gap <= tol and (best is None or gap < best):
                    best, bi = gap, i
            if bi >= 0:
                used[bi] = True
                hit += 1
        tp += hit
        fp += len(p_spans) - hit
        fn += len(t_spans) - hit
        denom = 2 * hit + (len(p_spans) - hit) + (len(t_spans) - hit)
        per_clip_f1.append((2 * hit / denom) if denom else 1.0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    lat = _latency(lat or [])
    return {
        "task": "events",
        "unit": "spans",
        "n_evaluated": len(truth_spans),
        "n_true_events": tp + fn,
        "n_pred_events": tp + fp,
        "precision": prec,
        "recall": rec,
        "event_f1": f1,
        "mean_clip_f1": float(np.mean(per_clip_f1)) if per_clip_f1 else 0.0,
        "mean_latency_ms": lat["mean"],
        "p50_latency_ms": lat["p50"],
        "p95_latency_ms": lat["p95"],
        "tolerance_s": tol,
    }


def gap_significance(sim_item_scores: list[float], real_item_scores: list[float], sim_acc: float, real_acc: float, sim_pred=None, real_pred=None, sim_true=None, real_true=None) -> dict:
    boot = paired_bootstrap_gap(sim_item_scores, real_item_scores)
    sign_p = None
    if sim_pred is not None and real_pred is not None and sim_true is not None and real_true is not None:
        # compare per-clip correctness counts (only valid when clips pair 1:1 by index)
        n = min(len(sim_true), len(real_true))
        a = sum(1 for i in range(n) if real_pred[i] == real_true[i] and sim_pred[i] != sim_true[i])
        b = sum(1 for i in range(n) if sim_pred[i] == sim_true[i] and real_pred[i] != real_true[i])
        sign_p = paired_sign_test(a, b)
    return {"bootstrap": boot, "sign_p": sign_p, "sim_acc": sim_acc, "real_acc": real_acc}


def calibration_from_flags(confs: list[float | None], ok: list[bool], bins: int = 10) -> tuple[float | None, int]:
    pairs = [(c, f) for c, f in zip(confs, ok) if c is not None]
    if not pairs:
        return None, 0
    cs = np.array([p[0] for p in pairs])
    hs = np.array([1.0 if p[1] else 0.0 for p in pairs])
    edges = np.linspace(0, 1, bins + 1)
    ece, n = 0.0, len(pairs)
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        sel = (cs >= lo) & (cs <= hi) if b == bins - 1 else (cs >= lo) & (cs < hi)
        k = int(sel.sum())
        if k:
            ece += (k / n) * abs(float(hs[sel].mean()) - float(cs[sel].mean()))
    return float(ece), n


def _latency(ms: list[float]) -> dict:
    if not ms:
        return {"mean": None, "p50": None, "p95": None}
    arr = np.asarray(ms, dtype=np.float64)
    return {"mean": float(arr.mean()), "p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95))}
