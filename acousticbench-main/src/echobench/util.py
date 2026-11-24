"""Tiny shared helpers: time, JSON safety, seeding, statistics."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_json(value):  # noqa: ANN001, ANN202
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        num = float(value)
        return None if (math.isnan(num) or math.isinf(num)) else num
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    np.random.seed(int(seed) % (2**32 - 1))


def mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, pct))


def paired_bootstrap_gap(
    sim_scores: list[float],
    real_scores: list[float],
    *,
    seed: int = 0,
    rounds: int = 2000,
) -> dict:
    """Bootstrap 95% CI for mean(real) - mean(sim) using per-run item scores.

    Falls back to an empty CI when a side has no scores.
    """
    rng = np.random.default_rng(seed)
    sim = np.asarray(sim_scores, dtype=np.float64)
    real = np.asarray(real_scores, dtype=np.float64)
    if sim.size == 0 or real.size == 0:
        return {"gap": None, "ci_low": None, "ci_high": None, "rounds": 0}
    gap = float(real.mean() - sim.mean())
    draws: list[float] = []
    for _ in range(rounds):
        a = rng.choice(sim, size=sim.size, replace=True).mean()
        b = rng.choice(real, size=real.size, replace=True).mean()
        draws.append(float(b - a))
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return {"gap": gap, "ci_low": lo, "ci_high": hi, "rounds": rounds}


def paired_sign_test(a_wins: int, b_wins: int, ties: int = 0) -> float | None:
    """Two-sided sign-test p-value via the normal approximation (no scipy)."""
    n = a_wins + b_wins
    if n < 6:
        return None
    diff = abs(a_wins - b_wins)
    # continuity-corrected z for fair-coin null
    z = (diff - 1.0) / math.sqrt(n) if n else 0.0
    # Phi via erf
    phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    p = 2.0 * (1.0 - phi)
    _ = ties
    return max(0.0, min(1.0, float(p)))
