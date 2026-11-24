"""Sabine T60 plus a dependency-free synthetic room-impulse generator.

T60 = 0.161 * V / (alpha * S). The synthetic RIR is exponentially-decaying
noise whose decay constant matches the Sabine target, with a sharp direct
peak placed at distance / 343 s. This keeps `synth-rooms` runnable without
Pyroomacoustics; when Pyroomacoustics is installed the adapter prefers it and
records which engine produced each file.
"""

from __future__ import annotations

import numpy as np


def room_surface(dims: tuple[float, float, float]) -> float:
    x, y, z = dims
    return 2.0 * (x * y + x * z + y * z)


def room_volume(dims: tuple[float, float, float]) -> float:
    x, y, z = dims
    return x * y * z


def sabine_t60(dims: tuple[float, float, float], alpha: float) -> float:
    if alpha <= 0:
        raise ValueError("Absorption must be positive.")
    area = room_surface(dims)
    vol = room_volume(dims)
    if area <= 0 or vol <= 0:
        raise ValueError("Room dims must be positive.")
    return 0.161 * vol / (alpha * area)


def alpha_for_t60(dims: tuple[float, float, float], want: float) -> float:
    if want <= 0:
        raise ValueError("Target T60 must be positive.")
    area = room_surface(dims)
    vol = room_volume(dims)
    return 0.161 * vol / (want * area)


def synth_rir(duration_s: float, rate: int, t60: float, distance_m: float, seed: int) -> np.ndarray:
    """Decaying-noise RIR with the requested T60 and a direct peak for range."""
    rng = np.random.default_rng(seed)
    n = max(1, int(duration_s * rate))
    tau = max(t60 / 6.9078, 1e-3)  # -60dB in T60 seconds
    t = np.arange(n) / float(rate)
    body = rng.normal(0.0, 1.0, size=n) * np.exp(-t / tau)
    peak_at = min(n - 1, int(distance_m / 343.0 * rate))
    body[peak_at] += 3.0
    body /= max(float(np.max(np.abs(body))), 1e-9)
    return (body * 0.9).astype(np.float64)
