"""WAV I/O on the standard library so Windows needs no extra audio backend.

Reads uncompressed PCM (8/16/24/32-bit) and mixes down to mono float64.
Writes 16-bit mono WAV. Resampling is linear interpolation: good enough to give
every model the same input rate, not a mastering-grade converter.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SUPPORTED_SUFFIXES = {".wav", ".wave"}


def read_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            width = handle.getsampwidth()
            channels = handle.getnchannels()
            frames = handle.getnframes()
            blob = handle.readframes(frames)
            kind = handle.getcomptype()
    except wave.Error as exc:
        raise ValueError(f"Cannot decode WAV {path}: {exc}") from exc
    if rate <= 0:
        raise ValueError(f"{path}: invalid sample rate")
    if channels < 1:
        raise ValueError(f"{path}: no channels")
    if kind not in {"NONE", "not compressed"}:
        raise ValueError(f"{path}: compressed WAV ({kind}) is not supported")
    mono = _decode(blob, width, path)
    if channels > 1:
        usable = (mono.size // channels) * channels
        if usable == 0:
            raise ValueError(f"{path}: incomplete interleaved frame")
        mono = mono[:usable].reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(mono, dtype=np.float64), int(rate)


def store_wav(path: Path, signal, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(signal, dtype=np.float64), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(rate))
        handle.writeframes(pcm.tobytes())


def to_rate(signal: np.ndarray, src: int, dst: int) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float64)
    if src == dst or arr.size == 0:
        return arr
    if src <= 0 or dst <= 0:
        raise ValueError("Sample rates must be positive")
    want = max(1, int(round(arr.size * dst / src)))
    if arr.size == 1 or want == 1:
        return np.full(want, float(arr[0]), dtype=np.float64)
    old = np.linspace(0.0, 1.0, num=arr.size, endpoint=False)
    new = np.linspace(0.0, 1.0, num=want, endpoint=False)
    return np.interp(new, old, arr).astype(np.float64)


def _decode(blob: bytes, width: int, path: Path) -> np.ndarray:
    if width == 1:
        raw = np.frombuffer(blob, dtype=np.uint8).astype(np.float64)
        return (raw - 128.0) / 128.0
    if width == 2:
        return np.frombuffer(blob, dtype="<i2").astype(np.float64) / 32768.0
    if width == 3:
        raw = np.frombuffer(blob, dtype=np.uint8)
        usable = (raw.size // 3) * 3
        tri = raw[:usable].reshape(-1, 3).astype(np.int32)
        vals = tri[:, 0] | (tri[:, 1] << 8) | (tri[:, 2] << 16)
        vals = np.where(vals >= 2**23, vals - 2**24, vals)
        return vals.astype(np.float64) / float(2**23)
    if width == 4:
        as_f32 = np.frombuffer(blob, dtype="<f4")
        ok = as_f32[np.isfinite(as_f32)]
        if ok.size and float(np.percentile(np.abs(ok), 95)) <= 2.5:
            return as_f32.astype(np.float64)
        return np.frombuffer(blob, dtype="<i4").astype(np.float64) / float(2**31)
    raise ValueError(f"{path}: {width * 8}-bit WAV is not supported (use 8/16/24/32-bit)")
