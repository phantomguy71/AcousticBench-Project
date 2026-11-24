"""Toy spectral rules for the built-in tone/noise/chirp demo sets."""

from __future__ import annotations

import numpy as np

from echobench.models_api import BaseModel, ModelResult

KINDS = ("chirp", "noise", "tone")
FLAT_IF_NOISE = 0.22
SLOPE_IF_CHIRP = 320.0


class SpectralToyClassifier(BaseModel):
    model_id = "spectral-toy"
    display_name = "Spectral toy rules"
    task = "classes"
    blurb = "Demo-only rules on flatness + centroid drift for generated tones, noise, chirps."

    def infer(self, waveform, sample_rate: int) -> ModelResult:
        feats = spectral_feats(waveform, sample_rate)
        tag = decide(feats)
        return ModelResult(tag=tag, confidences=soft_votes(tag, feats), detail={"feats": feats})


def spectral_feats(waveform, sample_rate: int) -> dict:
    sig = np.ascontiguousarray(np.asarray(waveform, dtype=np.float64))
    if sig.size == 0:
        raise ValueError("Empty clip.")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")
    rms = float(np.sqrt(np.mean(sig**2)))
    frame, hop = 512, 256
    if sig.size < frame:
        buf = np.zeros(frame)
        buf[: sig.size] = sig
        blocks = buf.reshape(1, frame)
    else:
        n = 1 + (sig.size - frame) // hop
        shape = (n, frame)
        strides = (sig.strides[0] * hop, sig.strides[0])
        blocks = np.array(np.lib.stride_tricks.as_strided(sig, shape=shape, strides=strides), dtype=np.float64, copy=True)
    win = np.hanning(frame)
    mag = np.abs(np.fft.rfft(blocks * win, axis=1))
    freqs = np.fft.rfftfreq(frame, d=1.0 / float(sample_rate))
    flat = [float(np.exp(np.mean(np.log(r + 1e-12)))) / float(np.mean(r + 1e-12)) for r in mag]
    cent = [float(np.sum(freqs * r) / max(float(np.sum(r)), 1e-12)) for r in mag]
    times = np.arange(len(cent)) * hop / float(sample_rate)
    slope = float(np.polyfit(times, cent, 1)[0]) if len(cent) > 1 else 0.0
    return {"rms": rms, "flatness": float(np.mean(flat)), "centroid_hz": float(np.mean(cent)), "drift_hz_s": slope}


def decide(feats: dict) -> str:
    if feats["rms"] < 1e-6 or feats["flatness"] >= FLAT_IF_NOISE:
        return "noise"
    if abs(feats["drift_hz_s"]) >= SLOPE_IF_CHIRP:
        return "chirp"
    return "tone"


def soft_votes(tag: str, feats: dict) -> dict:
    raw = {
        "noise": (feats["flatness"] - FLAT_IF_NOISE) * 8.0,
        "chirp": abs(feats["drift_hz_s"]) / SLOPE_IF_CHIRP - 1.0,
        "tone": 1.0 - feats["flatness"] * 4.0,
    }
    raw[tag] = raw.get(tag, 0.0) + 2.0
    keys = list(raw)
    vals = np.array([raw[k] for k in keys], dtype=np.float64)
    vals -= vals.max()
    exp = np.exp(vals)
    probs = exp / exp.sum()
    return {k: float(p) for k, p in zip(keys, probs)}
