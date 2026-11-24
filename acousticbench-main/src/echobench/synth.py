"""Demo corpora builders: toy sounds, synth rooms, and a SED set."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from echobench.audio_io import store_wav
from echobench.rooms import alpha_for_t60, sabine_t60, synth_rir

RATE = 16000
LENGTH_S = 2.4
EACH = 10
KINDS = ("tone", "noise", "chirp")
SEED = 20261123

DEMO_SETS = (
    {
        "id": "demo-clean",
        "title": "Demo clean sounds",
        "domain": "simulated",
        "folder": "clean",
        "messy": False,
        "standin": False,
        "note": "DEMO. Clean generated tones, noise and chirps. Stands in for a simulated-domain set; not room audio, not findings.",
    },
    {
        "id": "demo-messy",
        "title": "Demo degraded stand-in for real audio",
        "domain": "real",
        "folder": "messy",
        "messy": True,
        "standin": True,
        "note": "DEMO / SYNTHETIC STAND-IN. Same generators with noise, gain wobble and low-pass. Registered as real only so compare/ can be tried; not recordings.",
    },
)


def build_demos(root: Path, store) -> list[dict]:  # noqa: ANN001
    from echobench.service import register_dataset

    made = []
    for spec in DEMO_SETS:
        folder = root / "data" / "examples" / spec["folder"]
        _fill_demo_folder(folder, messy=spec["messy"], domain=spec["domain"])
        made.append(
            register_dataset(
                store,
                {
                    "id": spec["id"],
                    "name": spec["title"],
                    "domain": spec["domain"],
                    "layout": "class_folders",
                    "root_path": folder,
                    "description": spec["note"],
                    "is_example": True,
                    "synthetic_stand_in": spec["standin"],
                    "metadata": {"synthetic": True, "example": True, "seed": SEED, "rate": RATE, "kinds": list(KINDS)},
                },
                allow_flags=True,
            )
        )
    # SED demo clips (class folders + sidecar event files)
    sed_folder = root / "data" / "examples" / "sed"
    _fill_sed_folder(sed_folder)
    made.append(
        register_dataset(
            store,
            {
                "id": "demo-events",
                "name": "Demo sound-event clips",
                "domain": "simulated",
                "layout": "class_folders",
                "root_path": sed_folder,
                "description": "DEMO. Two-knock / one-knock synthetic clips with onset labels in sidecar .events.json files.",
                "is_example": True,
                "synthetic_stand_in": False,
                "metadata": {"synthetic": True, "example": True, "seed": SEED, "rate": RATE},
            },
            allow_flags=True,
        )
    )
    return made


ROOM_PLAN = (
    {"room": "office", "name": "office-a", "dims": (3.4, 3.1, 2.5), "t60": 0.32},
    {"room": "office", "name": "office-b", "dims": (4.4, 3.2, 2.6), "t60": 0.40},
    {"room": "office", "name": "office-c", "dims": (5.2, 3.6, 2.7), "t60": 0.47},
    {"room": "meeting", "name": "meeting-a", "dims": (5.8, 4.0, 2.9), "t60": 0.58},
    {"room": "meeting", "name": "meeting-b", "dims": (6.2, 5.2, 3.0), "t60": 0.68},
    {"room": "meeting", "name": "meeting-c", "dims": (7.4, 5.6, 3.1), "t60": 0.78},
    {"room": "lecture", "name": "lecture-a", "dims": (8.4, 6.2, 3.3), "t60": 0.95},
    {"room": "lecture", "name": "lecture-b", "dims": (10.4, 7.2, 3.6), "t60": 1.10},
    {"room": "lecture", "name": "lecture-c", "dims": (11.6, 8.4, 4.1), "t60": 1.25},
)
MIC_SPOTS = ((0.58, 0.46, 0.42), (0.66, 0.60, 0.58), (0.44, 0.72, 0.62), (0.72, 0.38, 0.52))
SRC_SPOT = (0.28, 0.34, 0.46)
ROOM_SECONDS = 3.0


def build_rooms(root: Path, store) -> dict:  # noqa: ANN001
    """Write dependency-free synthetic RIRs (Sabine labels) and register them."""
    from echobench.service import register_dataset

    folder = root / "data" / "synth-rooms" / "shoebox-dev"
    if folder.exists():
        for p in folder.rglob("*"):
            if p.is_file():
                p.unlink()
    wav_dir = folder / "rir"
    rows = []
    try:
        import pyroomacoustics as pra  # type: ignore

        engine = f"pyroomacoustics {getattr(pra, '__version__', 'unknown')}"
        use_pra = True
    except Exception:
        pra = None  # type: ignore
        engine = "echobench-decay-noise"
        use_pra = False
    for idx, spec in enumerate(ROOM_PLAN):
        dims = spec["dims"]
        alpha = alpha_for_t60(dims, spec["t60"])
        ref = sabine_t60(dims, alpha)
        src = _at(dims, SRC_SPOT)
        for mic_no, frac in enumerate(MIC_SPOTS, start=1):
            mic = _at(dims, frac)
            dist = float(np.linalg.norm(np.subtract(mic, src)))
            if use_pra:
                rir = _pra_rir(pra, dims, alpha, src, mic)
            else:
                rir = synth_rir(ROOM_SECONDS, RATE, ref, dist, SEED + idx * 17 + mic_no)
            name = f"{spec['name']}_{mic_no:02d}.wav"
            store_wav(wav_dir / name, rir, RATE)
            rows.append((f"rir/{name}", ref, dist, spec["room"]))
    with (folder / "manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["path", "target_t60", "target_distance", "label", "condition", "split"])
        for rel, t60, dist, room in rows:
            wr.writerow([rel, f"{t60:.6f}", f"{dist:.6f}", room, room, "dev"])
    (folder / "SYNTHETIC.txt").write_text(
        "SYNTHETIC DEVELOPMENT RIRs\nEngine: " + engine + "\nt60 column is the Sabine value for the wall absorption.\n"
        "distance column is straight-line source-to-mic meters.\nNot measured rooms; not sponsor data.\n",
        encoding="utf-8",
    )
    (folder / "metadata.json").write_text(
        json.dumps({"engine": engine, "t60_ref": "sabine", "rate": RATE, "rooms": len(ROOM_PLAN)}, indent=2), encoding="utf-8"
    )
    return register_dataset(
        store,
        {
            "id": "synth-rooms-dev",
            "name": "Synthetic shoebox development RIRs",
            "domain": "simulated",
            "layout": "csv",
            "root_path": folder,
            "description": "Shoebox RIRs with Sabine T60 labels and Euclidean distances. Engine: " + engine + ".",
            "metadata": {"engine": engine, "t60_ref": "sabine"},
        },
    )


def _at(dims, frac) -> tuple[float, float, float]:
    return (dims[0] * frac[0], dims[1] * frac[1], dims[2] * frac[2])


def _pra_rir(pra, dims, alpha, src, mic):  # noqa: ANN001
    room = pra.ShoeBox(list(dims), fs=RATE, materials=pra.Material(alpha), max_order=40, air_absorption=False)
    room.add_source(list(src))
    room.add_microphone(list(mic))
    room.image_source_model()
    room.compute_rir()
    rir = np.asarray(room.rir[0][0], dtype=np.float64).ravel()
    want = int(ROOM_SECONDS * RATE)
    if rir.size < want:
        rir = np.pad(rir, (0, want - rir.size))
    return (rir[:want] / max(float(np.max(np.abs(rir))), 1e-9) * 0.9).astype(np.float64)


def make_tone(kind: str, rng: np.random.Generator, rate: int = RATE, seconds: float = LENGTH_S) -> np.ndarray:
    n = int(rate * seconds)
    t = np.arange(n, dtype=np.float64) / rate
    if kind == "tone":
        f = float(rng.choice([220.0, 440.0, 660.0, 880.0, 1320.0]))
        return (0.6 * np.sin(2.0 * np.pi * f * t)).astype(np.float64)
    if kind == "noise":
        return rng.normal(0.0, 0.25, size=n)
    if kind == "chirp":
        a, b = float(rng.uniform(150, 600)), float(rng.uniform(1800, 5000))
        if rng.random() < 0.5:
            a, b = b, a
        phase = 2.0 * np.pi * (a * t + (b - a) * t**2 / (2.0 * seconds))
        return (0.6 * np.sin(phase)).astype(np.float64)
    raise ValueError(f"Unknown demo kind '{kind}'.")


def roughen(sig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    dirty = sig * float(rng.uniform(0.35, 0.85)) + rng.normal(0.0, 0.16, size=sig.shape)
    out = np.empty_like(dirty)
    out[0] = dirty[0]
    for i in range(1, len(dirty)):
        out[i] = 0.92 * out[i - 1] + 0.08 * dirty[i]
    return np.clip(out, -1.0, 1.0)


def _fill_demo_folder(folder: Path, *, messy: bool, domain: str) -> None:
    rng = np.random.default_rng(SEED + (1 if messy else 0))
    for kind in KINDS:
        for i in range(EACH):
            sig = make_tone(kind, rng)
            if messy:
                sig = roughen(sig, rng)
            store_wav(folder / kind / f"{kind}_{i:02d}.wav", sig, RATE)
    (folder / "EXAMPLE.txt").write_text(f"DEMO SYNTHETIC AUDIO\nmessy={messy}\ndomain={domain}\nseed={SEED}\n", encoding="utf-8")


def _fill_sed_folder(folder: Path) -> None:
    rng = np.random.default_rng(SEED + 99)
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(12):
        n = int(RATE * 3.0)
        sig = rng.normal(0.0, 0.02, size=n)
        knocks = 2 if i % 2 == 0 else 1
        spans = []
        for k in range(knocks):
            at = 0.4 + k * 1.1 + float(rng.uniform(-0.05, 0.05))
            s0, s1 = int(at * RATE), int(min(n, (at + 0.20) * RATE))
            burst = np.sin(2 * np.pi * 180.0 * np.arange(s1 - s0) / RATE) * np.exp(-np.arange(s1 - s0) / (RATE * 0.03))
            sig[s0:s1] += burst * 0.8
            spans.append({"label": "knock", "onset_s": round(s0 / RATE, 4), "offset_s": round(s1 / RATE, 4)})
        path = folder / ("double" if knocks == 2 else "single") / f"clip_{i:02d}.wav"
        store_wav(path, sig, RATE)
        path.with_suffix(".events.json").write_text(json.dumps(spans), encoding="utf-8")
    (folder / "EXAMPLE.txt").write_text("DEMO event clips with sidecar .events.json onsets.\n", encoding="utf-8")
