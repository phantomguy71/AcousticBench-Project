import json

import numpy as np

from echobench.audio_io import store_wav
from echobench.paths import db_file
from echobench.engine import run_one
from echobench.service import queue_eval, register_dataset
from echobench.storage import Store
from echobench.synth import LENGTH_S, RATE, make_tone


def _toy_tree(root, kinds=("tone", "noise")):
    rng = np.random.default_rng(0)
    for kind in kinds:
        for i in range(2):
            store_wav(root / kind / f"{kind}_{i}.wav", make_tone(kind, rng, RATE, 1.0), RATE)


def test_end_to_end_classes(tmp_path):
    root = tmp_path
    (root / "config").mkdir()
    store = Store(db_file(root))
    _toy_tree(root / "audio")
    ds = register_dataset(store, {"name": "tiny", "domain": "simulated", "layout": "class_folders",
                                  "root_path": root / "audio", "description": "t"})
    q = queue_eval(store, root, model_id="spectral-toy", dataset_id=ds["id"],
                   raw={"sample_rate": 16000, "max_seconds": 5, "seed": 0, "notes": "t"})
    done = run_one(store, q["id"], root)
    assert done["status"] == "completed"
    assert done["summary"]["task"] == "classes"
    assert done["summary"]["n_evaluated"] == 4


def test_end_to_end_events(tmp_path):
    import echobench.synth as synth

    root = tmp_path
    (root / "config").mkdir()
    store = Store(db_file(root))
    sed = root / "sed"
    # build two clips by hand with sidecars
    rng = np.random.default_rng(1)
    for i in range(2):
        n = RATE * 2
        sig = rng.normal(0, 0.02, size=n)
        s0 = int(0.5 * RATE)
        sig[s0:s0 + 2000] += 0.7 * np.sin(2 * np.pi * 180 * np.arange(2000) / RATE)
        p = sed / ("double" if i == 0 else "single") / f"c{i}.wav"
        store_wav(p, sig, RATE)
        p.with_suffix(".events.json").write_text(json.dumps([{"label": "knock", "onset_s": 0.5, "offset_s": 0.6}]))
    ds = register_dataset(store, {"name": "ev", "domain": "simulated", "layout": "class_folders",
                                  "root_path": sed, "description": "t"})
    _ = synth
    q = queue_eval(store, root, model_id="energy-events", dataset_id=ds["id"],
                   raw={"sample_rate": 16000, "max_seconds": 5, "seed": 0, "notes": "t"})
    done = run_one(store, q["id"], root)
    assert done["status"] == "completed"
    assert done["summary"]["task"] == "events"


def test_t60_and_distance(tmp_path):
    from echobench.rooms import synth_rir

    root = tmp_path
    (root / "config").mkdir()
    store = Store(db_file(root))
    folder = root / "rirset"
    folder.mkdir(parents=True)
    import csv

    rows = []
    for i, t60 in enumerate([0.4, 0.8]):
        rir = synth_rir(2.0, RATE, t60, 1.5, seed=i)
        store_wav(folder / "rir" / f"r{i}.wav", rir, RATE)
        rows.append((f"rir/r{i}.wav", t60, 1.5, "office" if t60 < 0.5 else "meeting"))
    with (folder / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "target_t60", "target_distance", "label", "condition", "split"])
        for rel, t60, dist, room in rows:
            w.writerow([rel, t60, dist, room, room, "dev"])
    ds = register_dataset(store, {"name": "rir", "domain": "simulated", "layout": "csv",
                                  "root_path": folder, "description": "t"})
    q = queue_eval(store, root, model_id="schroeder-t60", dataset_id=ds["id"],
                   raw={"sample_rate": 16000, "max_seconds": 5, "seed": 0, "notes": "t"})
    done = run_one(store, q["id"], root)
    assert done["status"] == "completed"
    assert done["summary"]["task"] == "t60"
