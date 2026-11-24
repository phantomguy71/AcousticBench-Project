"""Register the local ACE single-channel RIRs without copying audio."""

from __future__ import annotations

import csv
import tarfile
from pathlib import Path

from echobench.audio_io import SUPPORTED_SUFFIXES

SET_ID = "ace-single-rir"
MEAN_TABLE = "20150814T154139_Corpus_Mean_DRRs_and_T60s.csv"
ARCHIVE = "ACE_Corpus_RIRN_Single.tbz2"
SKIP = ("ambient", "babble", "fan", "noise")


def import_ace(root: Path, store) -> dict:  # noqa: ANN001
    downs = root / "data" / "ace" / "downloads"
    pack = downs / ARCHIVE
    if not pack.is_file() or pack.stat().st_size < 400_000_000:
        raise ValueError(f"ACE archive missing. Fetch {ARCHIVE} from https://doi.org/10.5281/zenodo.6257551 into {downs}.")
    out = root / "data" / "ace" / "single"
    if not any(out.rglob("*.wav")):
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(pack, "r:bz2") as tar:
            tar.extractall(out, filter="data")
    table = _locate_table(root)
    labels = _channel1_labels(table)
    wavs = [p for p in out.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and "rir" in p.stem.lower()]
    hits = []
    for wav in sorted(wavs):
        row = _match(wav.name, labels)
        if row is not None:
            hits.append((wav, row))
    if not hits:
        raise ValueError("No ACE impulse responses matched the published T60 table.")
    prep = root / "data" / "ace" / "prepared"
    prep.mkdir(parents=True, exist_ok=True)
    with (prep / "manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["path", "target_t60", "label", "condition", "split"])
        wr.writeheader()
        for wav, row in hits:
            wr.writerow({"path": wav.resolve().as_posix(), "target_t60": f"{row['t60']:.6f}", "label": family(row["room"]), "condition": row["room"], "split": "corpus"})
    (prep / "ACE_LICENSE.txt").write_text(
        "ACE Challenge corpus. Zenodo lists CC BY 4.0; Imperial page states CC BY-ND 4.0 — follow the stricter rule.\n"
        "Manifest points at extracted files; audio is never copied or committed.\nCite Eaton et al., IEEE/ACM TASLP 2016. Fullband T60, cruciform channel 1.\n",
        encoding="utf-8",
    )
    from echobench.service import register_dataset

    return register_dataset(
        store,
        {
            "id": SET_ID,
            "name": "ACE single-channel measured RIRs",
            "domain": "real",
            "layout": "csv",
            "root_path": prep,
            "description": "Measured ACE impulse responses; published fullband T60 for cruciform channel 1; room family from the room name.",
            "metadata": {"origin": "ACE Challenge", "doi": "10.5281/zenodo.6257551", "mic": "single cruciform-ch1"},
        },
        allow_flags=True,
    )


def family(room_name: str) -> str:
    low = room_name.lower()
    for key in ("office", "meeting", "lecture", "lobby"):
        if key in low:
            return key
    return room_name


def _locate_table(root: Path) -> Path:
    cands = list((root / "data" / "ace").rglob(MEAN_TABLE)) + list(root.rglob(MEAN_TABLE))
    if not cands:
        raise ValueError(f"ACE mean table {MEAN_TABLE} not found under data/ace.")
    return cands[0]


def _cell(row: dict, key: str) -> str:
    for k, v in row.items():
        if (k or "").strip().lower().rstrip(":") == key.strip().lower().rstrip(":"):
            return (v or "").strip()
    return ""


def _channel1_labels(table: Path) -> list[dict]:
    rows = []
    seen = set()
    with table.open(encoding="utf-8-sig", newline="") as fh:
        for rec in csv.DictReader(fh):
            if _cell(rec, "Mic config") != "Crucif" or _cell(rec, "Chan") != "1":
                continue
            key = (_cell(rec, "Room"), _cell(rec, "Room config"))
            if key in seen:
                continue
            seen.add(key)
            rows.append({"code": key[0], "room": _cell(rec, "Room decode"), "t60": float(_cell(rec, "FB T60"))})
    if not rows:
        raise ValueError(f"{table} has no cruciform channel-1 fullband rows.")
    return rows


def _match(filename: str, labels: list[dict]) -> dict | None:
    low = filename.lower()
    if any(w in low for w in SKIP):
        return None
    for row in labels:
        code = "".join(ch for ch in row["code"].lower() if ch.isalnum())
        stem = "".join(ch for ch in low if ch.isalnum())
        if code and code in stem:
            return row
    return None
