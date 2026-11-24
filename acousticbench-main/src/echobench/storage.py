"""SQLite store for datasets, extra models, and runs."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from uuid import uuid4

from echobench.util import utc_now

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  domain TEXT NOT NULL CHECK(domain IN ('simulated','real')),
  root_path TEXT NOT NULL,
  layout TEXT NOT NULL CHECK(layout IN ('class_folders','csv')),
  description TEXT NOT NULL DEFAULT '',
  is_example INTEGER NOT NULL DEFAULT 0,
  synthetic_stand_in INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS extra_models (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  qualified_name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  params_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  model_id TEXT NOT NULL,
  model_name TEXT NOT NULL,
  model_path TEXT NOT NULL DEFAULT '',
  model_params_json TEXT NOT NULL DEFAULT '{}',
  dataset_id TEXT NOT NULL,
  dataset_name TEXT NOT NULL,
  dataset_domain TEXT NOT NULL,
  config_json TEXT NOT NULL DEFAULT '{}',
  notes TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'queued',
  error TEXT NOT NULL DEFAULT '',
  summary_json TEXT NOT NULL DEFAULT '{}',
  predictions_path TEXT NOT NULL DEFAULT '',
  provenance_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  finished_at TEXT NOT NULL DEFAULT ''
);
"""


class Store:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._gate = threading.Lock()
        with sqlite3.connect(self.path) as db:
            db.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    # datasets -----------------------------------------------------------
    def save_dataset(self, rec: dict) -> dict:
        with self._gate, self._conn() as db:
            db.execute(
                "INSERT OR REPLACE INTO datasets VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    rec["id"], rec["name"], rec["domain"], rec["root_path"], rec["layout"],
                    rec.get("description", ""), int(bool(rec.get("is_example"))),
                    int(bool(rec.get("synthetic_stand_in"))), rec.get("created_at") or utc_now(),
                    json.dumps(rec.get("metadata") or {}),
                ),
            )
        return self.fetch_dataset(rec["id"])

    def fetch_dataset(self, key: str) -> dict:
        with self._conn() as db:
            row = db.execute("SELECT * FROM datasets WHERE id=?", (key,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown dataset '{key}'.")
        return _dataset_row(dict(row))

    def all_datasets(self) -> list[dict]:
        with self._conn() as db:
            rows = db.execute("SELECT * FROM datasets ORDER BY created_at").fetchall()
        return [_dataset_row(dict(r)) for r in rows]

    def drop_dataset(self, key: str) -> None:
        with self._gate, self._conn() as db:
            if db.execute("DELETE FROM datasets WHERE id=?", (key,)).rowcount == 0:
                raise KeyError(f"Unknown dataset '{key}'.")

    def dataset_count(self) -> int:
        with self._conn() as db:
            return int(db.execute("SELECT COUNT(*) FROM datasets").fetchone()[0])

    # extra models --------------------------------------------------------
    def save_model(self, rec: dict) -> dict:
        with self._gate, self._conn() as db:
            db.execute(
                "INSERT OR REPLACE INTO extra_models VALUES (?,?,?,?,?,?)",
                (rec["id"], rec["name"], rec["qualified_name"], rec.get("description", ""),
                 json.dumps(rec.get("params") or {}), utc_now()),
            )
        return self.extra_models_dict()[rec["id"]]

    def extra_models_dict(self) -> dict[str, dict]:
        with self._conn() as db:
            rows = db.execute("SELECT * FROM extra_models ORDER BY created_at").fetchall()
        out = {}
        for r in rows:
            d = dict(r)
            out[d["id"]] = {"id": d["id"], "name": d["name"], "qualified_name": d["qualified_name"],
                            "description": d["description"], "params": json.loads(d["params_json"] or "{}")}
        return out

    def extra_models(self) -> list[dict]:
        return list(self.extra_models_dict().values())

    def drop_model(self, key: str) -> None:
        with self._gate, self._conn() as db:
            if db.execute("DELETE FROM extra_models WHERE id=?", (key,)).rowcount == 0:
                raise KeyError(f"Unknown model '{key}'.")

    # runs -----------------------------------------------------------------
    def start_run(self, rec: dict) -> dict:
        rid = uuid4().hex[:10]
        payload = {
            "id": rid, "model_id": rec["model_id"], "model_name": rec.get("model_name", rec["model_id"]),
            "model_path": rec.get("model_path", ""), "model_params": rec.get("model_params") or {},
            "dataset_id": rec["dataset_id"], "dataset_name": rec.get("dataset_name", rec["dataset_id"]),
            "dataset_domain": rec.get("dataset_domain", ""), "config": rec.get("config") or {},
            "notes": rec.get("notes", ""), "status": "queued", "error": "",
            "summary": {}, "predictions_path": "", "provenance": rec.get("provenance") or {},
            "created_at": utc_now(), "finished_at": "",
        }
        with self._gate, self._conn() as db:
            db.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (payload["id"], payload["model_id"], payload["model_name"], payload["model_path"],
                 json.dumps(payload["model_params"]), payload["dataset_id"], payload["dataset_name"],
                 payload["dataset_domain"], json.dumps(payload["config"]), payload["notes"], "queued", "",
                 "{}", "", json.dumps(payload["provenance"]), payload["created_at"], ""),
            )
        return self.fetch_run(rid)

    def fetch_run(self, rid: str) -> dict:
        with self._conn() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown run '{rid}'.")
        return _run_row(dict(row))

    def list_runs(self, cap: int = 200) -> list[dict]:
        with self._conn() as db:
            rows = db.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (cap,)).fetchall()
        return [_run_row(dict(r)) for r in rows]

    def runs_for(self, model_id: str | None = None, dataset_id: str | None = None) -> list[dict]:
        query, args = "SELECT * FROM runs", []
        clauses = []
        if model_id:
            clauses.append("model_id=?")
            args.append(model_id)
        if dataset_id:
            clauses.append("dataset_id=?")
            args.append(dataset_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC"
        with self._conn() as db:
            rows = db.execute(query, args).fetchall()
        return [_run_row(dict(r)) for r in rows]

    def flag_running(self, rid: str) -> None:
        with self._gate, self._conn() as db:
            db.execute("UPDATE runs SET status='running' WHERE id=?", (rid,))

    def close_run(self, rid: str, *, status: str, summary=None, predictions_path="", error="") -> dict:
        from echobench.util import safe_json

        with self._gate, self._conn() as db:
            db.execute(
                "UPDATE runs SET status=?, summary_json=?, predictions_path=?, error=?, finished_at=? WHERE id=?",
                (status, json.dumps(safe_json(summary or {})), predictions_path, error, utc_now(), rid),
            )
        return self.fetch_run(rid)


def fresh_id(name: str, taken: set[str]) -> str:
    base = "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-") or "dataset"
    while "--" in base:
        base = base.replace("--", "-")
    cand, i = base[:48], 2
    while cand in taken:
        cand = f"{base[:40]}-{i}"
        i += 1
    return cand


def _dataset_row(row: dict) -> dict:
    return {
        "id": row["id"], "name": row["name"], "domain": row["domain"], "root_path": row["root_path"],
        "layout": row["layout"], "description": row.get("description", ""),
        "is_example": bool(row.get("is_example")), "synthetic_stand_in": bool(row.get("synthetic_stand_in")),
        "created_at": row.get("created_at", ""), "metadata": json.loads(row.get("metadata_json") or "{}"),
    }


def _run_row(row: dict) -> dict:
    def _j(key: str, fallback):
        try:
            return json.loads(row.get(key) or "")
        except Exception:
            return fallback
    return {
        "id": row["id"], "model_id": row["model_id"], "model_name": row["model_name"],
        "model_qualified_name": row.get("model_path", ""), "model_params": _j("model_params_json", {}),
        "dataset_id": row["dataset_id"], "dataset_name": row["dataset_name"], "dataset_domain": row["dataset_domain"],
        "config": _j("config_json", {}), "notes": row.get("notes", ""), "status": row["status"],
        "error": row.get("error", ""), "summary": _j("summary_json", {}),
        "predictions_path": row.get("predictions_path", ""), "provenance": _j("provenance_json", {}),
        "created_at": row.get("created_at", ""), "finished_at": row.get("finished_at", ""),
    }
