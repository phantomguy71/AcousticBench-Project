"""Local FastAPI front end for datasets, runs, and sim-to-real comparison."""

from __future__ import annotations

import threading
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from echobench import ADVISOR, PROJECT_LABEL, SYNTH_CAVEAT, TEAM, __version__
from echobench.comparison import GAP_LINE, compare_pair
from echobench.engine import run_one
from echobench.paths import db_file, ensure_importable
from echobench.plugins import list_entries
from echobench.reporting import compare_html, compare_markdown, leaderboard_rows, regression_alerts, run_html, run_markdown
from echobench.service import inventory, queue_eval, register_dataset, register_model
from echobench.storage import Store

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"


def build_app(root: Path | None = None) -> FastAPI:
    base = (root or Path.cwd()).resolve()
    ensure_importable(base)
    store = Store(db_file(base))
    app = FastAPI(title=PROJECT_LABEL, version=__version__)
    app.state.root = base
    app.state.store = store
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.exception_handler(ValueError)
    async def _bad(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(KeyError)
    async def _missing(_: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": exc.args[0] if exc.args else "Not found."})

    @app.get("/")
    def _index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/overview")
    def _overview() -> dict:
        runs = store.list_runs(cap=2000)
        return {"project": PROJECT_LABEL, "version": __version__, "team": list(TEAM), "advisor": ADVISOR,
                "datasets": store.dataset_count(), "runs": len(runs),
                "models": len(list_entries(base, store.extra_models())),
                "caveat": SYNTH_CAVEAT, "gap": GAP_LINE,
                "tasks": "room · classes · t60 · distance · events",
                "regressions": regression_alerts(runs)}

    @app.get("/api/datasets")
    def _datasets() -> list[dict]:
        return [inventory(r) for r in store.all_datasets()]

    @app.post("/api/datasets")
    def _add_dataset(body: dict) -> dict:
        return inventory(register_dataset(store, body))

    @app.post("/api/examples/generate")
    def _gen_examples() -> dict:
        from echobench.synth import build_demos

        recs = build_demos(base, store)
        return {"datasets": [inventory(r) for r in recs], "caveat": SYNTH_CAVEAT}

    @app.delete("/api/datasets/{key}")
    def _drop_dataset(key: str) -> dict:
        store.drop_dataset(key)
        return {"deleted": key}

    @app.get("/api/models")
    def _models() -> list[dict]:
        return [e.as_dict() for e in list_entries(base, store.extra_models())]

    @app.post("/api/models")
    def _add_model(body: dict) -> dict:
        saved = register_model(store, base, body)
        return next(e.as_dict() for e in list_entries(base, store.extra_models()) if e.key == saved["id"])

    @app.delete("/api/models/{key}")
    def _drop_model(key: str) -> dict:
        known = {e.key: e for e in list_entries(base, store.extra_models())}
        ent = known.get(key)
        if ent is None:
            raise KeyError(f"Unknown model '{key}'.")
        if ent.origin != "registered":
            raise ValueError("Only UI-registered models can be removed here.")
        store.drop_model(key)
        return {"deleted": key}

    @app.get("/api/runs")
    def _runs() -> list[dict]:
        return store.list_runs()

    @app.get("/api/runs/{rid}")
    def _one_run(rid: str) -> dict:
        return store.fetch_run(rid)

    @app.get("/api/runs/{rid}/report.md")
    def _run_md(rid: str):
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(run_markdown(store.fetch_run(rid)))

    @app.get("/api/runs/{rid}/report.html")
    def _run_html(rid: str) -> str:
        return run_html(store.fetch_run(rid))

    @app.post("/api/evaluations")
    def _eval(body: dict) -> dict:
        run = queue_eval(store, base, model_id=str(body.get("model_id") or ""),
                         dataset_id=str(body.get("dataset_id") or ""), raw=_cfg(body))
        _launch(store, base, run["id"])
        return run

    @app.post("/api/evaluations/paired")
    def _paired(body: dict) -> dict:
        group = uuid4().hex[:8]
        cfg = _cfg(body)
        cfg["paired_group"] = group
        sim_ds, real_ds = str(body.get("sim_dataset_id") or ""), str(body.get("real_dataset_id") or "")
        if not sim_ds or not real_ds:
            raise ValueError("Pick a simulated and a real dataset.")
        if sim_ds == real_ds:
            raise ValueError("Simulated and real datasets must differ.")
        model = str(body.get("model_id") or "")
        sim = queue_eval(store, base, model_id=model, dataset_id=sim_ds, raw=dict(cfg))
        real = queue_eval(store, base, model_id=model, dataset_id=real_ds, raw=dict(cfg))
        thread = threading.Thread(target=_run_pair, args=(store, base, sim["id"], real["id"]), daemon=True)
        thread.start()
        return {"paired_group": group, "sim_run_id": sim["id"], "real_run_id": real["id"]}

    @app.get("/api/compare")
    def _compare(sim: str, real: str) -> dict:
        return compare_pair(store.fetch_run(sim), store.fetch_run(real))

    @app.get("/api/compare.md")
    def _compare_md(sim: str, real: str):
        from fastapi.responses import PlainTextResponse

        comp = compare_pair(store.fetch_run(sim), store.fetch_run(real))
        return PlainTextResponse(compare_markdown(comp, store.fetch_run(sim), store.fetch_run(real)))

    @app.get("/api/compare.html")
    def _compare_html(sim: str, real: str) -> str:
        comp = compare_pair(store.fetch_run(sim), store.fetch_run(real))
        return compare_html(comp, store.fetch_run(sim), store.fetch_run(real))

    @app.get("/api/leaderboard")
    def _board() -> list[dict]:
        return leaderboard_rows(store.list_runs(cap=2000))

    @app.get("/api/regressions")
    def _regs() -> list[dict]:
        return regression_alerts(store.list_runs(cap=2000))

    return app


def _cfg(body: dict) -> dict:
    params = body.get("model_params") or {}
    if isinstance(params, str):
        import json

        params = json.loads(params or "{}")
    out = {"sample_rate": body.get("sample_rate", 16000), "split": body.get("split", "all"),
           "max_items": body.get("max_items"), "max_seconds": body.get("max_seconds", 30),
           "seed": body.get("seed", 11), "notes": body.get("notes", ""), "model_params": params,
           "event_tolerance_s": body.get("event_tolerance_s", 0.25)}
    const_tag = (body.get("constant_tag") or body.get("constant_label") or "").strip()
    if const_tag:
        out["model_params"] = {**params, "tag": const_tag}
    return out


def _launch(store: Store, root: Path, rid: str) -> None:
    thread = threading.Thread(target=run_one, args=(store, rid, root), daemon=True)
    thread.start()


def _run_pair(store: Store, root: Path, sim_id: str, real_id: str) -> None:
    run_one(store, sim_id, root)
    run_one(store, real_id, root)
