"""Command line for EchoBench."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from echobench import __version__
from echobench.comparison import compare_pair
from echobench.demo import run_demo
from echobench.engine import run_one
from echobench.paths import db_file, ensure_importable, find_root
from echobench.reporting import compare_markdown, leaderboard_rows, regression_alerts, run_markdown
from echobench.service import inventory, queue_eval, register_dataset, register_model
from echobench.storage import Store


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="echobench", description="Reproducible sim-to-real audio benchmark.")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument("--root", type=Path, help="Project root (default: auto-detect).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="Start the local web UI.")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=_serve)

    g = sub.add_parser("examples", help="Build demo tone/noise/chirp + event clips.")
    g.set_defaults(fn=_examples)

    r = sub.add_parser("synth-rooms", help="Build synthetic shoebox RIRs with Sabine labels.")
    r.set_defaults(fn=_rooms)

    a = sub.add_parser("import-ace", help="Register local ACE single-channel RIRs.")
    a.set_defaults(fn=_ace)

    d = sub.add_parser("demo", help="Score all built-in baselines and rewrite reports/.")
    d.set_defaults(fn=_demo)

    ds = sub.add_parser("datasets", help="List or register datasets.")
    ds_sub = ds.add_subparsers(dest="ds_cmd", required=True)
    ds_sub.add_parser("list", help="List datasets.").set_defaults(fn=_ds_list)
    add = ds_sub.add_parser("add", help="Register a dataset.")
    add.add_argument("--file", type=Path)
    add.add_argument("--name")
    add.add_argument("--domain", choices=["simulated", "real"])
    add.add_argument("--path")
    add.add_argument("--layout", choices=["class_folders", "csv"])
    add.add_argument("--description", default="")
    add.set_defaults(fn=_ds_add)

    m = sub.add_parser("models", help="List built-in + registered models.")
    m.set_defaults(fn=_models)

    e = sub.add_parser("run", help="Score one model on one dataset.")
    e.add_argument("--model", required=True)
    e.add_argument("--dataset", required=True)
    e.add_argument("--config", type=Path)
    e.add_argument("--sample-rate", type=int)
    e.add_argument("--split")
    e.add_argument("--max-items", type=int)
    e.add_argument("--max-seconds", type=float)
    e.add_argument("--seed", type=int)
    e.add_argument("--notes", default="")
    e.add_argument("--docker-image", default="", help="Record docker isolation (optional; see ARCHITECTURE.md).")
    e.add_argument("--model-param", action="append", default=[], metavar="KEY=VALUE")
    e.set_defaults(fn=_run)

    sub.add_parser("runs", help="List recent runs.").set_defaults(fn=_runs)
    sub.add_parser("leaderboard", help="Show best run per task/dataset.").set_defaults(fn=_board)
    sub.add_parser("regressions", help="Flag runs that slipped vs the prior best.").set_defaults(fn=_regs)

    c = sub.add_parser("compare", help="Compare a simulated run with a real run.")
    c.add_argument("--sim", required=True)
    c.add_argument("--real", required=True)
    c.add_argument("--out", type=Path)
    c.set_defaults(fn=_compare)

    sh = sub.add_parser("show", help="Print one run as markdown.")
    sh.add_argument("run_id")
    sh.set_defaults(fn=_show)

    args = ap.parse_args(argv)
    try:
        args.fn(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        print(f"Error: {msg}", file=sys.stderr)
        raise SystemExit(1) from exc


def _root(args) -> Path:
    return args.root.resolve() if args.root else find_root()


def _store(args) -> Store:
    root = _root(args)
    ensure_importable(root)
    return Store(db_file(root))


def _serve(args) -> None:
    import uvicorn

    from echobench.web.server import build_app

    root = _root(args)
    ensure_importable(root)
    print(f"EchoBench at http://{args.host}:{args.port}")
    uvicorn.run(build_app(root), host=args.host, port=args.port)


def _examples(args) -> None:
    from echobench.synth import build_demos

    root = _root(args)
    for rec in build_demos(root, _store(args)):
        print(f"- {rec['id']}: {rec['domain']} -> {rec['root_path']}")
    print("Demo audio is synthetic; scores on it are not findings.")


def _rooms(args) -> None:
    from echobench.synth import build_rooms

    rec = build_rooms(_root(args), _store(args))
    print(f"- {rec['id']}: {rec['domain']} -> {rec['root_path']}")


def _ace(args) -> None:
    from echobench.ace_import import import_ace

    rec = import_ace(_root(args), _store(args))
    print(f"- {rec['id']}: {rec['domain']} -> {rec['root_path']}")


def _demo(args) -> None:
    root = _root(args)
    for name, path in run_demo(root, _store(args)).items():
        print(f"- {name}: {path}")


def _ds_list(args) -> None:
    rows = [inventory(r) for r in _store(args).all_datasets()]
    if not rows:
        print("No datasets. Start with: echobench examples")
        return
    for row in rows:
        inv = row.get("inventory") or {}
        flags = []
        if row["is_example"]:
            flags.append("demo")
        if row["synthetic_stand_in"]:
            flags.append("stand-in")
        tag = f" [{', '.join(flags)}]" if flags else ""
        print(f"{row['id']}  {row['domain']}  items={inv.get('n_items', '?')}  {row['name']}{tag}")


def _ds_add(args) -> None:
    fields: dict = {}
    if args.file:
        fields.update(yaml.safe_load(args.file.read_text(encoding="utf-8")) or {})
    for key, val in (("name", args.name), ("domain", args.domain), ("layout", args.layout), ("description", args.description)):
        if val:
            fields[key] = val
    if args.path:
        fields["root_path"] = args.path
    print(f"Registered {register_dataset(_store(args), fields)['id']}")


def _models(args) -> None:
    from echobench.plugins import list_entries

    root = _root(args)
    for ent in list_entries(root, _store(args).extra_models()):
        print(f"{ent.key}  [{ent.origin}/{ent.title}] task via {ent.dotted}")


def _run(args) -> None:
    root = _root(args)
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config else {}
    if not isinstance(raw, dict):
        raise ValueError("Config file must hold a mapping.")
    for key, val in (("sample_rate", args.sample_rate), ("split", args.split), ("max_items", args.max_items),
                     ("max_seconds", args.max_seconds), ("seed", args.seed), ("notes", args.notes),
                     ("docker_image", args.docker_image)):
        if val not in (None, ""):
            raw[key] = val
    for pair in args.model_param:
        if "=" not in pair:
            raise ValueError(f"Model param '{pair}' must look like KEY=VALUE.")
        k, v = pair.split("=", 1)
        raw.setdefault("model_params", {})[k.strip()] = _coerce(v.strip())
    store = _store(args)
    queued = queue_eval(store, root, model_id=args.model, dataset_id=args.dataset, raw=raw)
    done = run_one(store, queued["id"], root)
    print(run_markdown(done))
    if done.get("status") != "completed":
        raise SystemExit(1)


def _runs(args) -> None:
    _ = args
    for row in _store(args).list_runs():
        s = row.get("summary") or {}
        headline = s.get("macro_f1", s.get("mae", s.get("event_f1", "—")))
        print(f"{row['id']}  {row['status']:<10} {row['dataset_domain']:<10} score={headline}  {row['model_id']}  {row['dataset_id']}")


def _board(args) -> None:
    _ = args
    rows = leaderboard_rows(_store(args).list_runs(cap=2000))
    if not rows:
        print("No completed runs yet.")
        return
    for r in rows:
        s = r["score"]
        best = s.get("macro_f1", s.get("mae", s.get("event_f1")))
        print(f"{r['task']:<9} {r['dataset_id']:<18} run={r['run_id']} model={r['model_id']} best={best}")


def _regs(args) -> None:
    _ = args
    alerts = regression_alerts(_store(args).list_runs(cap=2000))
    if not alerts:
        print("No regressions vs the prior best.")
        return
    for a in alerts:
        print(a)


def _compare(args) -> None:
    store = _store(args)
    comp = compare_pair(store.fetch_run(args.sim), store.fetch_run(args.real))
    text = compare_markdown(comp, store.fetch_run(args.sim), store.fetch_run(args.real))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text)


def _show(args) -> None:
    print(run_markdown(_store(args).fetch_run(args.run_id)))


def _coerce(text: str):
    if text[:1] in "[{":
        import json

        try:
            return json.loads(text)
        except ValueError:
            return text
    try:
        return float(text) if "." in text else int(text)
    except ValueError:
        return text
