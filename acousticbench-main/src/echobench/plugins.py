"""Model registry: built-ins plus config-file and UI-registered entries."""

from __future__ import annotations

import importlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from echobench.models_api import BaseModel

log = logging.getLogger(__name__)
DOTTED = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")


@dataclass
class ModelEntry:
    key: str
    title: str
    dotted: str
    blurb: str
    params: dict
    origin: str  # builtin | config | registered

    def as_dict(self) -> dict:
        return {
            "id": self.key,
            "name": self.title,
            "qualified_name": self.dotted,
            "description": self.blurb,
            "params": self.params,
            "source": self.origin,
            "deletable": self.origin == "registered",
        }


def _builtins() -> dict[str, type[BaseModel]]:
    from echobench.builtin.events import EnergyEventDetector
    from echobench.builtin.fixed import FixedTagModel
    from echobench.builtin.ranged import OnsetRangeModel
    from echobench.builtin.roomrule import DecayRoomRule
    from echobench.builtin.spectral import SpectralToyClassifier
    from echobench.builtin.t60 import SchroederT60

    return {
        SchroederT60.model_id: SchroederT60,
        DecayRoomRule.model_id: DecayRoomRule,
        OnsetRangeModel.model_id: OnsetRangeModel,
        SpectralToyClassifier.model_id: SpectralToyClassifier,
        FixedTagModel.model_id: FixedTagModel,
        EnergyEventDetector.model_id: EnergyEventDetector,
    }


def list_entries(root: Path, extra: list[dict] | None = None) -> list[ModelEntry]:
    table: dict[str, ModelEntry] = {}
    for key, cls in _builtins().items():
        table[key] = ModelEntry(key, cls.display_name, f"{cls.__module__}:{cls.__name__}", cls.blurb, {}, "builtin")
    for ent in _from_yaml(root / "config" / "models.yaml"):
        if ent.key in table:
            log.warning("config model %s collides with an existing id; skipped", ent.key)
            continue
        table[ent.key] = ent
    for row in extra or []:
        if row["id"] in table:
            log.warning("registered model %s collides; skipped", row["id"])
            continue
        table[row["id"]] = ModelEntry(row["id"], row["name"], row["qualified_name"], row.get("description") or "", row.get("params") or {}, "registered")
    return list(table.values())


def find_entry(root: Path, key: str, extra: list[dict] | None = None) -> ModelEntry:
    for ent in list_entries(root, extra):
        if ent.key == key:
            return ent
    raise ValueError(f"Unknown model '{key}'.")


def build(ent: ModelEntry, overrides: dict | None = None) -> BaseModel:
    args = dict(ent.params)
    args.update(overrides or {})
    natives = _builtins()
    cls = natives[ent.key] if (ent.key in natives and ent.origin == "builtin") else load_class(ent.dotted)
    try:
        obj = cls(**args)
    except TypeError as exc:
        raise ValueError(f"Cannot construct model '{ent.key}' with params {sorted(args)}: {exc}") from exc
    if not isinstance(obj, BaseModel):
        raise ValueError(f"{ent.dotted} must extend BaseModel.")
    return obj


def load_class(dotted: str) -> type[BaseModel]:
    if not DOTTED.match(dotted):
        raise ValueError("Model path must look like 'some.module:SomeClass'. Importing runs local code.")
    mod_name, cls_name = dotted.split(":")
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:
        raise ValueError(f"Cannot import {mod_name}: {exc}") from exc
    try:
        cls = getattr(mod, cls_name)
    except AttributeError as exc:
        raise ValueError(f"{mod_name} has no attribute {cls_name}.") from exc
    return cls


def _from_yaml(path: Path) -> list[ModelEntry]:
    if not path.is_file():
        return []
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"{path} must hold a mapping.")
    out: list[ModelEntry] = []
    for item in doc.get("models") or []:
        if not isinstance(item, dict):
            raise ValueError(f"Each model in {path} must be a mapping.")
        key = str(item.get("id") or "").strip()
        dotted = str(item.get("qualified_name") or "").strip()
        if not key or not dotted:
            raise ValueError(f"Each model in {path} needs id + qualified_name.")
        params = item.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError(f"Params for '{key}' must be a mapping.")
        out.append(ModelEntry(key, str(item.get("name") or key), dotted, str(item.get("description") or ""), params, "config"))
    return out
