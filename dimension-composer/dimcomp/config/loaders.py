"""Load spec/profile/style and resolve config with precedence:

base style -> profile.style_overrides -> spec.overrides -> session tweaks.

Each layer is deep-merged as a dict and re-validated, so a bad patch fails at
the layer that introduced it.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import ValidationError

from .models import Profile, Spec, Style


def deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_set(items: list[str]) -> dict:
    """['search.samples=50', 'annotations.ticks=arrows'] -> nested dict (values YAML-parsed)."""
    patch: dict = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects key.path=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = patch
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return patch


def _read(path: Path) -> dict:
    text = path.read_text()
    return json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)


@dataclass
class Resolved:
    root: Path
    spec: Spec
    profile: Profile
    style: Style
    layers: list[str] = field(default_factory=list)

    @property
    def photo_path(self) -> Path:
        return self.root / (self.spec.photo or f"photos/{self.spec.sku}.png")

    @property
    def fonts_dir(self) -> Path:
        return self.root / self.style.fonts_dir

    def dump(self) -> dict:
        return {
            "layers": self.layers,
            "spec": self.spec.model_dump(mode="json"),
            "profile": self.profile.model_dump(mode="json"),
            "style": self.style.model_dump(mode="json"),
        }


def project_root(spec_path: Path) -> Path:
    """Directory holding specs/, profiles/, styles/ (parent of the spec's folder)."""
    p = spec_path.resolve().parent
    return p.parent if p.name == "specs" else p


def resolve(spec_path: str | Path, style_name: str = "igloo_default",
            session: dict | None = None, root: Path | None = None) -> Resolved:
    spec_path = Path(spec_path)
    root = root or project_root(spec_path)
    spec = Spec.model_validate(_read(spec_path))
    profile = Profile.model_validate(_read(root / "profiles" / f"{spec.angle_profile}.yaml"))

    layers = [f"styles/{style_name}.yaml"]
    data = _read(root / "styles" / f"{style_name}.yaml")
    _validate(data, layers[-1])
    for label, patch in [(f"profile:{profile.name}", profile.style_overrides),
                         (f"sku:{spec.sku}", spec.overrides),
                         ("session", session or {})]:
        if patch:
            data = deep_merge(data, patch)
            _validate(data, label)
            layers.append(label)
    return Resolved(root=root, spec=spec, profile=profile, style=Style.model_validate(data), layers=layers)


def _validate(data: dict, layer: str) -> None:
    try:
        Style.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"style invalid after applying {layer}:\n{e}") from None
