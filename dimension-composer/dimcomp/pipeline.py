"""Glue: spec -> context (photo, silhouette, fit) -> layouts -> files."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .config import Resolved
from .geometry import silhouette
from .geometry.box import Box
from .geometry.camera import Intrinsics
from .geometry.fit import Evidence, FitResult, fit
from .layout.scene import Context, Layout
from .render.debug import draw_fit
from .render.png import svg_to_png
from .render.svg import to_svg


def fit_path(cfg: Resolved) -> Path:
    return cfg.root / "fits" / f"{cfg.spec.sku}.json"


def evidence_path(cfg: Resolved) -> Path:
    return cfg.root / "fits" / f"{cfg.spec.sku}_evidence.json"


def load_context(cfg: Resolved, refit: bool = False, debug_dir: Path | None = None) -> Context:
    photo = silhouette.load_rgba(cfg.photo_path)
    s = cfg.style.fit.working_long_edge / max(photo.size)
    size = (round(photo.width * s), round(photo.height * s))
    work = photo.resize(size, Image.LANCZOS)
    sil = silhouette.extract(work, cfg.style.fit)

    cached = fit_path(cfg)
    res = None
    if cached.exists() and not refit:
        res = FitResult.from_json(json.loads(cached.read_text()))
        if tuple(res.image_size) != size:
            res = None
    if res is None:
        K0 = Intrinsics.from_profile(cfg.profile.camera, *size)
        box = Box.from_spec(cfg.spec.dims_in, cfg.profile.axis_map)
        ev = Evidence.load(evidence_path(cfg)).scaled(s) if evidence_path(cfg).exists() else None
        res = fit(sil, box, K0, cfg.profile, cfg.style.fit, size, ev)
        res.save(cached)
    if debug_dir is not None:
        draw_fit(photo, res, sil.hull, debug_dir / "debug_fit.png")
    src = str(cfg.photo_path.relative_to(cfg.root))
    clutter = {side: sil.side_clutter(side) for side in ("left", "right")}
    return Context(cfg, res, photo, src, sil, clutter)


def write_layout(ctx: Context, lay: Layout, out_dir: Path, name: str, png: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.scene.json").write_text(json.dumps(lay.scene, indent=1))
    svg = to_svg(lay.scene, ctx.cfg.root, embed=True)
    (out_dir / f"{name}.svg").write_text(svg)
    files = {"scene": f"{name}.scene.json", "svg": f"{name}.svg"}
    if png:
        svg_to_png(svg, ctx.cfg.fonts_dir, out_dir / f"{name}.png")
        files["png"] = f"{name}.png"
    return files
