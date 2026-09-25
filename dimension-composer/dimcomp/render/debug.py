"""Fit debug overlay: wireframe box over the photo, labeled vertices, QA numbers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..geometry.box import EDGES
from ..geometry.fit import FitResult

OK, BAD = (40, 170, 60, 255), (220, 50, 50, 255)


def _dashed(draw, a, b, fill, width, dash=12):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(int(np.linalg.norm(b - a) / dash), 1)
    for i in range(0, n, 2):
        p, q = a + (b - a) * i / n, a + (b - a) * min(i + 1, n) / n
        draw.line([tuple(p), tuple(q)], fill=fill, width=width)


def draw_fit(photo: Image.Image, fit: FitResult, hull=None, out: Path | None = None,
             max_px: int = 1400, core=None) -> Image.Image:
    """Blue: full outline hull. Magenta: body hull (handles opened away) used for the angles."""
    img = photo.convert("RGBA").resize(fit.image_size, Image.LANCZOS)
    bg = Image.new("RGBA", img.size, (235, 235, 235, 255))
    bg.alpha_composite(img)
    pad = int(0.18 * max(img.size))
    canvas = Image.new("RGBA", (img.width + 2 * pad, img.height + 2 * pad), (250, 250, 250, 255))
    canvas.alpha_composite(bg, (pad, pad))
    d = ImageDraw.Draw(canvas)
    lw = max(2, img.width // 500)
    font = ImageFont.load_default(size=max(14, img.width // 70))
    off = np.array([pad, pad])

    for poly, col in ((hull, (0, 150, 220, 255)), (core, (200, 0, 160, 255))):
        if poly is not None:
            pts = [tuple(np.array(p) + off) for p in poly.exterior.coords]
            d.line(pts, fill=col, width=lw)

    uv = {k: v + off for k, v in fit.box.project(fit.pose, fit.K).items()}
    visible = fit.box.visible_edges(fit.pose)
    color = OK if fit.status == "ok" else BAD
    for name, (a, b) in EDGES.items():
        if name in visible:
            d.line([tuple(uv[a]), tuple(uv[b])], fill=color, width=lw)
        else:
            _dashed(d, uv[a], uv[b], (240, 140, 0, 255), max(1, lw // 2))
    for name, p in uv.items():
        d.ellipse([p[0] - 2 * lw, p[1] - 2 * lw, p[0] + 2 * lw, p[1] + 2 * lw], fill=color)
        d.text((p[0] + 3 * lw, p[1] + 2 * lw), name, fill=(20, 20, 20, 255), font=font)

    qa = fit.qa()
    lines = [f"{k}: {v}" for k, v in qa.items() if k != "notes"] + [f"! {n}" for n in qa["notes"]]
    d.multiline_text((12, 10), "\n".join(lines), fill=color, font=font, spacing=4)

    s = max_px / max(canvas.size)
    if s < 1:
        canvas = canvas.resize((round(canvas.width * s), round(canvas.height * s)), Image.LANCZOS)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        canvas.convert("RGB").save(out)
    return canvas
