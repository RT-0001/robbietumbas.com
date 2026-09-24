"""Real font metrics so label boxes match what the renderer draws."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import ImageFont

FALLBACKS = {
    400: "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    700: "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
}


@dataclass(frozen=True)
class FontFace:
    path: str
    family: str
    weight: int
    upm: int
    cap: int  # cap height, font units


@dataclass(frozen=True)
class TextBox:
    text: str
    face: FontFace
    size: float  # px
    advance: float  # px, incl. tracking
    cap: float  # px
    tracking_px: float

    @property
    def w(self) -> float:
        return self.advance

    @property
    def h(self) -> float:
        return self.cap


def _face(path: Path) -> FontFace | None:
    try:
        f = TTFont(path, lazy=True)
    except Exception:
        return None
    name = f["name"]
    fam = name.getDebugName(16) or name.getDebugName(1)
    os2 = f["OS/2"]
    cap = getattr(os2, "sCapHeight", 0) or int(0.7 * f["head"].unitsPerEm)
    return FontFace(str(path), fam, int(os2.usWeightClass), int(f["head"].unitsPerEm), int(cap))


@lru_cache(maxsize=None)
def registry(fonts_dir: str) -> tuple[FontFace, ...]:
    d = Path(fonts_dir)
    faces = [_face(p) for p in sorted(d.glob("*")) if p.suffix.lower() in (".ttf", ".otf")]
    return tuple(f for f in faces if f is not None)


@lru_cache(maxsize=None)
def resolve_face(fonts_dir: str, family: str, weight: int) -> FontFace:
    cands = [f for f in registry(fonts_dir) if f.family.lower() == family.lower()]
    if cands:
        return min(cands, key=lambda f: abs(f.weight - weight))
    fb = FALLBACKS[700 if weight >= 600 else 400]
    warnings.warn(f"font {family} {weight} not in {fonts_dir}; using {fb}")
    return _face(Path(fb))


@lru_cache(maxsize=4096)
def _pil(path: str, size_q: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size_q / 4)


@lru_cache(maxsize=4096)
def measure(text: str, fonts_dir: str, family: str, weight: int, size: float, tracking_em: float = 0.0) -> TextBox:
    face = resolve_face(fonts_dir, family, weight)
    font = _pil(face.path, round(size * 4))
    tr = tracking_em * size
    adv = font.getlength(text) + tr * max(len(text) - 1, 0)
    return TextBox(text, face, size, adv, face.cap / face.upm * size, tr)


def measure_tok(text: str, tok, canvas_w: int, fonts_dir: str) -> TextBox:
    return measure(text, fonts_dir, tok.family, tok.weight, tok.size_ratio * canvas_w, tok.tracking)
