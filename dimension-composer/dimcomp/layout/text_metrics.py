"""Real font metrics so label boxes match what the renderer draws.

The house face is Gibson (SemiBold + Regular). It is commercial, so it is not
in the repo: drop the licensed files in fonts/ or list their folder in
style.font_search_dirs. Until then the style's font_fallbacks stand-in is
used, sized by cap height so the layout barely moves, and every substitution
is flagged in the scene and report. The PSD export always requests the real
PostScript name.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import ImageFont

WEIGHT_NAMES = {100: "Thin", 200: "ExtraLight", 300: "Light", 400: "Regular", 500: "Medium",
                600: "SemiBold", 700: "Bold", 800: "ExtraBold", 900: "Black"}
LAST_RESORT = {400: "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
               700: "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"}


@dataclass(frozen=True)
class FontFace:
    path: str
    family: str
    weight: int
    upm: int
    cap: int  # cap height, font units
    postscript: str


@dataclass(frozen=True)
class TextBox:
    text: str
    face: FontFace  # what is actually measured / rendered
    requested: str  # family asked for by the style
    postscript: str  # what the PSD should request
    substituted: bool
    size: float  # px (em)
    advance: float  # px, incl. tracking and horizontal scale
    cap: float  # px
    tracking_px: float
    h_scale: float

    @property
    def w(self) -> float:
        return self.advance

    @property
    def h(self) -> float:
        return self.cap


def _face(path: Path) -> FontFace | None:
    try:
        f = TTFont(path, lazy=True, fontNumber=0)
    except Exception:
        return None
    name = f["name"]
    fam = name.getDebugName(16) or name.getDebugName(1)
    os2 = f["OS/2"]
    upm = int(f["head"].unitsPerEm)
    cap = getattr(os2, "sCapHeight", 0) or int(0.7 * upm)
    ps = name.getDebugName(6) or f"{fam}-{WEIGHT_NAMES.get(os2.usWeightClass, os2.usWeightClass)}"
    return FontFace(str(path), fam, int(os2.usWeightClass), upm, int(cap), ps)


@lru_cache(maxsize=None)
def registry(dirs: tuple[str, ...]) -> tuple[FontFace, ...]:
    faces = []
    for d in dirs:
        for p in sorted(Path(d).rglob("*")):
            if p.suffix.lower() in (".ttf", ".otf"):
                f = _face(p)
                if f is not None:
                    faces.append(f)
    return tuple(faces)


def _pick(dirs, family: str, weight: int) -> FontFace | None:
    cands = [f for f in registry(dirs) if f.family.lower() == family.lower()]
    return min(cands, key=lambda f: abs(f.weight - weight)) if cands else None


@lru_cache(maxsize=None)
def resolve_face(dirs: tuple[str, ...], family: str, weight: int,
                 fallbacks: tuple[tuple[str, str], ...] = ()) -> tuple[FontFace, bool]:
    face = _pick(dirs, family, weight)
    if face is not None:
        return face, False
    stand_in = dict(fallbacks).get(family)
    face = _pick(dirs, stand_in, weight) if stand_in else None
    if face is None:
        face = _face(Path(LAST_RESORT[700 if weight >= 600 else 400]))
    warnings.warn(f"{family} {weight} not found in {list(dirs)}; measuring with stand-in {face.family}. "
                  f"Add the licensed {family} files to fonts/ for exact metrics.", stacklevel=2)
    return face, True


@lru_cache(maxsize=4096)
def _pil(path: str, size_q: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size_q / 4)


@lru_cache(maxsize=4096)
def measure(text: str, dirs: tuple[str, ...], family: str, weight: int, cap_px: float | None,
            size_px: float | None, tracking_em: float = 0.0, h_scale: float = 1.0,
            fallbacks: tuple[tuple[str, str], ...] = (), postscript: str | None = None) -> TextBox:
    face, sub = resolve_face(dirs, family, weight, fallbacks)
    size = size_px if size_px is not None else cap_px * face.upm / face.cap
    font = _pil(face.path, round(size * 4))
    tr = tracking_em * size
    adv = (font.getlength(text) + tr * max(len(text) - 1, 0)) * h_scale
    ps = postscript or (face.postscript if not sub else f"{family}-{WEIGHT_NAMES.get(weight, weight)}")
    return TextBox(text, face, family, ps, sub, size, adv, face.cap / face.upm * size, tr, h_scale)


def measure_tok(text: str, tok, canvas_w: int, dirs: tuple[str, ...], fallbacks: dict | None = None,
                scale: float = 1.0) -> TextBox:
    """scale: multiplies the size (used for superscript runs)."""
    cap = tok.cap_ratio * canvas_w * scale if tok.cap_ratio else None
    size = tok.size_ratio * canvas_w * scale if tok.size_ratio else None
    return measure(text, dirs, tok.family, tok.weight, cap, size, tok.tracking, tok.h_scale,
                   tuple(sorted((fallbacks or {}).items())), tok.postscript)
