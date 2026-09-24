"""SVG -> PNG via resvg, loading only the project's fonts so metrics match."""
from __future__ import annotations

from pathlib import Path

import resvg_py


def svg_to_png(svg: str, fonts_dir: Path, out: Path, width: int | None = None) -> Path:
    data = resvg_py.svg_to_bytes(svg_string=svg, font_dirs=[str(fonts_dir)], skip_system_fonts=True,
                                 width=width, text_rendering="geometric_precision")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(data))
    return out
