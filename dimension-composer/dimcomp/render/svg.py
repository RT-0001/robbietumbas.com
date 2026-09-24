"""Scene graph -> SVG. Pure translation: no layout decisions here."""
from __future__ import annotations

import base64
import io
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image


def _attr(v) -> str:
    return escape(str(v), {'"': "&quot;"})


def _text(layer: dict) -> str:
    """render_family is the face actually measured (a stand-in if Gibson is missing)."""
    f = layer["font"]
    h = f.get("h_scale", 1.0)
    pos = (f'transform="translate({layer["x"]} {layer["y"]}) scale({h} 1)" x="0" y="0"' if h != 1.0
           else f'x="{layer["x"]}" y="{layer["y"]}"')
    out = [f'<text id="{_attr(layer["id"])}" {pos} '
           f'font-family="{_attr(f.get("render_family", f["family"]))}" font-weight="{f["weight"]}" '
           f'font-size="{f["size"]}" fill="{layer["fill"]}" xml:space="preserve"'
           + (f' letter-spacing="{f["tracking_px"]}"' if f.get("tracking_px") else "") + ">"]
    for r in layer["runs"]:
        if r["scale"] == 1.0 and r["rise"] == 0.0:
            out.append(f"<tspan>{escape(r['text'])}</tspan>")
        else:
            size = round(f["size"] * r["scale"], 2)
            shift = round(r["rise"] * f["cap"], 2)
            out.append(f'<tspan font-size="{size}" baseline-shift="{shift}">{escape(r["text"])}</tspan>')
    out.append("</text>")
    return "".join(out)


def _ticks(a, b, kind, length, stroke, color) -> str:
    a, b = np.asarray(a), np.asarray(b)
    d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    n = np.array([-d[1], d[0]])
    out = []
    for p, sgn in ((a, 1), (b, -1)):
        if kind == "ticks":
            q0, q1 = p - n * length / 2, p + n * length / 2
            out.append(f'<line x1="{q0[0]:.2f}" y1="{q0[1]:.2f}" x2="{q1[0]:.2f}" y2="{q1[1]:.2f}"/>')
        elif kind == "arrows":
            tip = p
            base = p + sgn * d * length
            l, r = base + n * length * 0.35, base - n * length * 0.35
            out.append(f'<path d="M {tip[0]:.2f} {tip[1]:.2f} L {l[0]:.2f} {l[1]:.2f} L {r[0]:.2f} {r[1]:.2f} Z" '
                       f'fill="{color}" stroke="none"/>')
    return "".join(out)


def _dimension(layer: dict) -> str:
    c, w = layer["color"], layer["stroke"]
    parts = [f'<g id="{layer["id"]}" stroke="{c}" stroke-width="{w}" stroke-linecap="butt" fill="none">']
    for a, b in layer["segments"]:
        parts.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}"/>')
    for a, b in layer.get("extension", []):
        parts.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke-width="{w * 0.6:.2f}"/>')
    if layer["ticks"] != "none":
        a, b = layer["line"]
        parts.append(_ticks(a, b, layer["ticks"], layer["tick_len"], w, c))
    parts.append("</g>")
    lab = dict(layer["label"])
    parts.append(_text(lab))
    return "".join(parts)


def _image(layer: dict, root: Path, embed: bool, images: dict) -> str:
    """Embeds a PNG preview resampled to its placed size; the scene keeps the original
    (e.g. a 16-bit TIFF) as src for the PSD export."""
    t = layer["transform"]
    w, h = layer["size"]
    if embed:
        img = images.get(layer["src"])
        if img is None:
            from ..geometry.photo import load_photo
            img, _ = load_photo(root / layer["src"], require_alpha=False)
        pw, ph = max(1, round(w * t["scale"])), max(1, round(h * t["scale"]))
        if pw < img.width:
            img = img.resize((pw, ph), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        href = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        # keep placement in original-pixel units regardless of preview size
        return (f'<image id="{layer["id"]}" href="{href}" width="{w}" height="{h}" '
                f'transform="matrix({t["scale"]} 0 0 {t["scale"]} {t["tx"]} {t["ty"]})" '
                f'preserveAspectRatio="none"/>')
    href = _attr(layer["src"])
    return (f'<image id="{layer["id"]}" href="{href}" width="{w}" height="{h}" '
            f'transform="matrix({t["scale"]} 0 0 {t["scale"]} {t["tx"]} {t["ty"]})" '
            f'preserveAspectRatio="none"/>')


def _node(layer: dict, root: Path, embed: bool, images: dict) -> str:
    kind = layer["type"]
    if kind == "text":
        return _text(layer)
    if kind == "image":
        return _image(layer, root, embed, images)
    if kind == "dimension":
        return _dimension(layer)
    if kind == "rect":
        return (f'<rect id="{layer["id"]}" x="{layer["x"]:.2f}" y="{layer["y"]:.2f}" width="{layer["w"]:.2f}" '
                f'height="{layer["h"]:.2f}" rx="{layer.get("rx", 0):.2f}" fill="{layer["fill"]}"/>')
    if kind == "path":
        return f'<path id="{layer["id"]}" d="{layer["d"]}" fill="{layer["fill"]}"/>'
    if kind == "group":
        inner = "".join(_node(c, root, embed, images) for c in layer["children"])
        return f'<g id="{layer["id"]}">{inner}</g>'
    raise ValueError(f"unknown layer type {kind}")


def to_svg(scene: dict, root: Path, embed: bool = True, images: dict | None = None) -> str:
    c = scene["canvas"]
    body = "".join(_node(l, root, embed, images or {}) for l in scene["layers"])
    return (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{c["w"]}" height="{c["h"]}" viewBox="0 0 {c["w"]} {c["h"]}">'
            f'<rect id="background" width="{c["w"]}" height="{c["h"]}" fill="{c["bg"]}"/>{body}</svg>')
