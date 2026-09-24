"""Candidate params -> laid-out geometry (canvas px) + scene graph.

The scene graph is the single source of truth for every renderer; renderers
never make layout decisions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from shapely import affinity
from shapely.geometry import LineString, MultiLineString, box as rect

from ..config import Resolved
from ..geometry.fit import FitResult
from ..geometry.silhouette import Silhouette
from .annotations import Dim, build_dims
from .text_metrics import TextBox, measure_tok

SUPERSCRIPTS = "®™"  # ® ™
SUP_SCALE, SUP_RISE = 0.5, 0.42  # of font size / of cap height


@dataclass
class Params:
    offset_ratio: float = 0.06
    label_t: dict = field(default_factory=lambda: {"L": 0.5, "W": 0.5, "H": 0.5})
    height_side: str = "right"
    group_scale: float = 0.9
    nudge: tuple = (0.0, 0.0)
    extension_lines: bool = False

    def as_dict(self) -> dict:
        d = asdict(self)
        d["nudge"] = list(self.nudge)
        return d


@dataclass
class Context:
    """Everything fixed per SKU: config, fit, photo, silhouette, text metrics."""
    cfg: Resolved
    fit: FitResult
    photo: Image.Image  # original resolution
    photo_src: str
    sil: Silhouette  # working resolution
    clutter: dict  # side -> clutter score
    cache: dict = field(default_factory=dict)  # param-independent pieces (title, callouts)

    @property
    def work_scale(self) -> float:
        return self.fit.image_size[0] / self.photo.width

    @property
    def canvas(self):
        return self.cfg.style.canvas

    def text(self, s: str, tok_name: str) -> TextBox:
        st = self.cfg.style
        return measure_tok(s, getattr(st.tokens.fonts, tok_name), st.canvas.w, str(self.cfg.fonts_dir))


@dataclass
class PlacedDim:
    dim: Dim
    p0: np.ndarray  # canvas
    p1: np.ndarray
    label: TextBox
    anchor: np.ndarray
    label_box: tuple  # x0, y0, x1, y1
    segments: list
    extension: list

    @property
    def geom(self) -> MultiLineString:
        return MultiLineString([[tuple(a), tuple(b)] for a, b in self.segments + self.extension])


@dataclass
class Layout:
    params: Params
    s: float  # working px -> canvas px
    T: np.ndarray
    dims: list[PlacedDim]
    hull: object  # shapely, canvas
    contour: object
    group_bbox: tuple
    title_box: tuple
    callout_boxes: dict
    scene: dict

    def dim(self, axis: str) -> PlacedDim:
        return next(d for d in self.dims if d.dim.axis == axis)


# ------------------------------------------------------------------ helpers

def fmt(value: float, style) -> str:
    return style.tokens.number_format.format(value=value)


def split_line(p0, p1, box, gap):
    """Line minus the padded label box -> list of (a, b) segments."""
    line = LineString([tuple(p0), tuple(p1)])
    hole = rect(box[0] - gap, box[1] - gap, box[2] + gap, box[3] + gap)
    rest = line.difference(hole)
    parts = [rest] if rest.geom_type == "LineString" else list(getattr(rest, "geoms", []))
    return [(np.array(g.coords[0]), np.array(g.coords[-1])) for g in parts if g.length > 1.0]


def rich_runs(text: str, tb: TextBox) -> tuple[list[dict], float]:
    """Split ®/™ into superscript runs; return runs and total advance."""
    from .text_metrics import measure
    runs, buf = [], ""
    for ch in text:
        if ch in SUPERSCRIPTS:
            if buf:
                runs.append({"text": buf, "scale": 1.0, "rise": 0.0})
                buf = ""
            runs.append({"text": ch, "scale": SUP_SCALE, "rise": SUP_RISE})
        else:
            buf += ch
    if buf:
        runs.append({"text": buf, "scale": 1.0, "rise": 0.0})
    adv = 0.0
    fonts_dir = str(Path(tb.face.path).parent)
    for r in runs:
        m = measure(r["text"], fonts_dir, tb.face.family, tb.face.weight, tb.size * r["scale"], 0.0)
        r["advance"] = m.advance
        adv += m.advance
    return runs, adv


def text_layer(id_: str, tb: TextBox, x: float, cy: float, fill: str, align="center", text=None) -> tuple[dict, tuple]:
    """Text centered on cap height at (x, cy). Returns layer and box."""
    text = tb.text if text is None else text
    runs, adv = rich_runs(text, tb)
    baseline = cy + tb.cap / 2
    x0 = x - adv / 2 if align == "center" else x
    layer = {"id": id_, "type": "text", "text": text, "runs": runs,
             "font": {"family": tb.face.family, "weight": tb.face.weight, "size": round(tb.size, 2),
                      "file": Path(tb.face.path).name, "tracking_px": round(tb.tracking_px, 2),
                      "cap": round(tb.cap, 2)},
             "x": round(x0, 2), "y": round(baseline, 2), "align": "left", "fill": fill,
             "box": [round(v, 2) for v in (x0, cy - tb.cap / 2, x0 + adv, cy + tb.cap / 2)]}
    return layer, tuple(layer["box"])


def zone_px(zone, canvas) -> tuple:
    return (zone.x * canvas.w, zone.y * canvas.h, (zone.x + zone.w) * canvas.w, (zone.y + zone.h) * canvas.h)


def safe_px(style) -> tuple:
    c, m = style.canvas, style.layout.safe_area
    return (m.left * c.w, m.top * c.h, (1 - m.right) * c.w, (1 - m.bottom) * c.h)


# ------------------------------------------------------------------ callouts

CAN_PATH = ("M {x0} {y1} L {x0} {yb} Q {x0} {yt} {x0p} {yt} L {x1p} {yt} Q {x1} {yt} {x1} {yb} "
            "L {x1} {y1} Q {x1} {yf} {x1p} {yf} L {x0p} {yf} Q {x0} {yf} {x0} {y1} Z")


def callout_cans(ctx: Context, zone: tuple) -> tuple[dict, tuple]:
    st = ctx.cfg.style
    col = st.tokens.colors
    x0, y0, x1, y1 = zone
    cx = (x0 + x1) / 2
    head = ctx.text("HOLDS UP TO", "callout_text_bold")
    head_l, head_box = text_layer("cans_head", head, cx, y0 + head.cap / 2, col.callout_text)
    num = ctx.text(str(ctx.cfg.spec.can_count), "callout_num")
    word = ctx.text("CANS", "callout_text_bold")
    can_top = head_box[3] + 0.9 * head.cap
    can_w = max(num.advance, word.advance) * 1.35
    can_h = y1 - can_top
    cx0, cx1 = cx - can_w / 2, cx + can_w / 2
    r = can_w * 0.12
    tab_w, tab_h = can_w * 0.28, can_h * 0.07
    body_top = can_top + tab_h
    path = CAN_PATH.format(x0=cx0, x1=cx1, y1=y1 - r, yb=body_top + r, yt=body_top,
                           x0p=cx0 + r, x1p=cx1 - r, yf=y1)
    tab = {"id": "cans_tab", "type": "rect", "x": cx - tab_w / 2, "y": can_top, "w": tab_w,
           "h": tab_h + r, "rx": tab_h * 0.4, "fill": col.can_fill}
    body = {"id": "cans_body", "type": "path", "d": path, "fill": col.can_fill}
    gap = 0.35 * word.cap
    stack = num.cap + gap + word.cap
    top = body_top + (y1 - body_top - stack) / 2
    num_l, _ = text_layer("cans_num", num, cx, top + num.cap / 2, st.tokens.colors.text)
    word_l, _ = text_layer("cans_word", word, cx, top + num.cap + gap + word.cap / 2, st.tokens.colors.text)
    group = {"id": "callout_cans", "type": "group", "children": [head_l, tab, body, num_l, word_l]}
    return group, (min(head_box[0], cx0), y0, max(head_box[2], cx1), y1)


def callout_interior(ctx: Context, zone: tuple) -> tuple[dict, tuple]:
    st = ctx.cfg.style
    d = ctx.cfg.spec.interior_dims_in
    l1 = ctx.text("INTERIOR DIMENSIONS (L × W × H):", "callout_text")
    l2 = ctx.text(" × ".join(fmt(v, st) for v in (d.L, d.W, d.H)), "callout_text_bold")
    x0, y0, x1, y1 = zone
    cx = (x0 + x1) / 2
    gap = 0.8 * l1.cap
    top = (y0 + y1) / 2 - (l1.cap + gap + l2.cap) / 2
    a, ab = text_layer("interior_label", l1, cx, top + l1.cap / 2, st.tokens.colors.callout_text)
    b, bb = text_layer("interior_value", l2, cx, top + l1.cap + gap + l2.cap / 2, st.tokens.colors.text)
    return ({"id": "callout_interior", "type": "group", "children": [a, b]},
            (min(ab[0], bb[0]), ab[1], max(ab[2], bb[2]), bb[3]))


# ------------------------------------------------------------------ build

def _group_bbox(s, work_pts, dims: list[Dim], labels: dict[str, TextBox], t: dict):
    pts = [s * work_pts]
    for d in dims:
        a, b = s * d.p0, s * d.p1
        c = a + t[d.axis] * (b - a)
        tb = labels[d.axis]
        pts.append(np.array([c - [tb.w / 2, tb.h / 2], c + [tb.w / 2, tb.h / 2], a, b]))
        for e0, e1 in d.extension:
            pts.append(s * np.array([e0, e1]))
    P = np.vstack(pts)
    return P.min(0), P.max(0)


def _static(ctx: Context):
    """Title and callouts: identical for every candidate, so built once."""
    if "static" in ctx.cache:
        return ctx.cache["static"]
    cfg, st = ctx.cfg, ctx.cfg.style
    cw, ch = st.canvas.w, st.canvas.h
    # title
    tb = ctx.text(cfg.spec.title, "title")
    band = st.layout.title_band
    title_l, title_box = text_layer("title", tb, cw / 2, (band.top + band.height / 2) * ch, st.tokens.colors.text)

    # callouts
    callouts, callout_boxes = [], {}
    for cid, zone_name in st.layout.callouts.items():
        zone = st.layout.reserved_zones.get(zone_name)
        if zone is None:
            continue
        z = zone_px(zone, st.canvas)
        if cid == "cans" and cfg.spec.can_count:
            g, b = callout_cans(ctx, z)
        elif cid == "interior" and cfg.spec.interior_dims_in:
            g, b = callout_interior(ctx, z)
        else:
            continue
        g["slot"] = zone_name
        callouts.append(g)
        callout_boxes[cid] = b

    ctx.cache["static"] = (title_l, title_box, callouts, callout_boxes)
    return ctx.cache["static"]


def build_layout(ctx: Context, p: Params) -> Layout:
    cfg, st = ctx.cfg, ctx.cfg.style
    cw, ch = st.canvas.w, st.canvas.h
    dims = build_dims(ctx.fit, cfg.profile, cfg.spec.dims_in, st, p.offset_ratio, p.height_side, p.extension_lines)
    labels = {d.axis: ctx.text(fmt(d.value, st), "label") for d in dims}
    hull_pts = np.array(ctx.sil.hull.exterior.coords)

    sx0, sy0, sx1, sy1 = safe_px(st)
    sw, sh = sx1 - sx0, sy1 - sy0
    gw, gh = np.ptp(hull_pts[:, 0]), np.ptp(hull_pts[:, 1])
    s = p.group_scale * min(sw / gw, sh / gh)
    for _ in range(6):
        lo, hi = _group_bbox(s, hull_pts, dims, labels, p.label_t)
        bw, bh = hi - lo
        s *= min(p.group_scale * sw / bw, p.group_scale * sh / bh)
    lo, hi = _group_bbox(s, hull_pts, dims, labels, p.label_t)
    center = np.array([(sx0 + sx1) / 2 + p.nudge[0] * cw, (sy0 + sy1) / 2 + p.nudge[1] * ch])
    T = center - (lo + hi) / 2
    group_bbox = (*(lo + T), *(hi + T))

    def C(pt):
        return s * np.asarray(pt, float) + T

    gap_base = st.tokens.label_gap_capheights
    placed = []
    for d in dims:
        a, b = C(d.p0), C(d.p1)
        tb = labels[d.axis]
        anchor = a + p.label_t[d.axis] * (b - a)
        box = (anchor[0] - tb.w / 2, anchor[1] - tb.h / 2, anchor[0] + tb.w / 2, anchor[1] + tb.h / 2)
        segs = split_line(a, b, box, gap_base * tb.cap)
        ext = [(C(e0), C(e1)) for e0, e1 in d.extension]
        placed.append(PlacedDim(d, a, b, tb, anchor, box, segs, ext))

    hull_c = affinity.affine_transform(ctx.sil.hull, [s, 0, 0, s, T[0], T[1]])
    contour_c = affinity.affine_transform(ctx.sil.contour, [s, 0, 0, s, T[0], T[1]])

    title_l, title_box, callouts, callout_boxes = _static(ctx)
    scene = _scene(ctx, p, s, T, placed, title_l, callouts)
    return Layout(p, s, T, placed, hull_c, contour_c, group_bbox, title_box, callout_boxes, scene)


def _r(v):
    return [round(float(x), 2) for x in v]


def _scene(ctx: Context, p: Params, s, T, placed: list[PlacedDim], title_l, callouts) -> dict:
    st = ctx.cfg.style
    cw = st.canvas.w
    stroke = st.tokens.stroke_ratio * cw
    img_scale = s * ctx.work_scale
    layers = [title_l, {
        "id": "product", "type": "image", "src": ctx.photo_src,
        "size": [ctx.photo.width, ctx.photo.height],
        "transform": {"scale": round(img_scale, 6), "tx": round(float(T[0]), 2), "ty": round(float(T[1]), 2)},
    }]
    for d in placed:
        lab, _ = text_layer(f"dim_{d.dim.axis}_label", d.label, d.anchor[0], d.anchor[1], st.tokens.colors.text)
        layers.append({
            "id": f"dim_{d.dim.axis}", "type": "dimension", "axis": d.dim.axis, "value": d.dim.value,
            "line": [_r(d.p0), _r(d.p1)],
            "segments": [[_r(a), _r(b)] for a, b in d.segments],
            "extension": [[_r(a), _r(b)] for a, b in d.extension],
            "ticks": st.annotations.ticks, "tick_len": round(st.annotations.tick_len_ratio * cw, 2),
            "stroke": round(stroke, 2), "color": st.tokens.colors.line,
            "label": lab,
        })
    layers += callouts
    return {
        "canvas": {"w": st.canvas.w, "h": st.canvas.h, "bg": st.canvas.bg},
        "layers": layers,
        "meta": {"sku": ctx.cfg.spec.sku, "style": st.name, "profile": ctx.cfg.profile.name,
                 "params": p.as_dict(), "fit": ctx.fit.qa()},
    }
