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
from .text_metrics import TextBox, _pil, measure_tok

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
    photo_info: object = None  # geometry.photo.PhotoInfo

    @property
    def work_scale(self) -> float:
        return self.fit.image_size[0] / self.photo.width

    @property
    def canvas(self):
        return self.cfg.style.canvas

    def text(self, s: str, tok_name: str) -> TextBox:
        st = self.cfg.style
        return measure_tok(s, getattr(st.tokens.fonts, tok_name), st.canvas.w, self.cfg.font_dirs, st.font_fallbacks)


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
    for r in runs:
        font = _pil(tb.face.path, round(tb.size * r["scale"] * 4))
        n = len(r["text"])
        r["advance"] = (font.getlength(r["text"]) + tb.tracking_px * r["scale"] * max(n - 1, 0)) * tb.h_scale
        adv += r["advance"]
    return runs, adv


def text_layer(id_: str, tb: TextBox, x: float, cy: float, fill: str, align="center", text=None) -> tuple[dict, tuple]:
    """Text centered on cap height at (x, cy). Returns layer and box."""
    text = tb.text if text is None else text
    runs, adv = rich_runs(text, tb)
    baseline = cy + tb.cap / 2
    x0 = x - adv / 2 if align == "center" else x
    layer = {"id": id_, "type": "text", "text": text, "runs": runs,
             "font": {"family": tb.requested, "postscript": tb.postscript, "substituted": tb.substituted,
                      "render_family": tb.face.family, "weight": tb.requested_weight,
                      "render_weight": tb.face.weight, "file": Path(tb.face.path).name,
                      "size": round(tb.size, 2), "tracking_px": round(tb.tracking_px, 2),
                      "h_scale": tb.h_scale, "cap": round(tb.cap, 2)},
             "x": round(x0, 2), "y": round(baseline, 2), "align": "left", "fill": fill,
             "box": [round(v, 2) for v in (x0, cy - tb.cap / 2, x0 + adv, cy + tb.cap / 2)]}
    return layer, tuple(layer["box"])


def zone_px(zone, canvas) -> tuple:
    return (zone.x * canvas.w, zone.y * canvas.h, (zone.x + zone.w) * canvas.w, (zone.y + zone.h) * canvas.h)


def safe_px(style) -> tuple:
    c, m = style.canvas, style.layout.safe_area
    return (m.left * c.w, m.top * c.h, (1 - m.right) * c.w, (1 - m.bottom) * c.h)


# ------------------------------------------------------------------ callouts

def _arc(cx, cy, r, a0, a1, n=6):
    t = np.linspace(np.radians(a0), np.radians(a1), n)
    return [(cx + r * np.cos(v), cy + r * np.sin(v)) for v in t]


def _rounded_top_rect(x0, y0, x1, y1, r):
    """Polygon (y down): rounded top corners, square bottom."""
    return ([(x0, y1)] + _arc(x0 + r, y0 + r, r, 180, 270) + _arc(x1 - r, y0 + r, r, 270, 360) + [(x1, y1)])


def can_polygons(cx: float, top: float, w: float, bottom: float) -> dict[str, list]:
    """Can silhouette as polygons (vector shapes in Photoshop), proportions from the Yukon export."""
    tab_w, tab_h = 0.22 * w, 0.21 * w
    neck_w, neck_h, shoulder_h = 0.873 * w, 0.115 * w, 0.135 * w
    y_neck = top + tab_h
    y_sh = y_neck + neck_h
    y_body = y_sh + shoulder_h
    return {
        "tab": _rounded_top_rect(cx - tab_w / 2, top, cx + tab_w / 2, y_neck + 1, tab_w * 0.3),
        "neck": _rounded_top_rect(cx - neck_w / 2, y_neck, cx + neck_w / 2, y_sh + 1, neck_h * 0.4),
        "body": [(cx - neck_w / 2, y_sh), (cx + neck_w / 2, y_sh), (cx + w / 2, y_body), (cx + w / 2, bottom),
                 (cx - w / 2, bottom), (cx - w / 2, y_body)],
    }


def _poly_layer(id_, pts, fill):
    return {"id": id_, "type": "polygon", "points": [[round(x, 2), round(y, 2)] for x, y in pts], "fill": fill}


def callout_cans(ctx: Context) -> tuple[dict, tuple]:
    st = ctx.cfg.style
    col, g = st.tokens.colors, st.layout.cans
    cw, ch = st.canvas.w, st.canvas.h
    cx, can_w = g.center_x * cw, g.can_w * cw
    head = ctx.text("HOLDS UP TO", "callout_text_bold")
    head_l, head_box = text_layer("cans_head", head, cx, g.head_cy * ch, col.callout_text)
    tok = st.tokens.fonts.callout_num
    num = ctx.text(str(ctx.cfg.spec.can_count), "callout_num")
    fit = min(1.0, g.num_fill * can_w / num.advance)
    if fit < 1.0:  # shrink to fit the can
        from .text_metrics import measure_tok
        num = measure_tok(str(ctx.cfg.spec.can_count), tok, cw, ctx.cfg.font_dirs, st.font_fallbacks, scale=fit)
    word = ctx.text("CANS", "callout_word")
    bottom = ch + 4 if g.bleed else ch * 0.995
    polys = can_polygons(cx, g.can_top * ch, can_w, bottom)
    shapes = [_poly_layer("can_body", polys["body"], col.can_fill),
              _poly_layer("can_neck", polys["neck"], col.can_neck),
              _poly_layer("can_tab", polys["tab"], col.can_tab)]
    num_l, _ = text_layer("cans_num", num, cx, g.num_cy * ch, col.text)
    word_l, _ = text_layer("cans_word", word, cx, g.word_cy * ch, col.text)
    group = {"id": "callout_cans", "type": "group", "children": [head_l, *shapes, num_l, word_l]}
    return group, (min(head_box[0], cx - can_w / 2), head_box[1], max(head_box[2], cx + can_w / 2), bottom)


def callout_interior(ctx: Context) -> tuple[dict, tuple]:
    st = ctx.cfg.style
    g = st.layout.interior
    d = ctx.cfg.spec.interior_dims_in
    cw, ch = st.canvas.w, st.canvas.h
    l1 = ctx.text("INTERIOR DIMENSIONS (L \u00d7 W \u00d7 H):", "callout_text")
    l2 = ctx.text(" \u00d7 ".join(fmt(v, st) for v in (d.L, d.W, d.H)), "callout_value")
    a, ab = text_layer("interior_label", l1, g.cx * cw, g.l1_cy * ch, st.tokens.colors.callout_text)
    b, bb = text_layer("interior_value", l2, g.cx * cw, g.l2_cy * ch, st.tokens.colors.text)
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
        if cid == "cans" and cfg.spec.can_count:
            g, b = callout_cans(ctx)
        elif cid == "interior" and cfg.spec.interior_dims_in:
            g, b = callout_interior(ctx)
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
    pl = st.layout.placement
    if pl is None:
        for _ in range(6):
            lo, hi = _group_bbox(s, hull_pts, dims, labels, p.label_t)
            bw, bh = hi - lo
            s *= min(p.group_scale * sw / bw, p.group_scale * sh / bh)
        lo, hi = _group_bbox(s, hull_pts, dims, labels, p.label_t)
        center = np.array([(sx0 + sx1) / 2 + p.nudge[0] * cw, (sy0 + sy1) / 2 + p.nudge[1] * ch])
        T = center - (lo + hi) / 2
    else:
        # template rule: product bbox centered at product_center_x, group bottom on group_bottom;
        # scale is the largest that keeps the group inside the safe area on every side
        hx = (hull_pts[:, 0].min() + hull_pts[:, 0].max()) / 2
        ax = (pl.product_center_x + p.nudge[0]) * cw
        ay = pl.group_bottom * ch
        for _ in range(8):
            lo, hi = _group_bbox(s, hull_pts, dims, labels, p.label_t)
            px = s * hx
            need = np.array([px - lo[0], hi[0] - px, hi[1] - lo[1]])
            room = np.array([ax - sx0, sx1 - ax, ay - sy0])
            s *= p.group_scale * float(np.min(room / np.maximum(need, 1e-9)))
        lo, hi = _group_bbox(s, hull_pts, dims, labels, p.label_t)
        T = np.array([ax - s * hx, ay - hi[1]])
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


def _alpha_bbox(ctx: Context, img_scale: float, T) -> list:
    if "alpha_bbox" not in ctx.cache:
        ctx.cache["alpha_bbox"] = ctx.photo.getchannel("A").point(lambda v: 255 if v > 0 else 0).getbbox()
    x0, y0, x1, y1 = ctx.cache["alpha_bbox"]
    return _r([x0 * img_scale + T[0], y0 * img_scale + T[1], x1 * img_scale + T[0], y1 * img_scale + T[1]])


def _r(v):
    return [round(float(x), 2) for x in v]


def _scene(ctx: Context, p: Params, s, T, placed: list[PlacedDim], title_l, callouts) -> dict:
    st = ctx.cfg.style
    cw = st.canvas.w
    stroke = st.tokens.stroke_ratio * cw
    img_scale = s * ctx.work_scale
    layers = [title_l, {
        "id": "product", "type": "image", "src": ctx.photo_src, "src_name": Path(ctx.photo_src).name,
        "size": [ctx.photo.width, ctx.photo.height],
        "transform": {"scale": round(img_scale, 6), "tx": round(float(T[0]), 2), "ty": round(float(T[1]), 2)},
        # where the product's visible pixels land on the canvas; the Photoshop script scales the
        # full-res TIFF to this box, so the layout photo can be any smaller copy of it
        "alpha_bbox": _alpha_bbox(ctx, img_scale, T),
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
            "gap": round(st.tokens.label_gap_capheights * d.label.cap, 2),
            "label": lab,
        })
    layers += callouts
    return {
        "canvas": {"w": st.canvas.w, "h": st.canvas.h, "bg": st.canvas.bg},
        "layers": layers,
        "meta": {"sku": ctx.cfg.spec.sku, "style": st.name, "profile": ctx.cfg.profile.name,
                 "params": p.as_dict(), "fit": ctx.fit.qa()},
    }
