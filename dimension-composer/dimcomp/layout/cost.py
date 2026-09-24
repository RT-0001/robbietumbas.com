"""Cost terms on vector geometry. Every term is reported raw and weighted."""
from __future__ import annotations

import itertools

import numpy as np
from shapely.geometry import LineString, box as rect
from shapely.ops import unary_union

from .scene import Context, Layout, safe_px, zone_px

HARD = ("hard_product_clearance", "hard_label_overlap", "hard_outside_safe", "hard_reserved", "hard_legibility")


def _elements(lay: Layout):
    """(name, geometry) for everything the annotation group draws."""
    for d in lay.dims:
        yield f"label_{d.dim.axis}", rect(*d.label_box)
        if d.segments or d.extension:
            yield f"line_{d.dim.axis}", d.geom


def terms(ctx: Context, lay: Layout) -> dict[str, float]:
    st = ctx.cfg.style
    cw, ch = st.canvas.w, st.canvas.h
    t: dict[str, float] = {}

    # --- hard -----------------------------------------------------------
    clear = st.annotations.clearance_ratio * cw
    keep_out = lay.hull.buffer(clear)
    bad = 0.0
    for d in lay.dims:
        lb = rect(*d.label_box)
        bad += lb.intersection(keep_out).area / lb.area
        for a, b in d.segments:
            seg = LineString([tuple(a), tuple(b)])
            bad += seg.intersection(keep_out).length / max(seg.length, 1.0)
    t["hard_product_clearance"] = bad

    pad = 0.5 * np.mean([d.label.cap for d in lay.dims])
    boxes = [rect(*d.label_box).buffer(pad) for d in lay.dims]
    t["hard_label_overlap"] = sum(a.intersection(b).area / min(a.area, b.area)
                                  for a, b in itertools.combinations(boxes, 2))

    safe = rect(*safe_px(st))
    group = unary_union([g for _, g in _elements(lay)] + [lay.hull])
    outside = group.difference(safe)
    t["hard_outside_safe"] = (outside.area + outside.length) / (0.01 * cw * ch) if not outside.is_empty else 0.0

    zones = [rect(*zone_px(z, st.canvas)) for z in st.layout.reserved_zones.values()] + [rect(*lay.title_box)]
    hit = 0.0
    for z in zones:
        for _, g in _elements(lay):
            inter = g.intersection(z)
            hit += (inter.area + inter.length) / (0.001 * cw * ch)
        hit += lay.hull.intersection(z).area / (0.001 * cw * ch)
    t["hard_reserved"] = hit

    cap = min(d.label.cap for d in lay.dims)
    t["hard_legibility"] = max(0.0, (st.layout.legibility_min_cap_px - cap) / st.layout.legibility_min_cap_px)

    # --- soft -----------------------------------------------------------
    offs = np.array([d.dim.offset_px * lay.s for d in lay.dims])
    t["offset_equality"] = float(np.var(offs) / max(np.mean(offs) ** 2, 1e-9))
    tgt = st.targets.offset_ratio
    t["offset_target"] = ((lay.params.offset_ratio - tgt) / tgt) ** 2
    t["label_centering"] = float(np.mean([abs(lay.params.label_t[d.dim.axis] - 0.5) for d in lay.dims]))

    sx0, sy0, sx1, sy1 = safe_px(st)
    gx0, gy0, gx1, gy1 = lay.group_bbox
    t["whitespace_balance"] = abs((gx0 - sx0) - (sx1 - gx1)) / (sx1 - sx0) + abs((gy0 - sy0) - (sy1 - gy1)) / (sy1 - sy0)

    ox, oy = st.layout.optical_center_target
    c = lay.contour.centroid
    t["optical_center"] = float(np.hypot(c.x / cw - ox, c.y / ch - oy))

    fill = (gx1 - gx0) * (gy1 - gy0) / ((sx1 - sx0) * (sy1 - sy0))
    t["fill"] = abs(fill - st.targets.fill) / st.targets.fill

    side = lay.params.height_side
    cl = ctx.clutter
    # relative to the cleaner side, so the clean choice costs 0
    t["height_side_clutter"] = (cl[side] - min(cl.values())) / max(max(cl.values()), 1e-9)
    t["extension_lines"] = 1.0 if lay.params.extension_lines else 0.0
    t["style_deviation"] = 0.0  # phase 2: distance from learned house-style medians
    return t


def score(ctx: Context, lay: Layout) -> tuple[float, dict]:
    w = ctx.cfg.style.weights
    raw = terms(ctx, lay)
    breakdown = {}
    total = 0.0
    for k, v in raw.items():
        wk = w.hard if k in HARD else getattr(w, k)
        breakdown[k] = {"raw": round(float(v), 5), "weighted": round(float(v * wk), 5)}
        total += v * wk
    return float(total), breakdown
