"""3D dimension annotations -> 2D primitives (working-image pixels).

L and W lines are box bottom edges pushed outward in the ground plane, then
projected, so they converge with the product by construction. H is either the
projected vertical edge ("edge") or a plumb line spanning the projected box end
on that side ("screen_vertical"), which is what hand-built images do.

Offsets are specified VISUALLY (fraction of the projected box diagonal). A 3D
ground offset toward the camera is foreshortened by ~sin(elevation), so equal
3D offsets look wildly unequal; instead we solve, per edge, for the 3D offset
whose projection lands at the requested on-screen distance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry.box import BOTTOM_EDGE_FACE, EDGES, Box
from ..geometry.camera import local_scale, project
from ..geometry.fit import FitResult

VERTICAL_EDGES = ["front_left", "front_right", "back_left", "back_right"]


@dataclass
class Dim:
    axis: str  # L | W | H
    value: float
    p0: np.ndarray  # image px
    p1: np.ndarray
    offset_px: float  # distance from the box edge it measures
    extension: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    side: str | None = None  # H only

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))


class HiddenEdgeError(ValueError):
    pass


def _axis_of_face(face: str, axis_map) -> str:
    return axis_map.front_face if face in ("front", "back") else axis_map.side_face


def _proj(fit: FitResult, pts) -> np.ndarray:
    return project(np.atleast_2d(pts), fit.pose, fit.K)


def _perp(p, a, b) -> float:
    d = b - a
    q = p - a
    return float(abs(d[0] * q[1] - d[1] * q[0]) / max(np.linalg.norm(d), 1e-9))


def solve_offset(fit: FitResult, a, b, direction, target_px: float) -> float:
    """3D offset (inches) along `direction` whose projected distance from edge ab is target_px."""
    m = _proj(fit, (a + b) / 2)[0]
    off = target_px / local_scale((a + b) / 2, fit.pose, fit.K)
    for _ in range(6):
        p0, p1 = _proj(fit, [a + direction * off, b + direction * off])
        seen = _perp(m, p0, p1)
        if seen < 1e-6:
            break
        off *= target_px / seen
    return off


def projected_diagonal(fit: FitResult) -> float:
    uv = np.array(list(fit.box.project(fit.pose, fit.K).values()))
    return float(np.hypot(*np.ptp(uv, axis=0)))


def ground_dims(fit: FitResult, profile, dims, target_px: float, extension: bool,
                corner_gap_px: float | None = None) -> list[Dim]:
    box: Box = fit.box
    visible = set(box.visible_faces(fit.pose))
    lines = []
    for edge_key in profile.visible_bottom_edges:
        face = BOTTOM_EDGE_FACE[edge_key]
        if face not in visible:
            raise HiddenEdgeError(f"profile {profile.name} asks for the {edge_key} bottom edge, "
                                  f"but the {face} face is hidden at the fitted pose")
        ename = f"bottom_{face}"
        a, b = box.edge(ename)
        normal = box.face_normal(face)
        n = normal * solve_offset(fit, a, b, normal, target_px)
        lines.append({"face": face, "verts": EDGES[ename], "a": a, "b": b, "n": n})

    # organized corner: two lines sharing a box corner are extended to their common
    # virtual corner, then each stops corner_gap_px short of it (symmetric gap)
    ends = {id(l): [l["a"] + l["n"], l["b"] + l["n"]] for l in lines}
    trims = {}
    if corner_gap_px is not None:
        for i, la in enumerate(lines):
            for lb in lines[i + 1:]:
                shared = set(la["verts"]) & set(lb["verts"])
                if not shared:
                    continue
                v = box.vertices[shared.pop()]
                corner = v + la["n"] + lb["n"]
                for l in (la, lb):
                    k = 0 if np.allclose(l["a"], v) else 1
                    ends[id(l)][k] = corner
                    trims[(id(l), k)] = corner_gap_px

    out = []
    for l in lines:
        p0, p1 = _proj(fit, ends[id(l)])
        a2, b2 = _proj(fit, [l["a"], l["b"]])
        for k in (0, 1):
            if (id(l), k) in trims:
                end, other = (p0, p1) if k == 0 else (p1, p0)
                u = (other - end) / np.linalg.norm(other - end)
                moved = end + u * trims[(id(l), k)]
                p0, p1 = (moved, p1) if k == 0 else (p0, moved)
        axis = _axis_of_face(l["face"], profile.axis_map)
        off_px = _perp((a2 + b2) / 2, p0, p1)
        ext = [(a2, p0), (b2, p1)] if extension else []
        out.append(Dim(axis, getattr(dims, axis), p0, p1, off_px, ext))
    return out


def _line_intersect(p, u, q, v):
    """Intersection of 2D lines p + s*u and q + t*v (None if parallel)."""
    den = u[0] * v[1] - u[1] * v[0]
    if abs(den) < 1e-9:
        return None
    s = ((q[0] - p[0]) * v[1] - (q[1] - p[1]) * v[0]) / den
    return p + s * u


def screen_dims(fit: FitResult, profile, dims, target_px: float, extension: bool,
                corner_gap_px: float | None = None, sil=None) -> list[Dim]:
    """How a designer draws them: each visible bottom edge, projected, then slid straight out
    (perpendicular on screen, away from the product) by target_px. The line keeps the edge's
    perspective angle and length; its ends sit square to the product's corners."""
    box: Box = fit.box
    visible = set(box.visible_faces(fit.pose))
    uv = box.project(fit.pose, fit.K)
    center = np.mean(list(uv.values()), axis=0)
    lines = []
    for edge_key in profile.visible_bottom_edges:
        face = BOTTOM_EDGE_FACE[edge_key]
        if face not in visible:
            raise HiddenEdgeError(f"profile {profile.name} asks for the {edge_key} bottom edge, "
                                  f"but the {face} face is hidden at the fitted pose")
        va, vb = EDGES[f"bottom_{face}"]
        a2, b2 = uv[va], uv[vb]
        u = (b2 - a2) / np.linalg.norm(b2 - a2)
        n = np.array([-u[1], u[0]])
        if np.dot(n, (a2 + b2) / 2 - center) < 0:
            n = -n
        # distance is measured from the product OUTLINE (what the eye sees), not the box edge,
        # which floats off rounded corners
        off = target_px
        if sil is not None:
            pts = np.asarray(sil.contour.exterior.coords)
            t = (pts - a2) @ u
            span = pts[(t >= 0) & (t <= np.linalg.norm(b2 - a2))]
            if len(span):
                off = float(((span - a2) @ n).max()) + target_px
        lines.append({"face": face, "verts": (va, vb), "e": [a2, b2], "p": [a2 + n * off, b2 + n * off], "u": u})

    if corner_gap_px is not None:
        for i, la in enumerate(lines):
            for lb in lines[i + 1:]:
                shared = set(la["verts"]) & set(lb["verts"])
                if not shared:
                    continue
                v = shared.pop()
                x = _line_intersect(la["p"][0], la["u"], lb["p"][0], lb["u"])
                if x is None:
                    continue
                for l in (la, lb):
                    k = l["verts"].index(v)
                    other = l["p"][1 - k]
                    d = (other - x) / np.linalg.norm(other - x)
                    l["p"][k] = x + d * corner_gap_px

    out = []
    for l in lines:
        p0, p1 = l["p"]
        axis = _axis_of_face(l["face"], profile.axis_map)
        ext = [(l["e"][0], p0), (l["e"][1], p1)] if extension else []
        out.append(Dim(axis, getattr(dims, axis), p0, p1, target_px, ext))
    return out


def silhouette_edges(fit: FitResult) -> dict[str, str]:
    """Which vertical box edge forms the left / right image silhouette."""
    xs = {e: float(_proj(fit, fit.box.edge(e)[0])[0, 0]) for e in VERTICAL_EDGES}
    return {"left": min(xs, key=xs.get), "right": max(xs, key=xs.get)}


def height_dim(fit: FitResult, dims, target_px: float, side: str, mode: str, extension: bool,
               sil=None, band: float = 0.06) -> Dim:
    box = fit.box
    edge = silhouette_edges(fit)[side]
    bottom, top = box.edge(edge)

    if mode == "edge":
        outward = np.array([np.sign(bottom[0]), np.sign(bottom[1]), 0.0]) / np.sqrt(2)
        n = outward * solve_offset(fit, bottom, top, outward, target_px)
        p0, p1 = _proj(fit, [bottom + n, top + n])
        b2, t2 = _proj(fit, [bottom, top])
        ext = [(b2, p0), (t2, p1)] if extension else []
        return Dim("H", dims.H, p0, p1, _perp((b2 + t2) / 2, p0, p1), ext, side)

    off_px = target_px

    if mode == "silhouette" and sil is not None:
        # plumb line spanning the product's VISIBLE end on this side (what a designer traces),
        # set out from the outline's extreme point
        pts = np.asarray(sil.contour.exterior.coords)
        x0, x1 = pts[:, 0].min(), pts[:, 0].max()
        w = x1 - x0
        sel = pts[:, 0] >= x1 - band * w if side == "right" else pts[:, 0] <= x0 + band * w
        y_top, y_bot = pts[sel, 1].min(), pts[sel, 1].max()
        x = x1 + off_px if side == "right" else x0 - off_px
        p0, p1 = np.array([x, y_bot]), np.array([x, y_top])
        ext = []
        if extension:
            ext = [(pts[sel][np.argmax(pts[sel, 1])], p0), (pts[sel][np.argmin(pts[sel, 1])], p1)]
        return Dim("H", dims.H, p0, p1, off_px, ext, side)

    # screen_vertical: plumb line beside the box end on this side, spanning that end's projection
    uv = box.project(fit.pose, fit.K)
    xs = {e: float(uv[EDGES[e][0]][0]) for e in VERTICAL_EDGES}
    ends = sorted(xs, key=xs.get)[-2:] if side == "right" else sorted(xs, key=xs.get)[:2]
    verts = [uv[v] for e in ends for v in EDGES[e]]
    ys = [p[1] for p in verts]
    all_x = [p[0] for p in uv.values()]
    x = (max(all_x) + off_px) if side == "right" else (min(all_x) - off_px)
    p0, p1 = np.array([x, max(ys)]), np.array([x, min(ys)])
    ext = []
    if extension:
        lo = max(verts, key=lambda p: p[1])
        hi = min(verts, key=lambda p: p[1])
        ext = [(lo, p0), (hi, p1)]
    return Dim("H", dims.H, p0, p1, off_px, ext, side)


def build_dims(fit: FitResult, profile, dims, style, offset_ratio: float, height_side: str,
               extension: bool, sil=None) -> list[Dim]:
    diag = projected_diagonal(fit)
    target_px = offset_ratio * diag
    gap = style.annotations.corner_gap_ratio
    gap_px = None if gap is None else gap * diag
    if style.annotations.offset_mode == "screen":
        out = screen_dims(fit, profile, dims, target_px, extension, gap_px, sil)
    else:
        out = ground_dims(fit, profile, dims, target_px, extension, gap_px)
    out.append(height_dim(fit, dims, target_px, height_side, style.annotations.height_mode, extension, sil))
    return out
