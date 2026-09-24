"""Box fit = pose estimation of a known cuboid.

Auto: optimize (yaw, elevation, roll, distance, shift) inside the profile's
bounds so the projected box CONTAINS the silhouette hull with minimal excess
(the product is not a cuboid: lids overhang, handles protrude, corners float).

Assisted: the same objective plus human evidence. Clicked box corners rarely
exist on real products (rounded, overhung), so the primary evidence is
*lines*: straight features known to run along L, W or H. They fix the
vanishing points, i.e. focal length and yaw, which the silhouette cannot.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize
from shapely.geometry import MultiPoint, Polygon

from .box import Box
from .camera import Intrinsics, Pose, angles_from_rotation, camera_from_params, elevation_deg, project
from .silhouette import Silhouette


@dataclass
class FitResult:
    pose: Pose
    K: Intrinsics
    box: Box
    image_size: tuple[int, int]  # working (w, h)
    method: str  # auto | manual
    uncovered: float
    excess: float
    status: str  # ok | needs_manual
    reproj_rms: float | None = None
    verticals: str = "plumb"
    notes: list[str] = field(default_factory=list)
    line_rms_deg: float | None = None

    @property
    def angles(self) -> tuple[float, float, float]:
        return angles_from_rotation(self.pose.R)

    def projected_polygon(self) -> Polygon:
        return box_polygon(self.box, self.pose, self.K)

    def qa(self) -> dict:
        yaw, _, roll = self.angles
        return {
            "method": self.method, "status": self.status,
            "uncovered": round(self.uncovered, 4), "excess": round(self.excess, 4),
            "corner_rms_px": None if self.reproj_rms is None else round(self.reproj_rms, 2),
            "line_rms_deg": None if self.line_rms_deg is None else round(self.line_rms_deg, 3),
            "verticals": self.verticals, "yaw_deg": round(yaw, 2),
            "elevation_deg": round(elevation_deg(self.pose, self.box.center), 2), "roll_deg": round(roll, 2),
            "focal_px": round(self.K.f, 1), "principal_point": [round(self.K.cx, 1), round(self.K.cy, 1)],
            "distance_in": round(float(np.linalg.norm(self.pose.center - self.box.center)), 2),
            "notes": self.notes,
        }

    def to_json(self) -> dict:
        return {
            "R": self.pose.R.tolist(), "t": self.pose.t.tolist(),
            "K": [self.K.f, self.K.cx, self.K.cy], "box": [self.box.sx, self.box.sy, self.box.sz],
            "image_size": list(self.image_size), "method": self.method,
            "uncovered": self.uncovered, "excess": self.excess, "status": self.status,
            "reproj_rms": self.reproj_rms, "line_rms_deg": self.line_rms_deg,
            "verticals": self.verticals, "notes": self.notes,
        }

    @classmethod
    def from_json(cls, d: dict) -> "FitResult":
        return cls(Pose(np.array(d["R"]), np.array(d["t"])), Intrinsics(*d["K"]), Box(*d["box"]),
                   tuple(d["image_size"]), d["method"], d["uncovered"], d["excess"], d["status"],
                   d.get("reproj_rms"), d.get("verticals", "plumb"), d.get("notes", []),
                   d.get("line_rms_deg"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2))


def box_polygon(box: Box, pose: Pose, K: Intrinsics) -> Polygon:
    return MultiPoint([tuple(p) for p in project(box.array, pose, K)]).convex_hull


def coverage(hull: Polygon, proj: Polygon) -> tuple[float, float]:
    uncovered = hull.difference(proj).area / hull.area
    excess = proj.difference(hull).area / max(proj.area, 1e-9)
    return uncovered, excess


# ------------------------------------------------------------------ evidence

@dataclass
class Evidence:
    """Human-supplied constraints, in working-image pixels.

    lines:   axis ("L" | "W" | "H") -> segments lying along that box axis
             (any straight product feature: a gasket band, a lid edge, a seam).
             These pin focal length and yaw, which a silhouette alone cannot.
    corners: box vertex name -> pixel. Only usable where a real corner exists.
    """
    lines: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]] = field(default_factory=dict)
    corners: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __bool__(self):
        return bool(self.lines or self.corners)

    def scaled(self, s: float) -> "Evidence":
        return Evidence({a: [((p[0] * s, p[1] * s), (q[0] * s, q[1] * s)) for p, q in segs]
                         for a, segs in self.lines.items()},
                        {k: (v[0] * s, v[1] * s) for k, v in self.corners.items()})

    @classmethod
    def load(cls, path: Path) -> "Evidence":
        d = json.loads(Path(path).read_text())
        return cls({a: [tuple(map(tuple, seg)) for seg in segs] for a, segs in d.get("lines", {}).items()},
                   {k: tuple(v) for k, v in d.get("corners", {}).items()})


def axis_vector(axis: str, axis_map) -> np.ndarray:
    if axis == "H":
        return np.array([0.0, 0.0, 1.0])
    return np.array([1.0, 0.0, 0.0]) if axis_map.front_face == axis else np.array([0.0, 1.0, 0.0])


def line_residuals(pose: Pose, K: Intrinsics, lines, axis_map) -> np.ndarray:
    """Sine of the angle between each segment and the direction to its axis' vanishing point."""
    out = []
    for axis, segs in lines.items():
        v = K.K @ pose.R @ axis_vector(axis, axis_map)  # homogeneous VP (w=0 at infinity)
        for p, q in segs:
            p, q = np.asarray(p, float), np.asarray(q, float)
            m = (p + q) / 2
            to_vp = v[:2] - m * v[2]
            seg = q - p
            n1, n2 = np.linalg.norm(seg), np.linalg.norm(to_vp)
            out.append(0.0 if n2 < 1e-12 else float((seg[0] * to_vp[1] - seg[1] * to_vp[0]) / (n1 * n2)))
    return np.array(out)


# ------------------------------------------------------------------ model

class Model:
    """Maps (yaw, elev, roll, 10 ln d, a, b[, 10 ln f]) to (Pose, Intrinsics)."""

    def __init__(self, box: Box, K0: Intrinsics, verticals: str, free_focal: bool = False):
        self.box, self.K0, self.verticals, self.free_focal = box, K0, verticals, free_focal
        self.steps = np.array([3.0, 3.0, 1.0, 0.5, 0.5, 0.5] + ([1.0] if free_focal else []))

    def camera(self, x) -> tuple[Pose, Intrinsics]:
        yaw, elev, roll, logd, a, b = x[:6]
        K0 = self.K0
        if self.free_focal:
            K0 = Intrinsics(float(np.exp(x[6] / 10.0)), K0.cx, K0.cy)
        return camera_from_params(yaw, elev, roll, float(np.exp(logd / 10.0)), a, b,
                                  K0, self.box.center, self.verticals)

    def project(self, x, pts) -> np.ndarray:
        pose, K = self.camera(x)
        return project(pts, pose, K)

    def init(self, yaw, elev, roll, target_bounds, f=None) -> np.ndarray:
        """Distance and shift so the projected box bbox matches target bounds."""
        x0, y0, x1, y1 = target_bounds
        tail = [10 * np.log(f or self.K0.f)] if self.free_focal else []
        fpx = f or self.K0.f
        d, a, b = 4 * self.box.diagonal * fpx / self.K0.f, 0.0, 0.0
        for _ in range(10):
            uv = self.project([yaw, elev, roll, 10 * np.log(d), a, b, *tail], self.box.array)
            d *= np.ptp(uv[:, 1]) / (y1 - y0)
            uv = self.project([yaw, elev, roll, 10 * np.log(d), a, b, *tail], self.box.array)
            du = (x0 + x1) / 2 - (uv[:, 0].min() + uv[:, 0].max()) / 2
            dv = (y0 + y1) / 2 - (uv[:, 1].min() + uv[:, 1].max()) / 2
            a += du * d / fpx
            b += dv * d / fpx
        return np.array([yaw, elev, roll, 10 * np.log(d), a, b, *tail])


def _angle_box(profile):
    nom, b = profile.nominal_pose, profile.fit_bounds
    mid = np.array([nom.yaw_deg, nom.pitch_deg, nom.roll_deg])
    lo = mid + np.array([b.yaw_deg[0], b.pitch_deg[0], b.roll_deg[0]])
    hi = mid + np.array([b.yaw_deg[1], b.pitch_deg[1], b.roll_deg[1]])
    return mid, lo, hi, np.maximum((hi - lo) / 2, 1e-6)


# ------------------------------------------------------------------ fit

def fit(sil: Silhouette, box: Box, K0: Intrinsics, profile, fit_cfg, image_size: tuple[int, int],
        evidence: Evidence | None = None) -> FitResult:
    """Containment of the silhouette, plus any human evidence, in one objective.

    Without evidence this is the plan's auto fit. Lines free the focal length
    (within profile.camera.focal_bounds_at_2000); corners add reprojection error.
    """
    evidence = evidence or Evidence()
    free_f = bool(evidence.lines)
    model = Model(box, K0, profile.camera.verticals, free_focal=free_f)
    mid, lo, hi, half = _angle_box(profile)
    scale = max(image_size) / 2000.0
    flo, fhi = (np.array(profile.camera.focal_bounds_at_2000) * scale if free_f else (K0.f, K0.f))
    # containment tolerance: anti-aliased / JPEG edges put ~1px of "product" outside the true outline
    hull = sil.hull.buffer(-fit_cfg.hull_tolerance_px * scale)
    cnames = [n for n in evidence.corners if n in box.vertices]
    if set(evidence.corners) - set(cnames):
        raise ValueError(f"unknown corner names {sorted(set(evidence.corners) - set(cnames))}; valid: {box.names}")
    cobj = np.array([box.vertices[n] for n in cnames]).reshape(-1, 3)
    cimg = np.array([evidence.corners[n] for n in cnames]).reshape(-1, 2)
    px_unit = 0.005 * max(image_size)

    def terms(x):
        pose, K = model.camera(x)
        unc, exc = coverage(hull, box_polygon(box, pose, K))
        t = {"uncovered": unc, "excess": exc}
        if evidence.lines:
            t["line_sin_rms"] = float(np.sqrt(np.mean(line_residuals(pose, K, evidence.lines, profile.axis_map) ** 2)))
        if len(cnames):
            err = project(cobj, pose, K) - cimg
            t["corner_rms_px"] = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))
        return t

    def cost(x):
        ang = x[:3]
        out = np.clip((ang - hi) / half, 0, None) + np.clip((lo - ang) / half, 0, None)
        pen = 1e3 * float(np.sum(out ** 2))
        if free_f:
            f = np.exp(x[6] / 10.0)
            pen += 1e3 * (max(0.0, np.log(f / fhi)) + max(0.0, np.log(flo / f))) ** 2
        try:
            t = terms(x)
        except ValueError:
            return 1e6
        prior = float(np.sum(((ang - mid) / half) ** 2))
        c = fit_cfg.w_uncovered * t["uncovered"] + fit_cfg.w_excess * t["excess"] + fit_cfg.w_prior * prior + pen
        c += fit_cfg.w_lines * t.get("line_sin_rms", 0.0) ** 2
        c += fit_cfg.w_corners * (t.get("corner_rms_px", 0.0) / px_unit) ** 2
        return c

    n = len(model.steps)

    def nm(x0, s, tol):
        simplex = np.vstack([x0] + [x0 + np.eye(n)[i] * model.steps[i] * s for i in range(n)])
        return minimize(cost, x0, method="Nelder-Mead",
                        options={"initial_simplex": simplex, "xatol": tol, "fatol": tol ** 2, "maxiter": 6000})

    # screen many cheap starts (bbox-matched init only), then run Nelder-Mead from the best few:
    # in plumb mode elevation trades off against the principal-point shift, which leaves local minima
    rng = np.random.default_rng(0)
    starts = [(mid, None)]
    if free_f:
        starts += [(mid, float(flo)), (mid, float(np.sqrt(flo * fhi))), (mid, float(fhi))]
    for _ in range(fit_cfg.screen_starts):
        f = float(np.exp(rng.uniform(np.log(flo), np.log(fhi)))) if free_f else None
        starts.append((lo + rng.random(3) * (hi - lo), f))
    x0s = sorted((model.init(*ang, hull.bounds, f) for ang, f in starts), key=cost)
    best = min((nm(x0, 1.0, 1e-4) for x0 in x0s[: fit_cfg.restarts]), key=lambda r: r.fun)
    # Nelder-Mead collapses its simplex on this piecewise-smooth area cost; restarting
    # from a fresh simplex around the incumbent is the standard remedy
    for _ in range(fit_cfg.polish_rounds):
        improved = False
        for s in (1.0, 0.3, 0.1):
            r = nm(best.x, s, 1e-6)
            if r.fun < best.fun - 1e-9:
                best, improved = r, True
        if not improved:
            break

    x = best.x
    pose, K = model.camera(x)
    t = terms(x)
    res = FitResult(pose, K, box, image_size, "assisted" if evidence else "auto",
                    t["uncovered"], t["excess"], "ok", reproj_rms=t.get("corner_rms_px"),
                    verticals=model.verticals)
    res.line_rms_deg = float(np.degrees(np.arcsin(min(1.0, t["line_sin_rms"])))) if "line_sin_rms" in t else None
    if t["uncovered"] >= fit_cfg.max_uncovered:
        res.notes.append(f"uncovered {t['uncovered']:.3f} >= max {fit_cfg.max_uncovered}")
    elo, ehi = profile.expected_excess
    if not (elo <= t["excess"] <= ehi):
        res.notes.append(f"excess {t['excess']:.3f} outside expected [{elo}, {ehi}]")
    if res.line_rms_deg is not None and res.line_rms_deg > fit_cfg.max_line_deg:
        res.notes.append(f"line evidence off by {res.line_rms_deg:.2f} deg rms > {fit_cfg.max_line_deg}")
    if res.reproj_rms is not None and res.reproj_rms > fit_cfg.max_corner_px * scale:
        res.notes.append(f"corner reprojection {res.reproj_rms:.1f}px rms; re-click corners")
    if res.notes:
        res.status = "needs_manual"
    ang = x[:3]
    if np.any(np.abs(ang - lo) < 0.25) or np.any(np.abs(ang - hi) < 0.25):
        res.notes.append("pose hit a fit_bounds limit; check nominal_pose")
    if not evidence and profile.camera.calibrated is False:
        res.notes.append("focal uncalibrated and no line evidence: receding-edge angles are a guess")
    return res
