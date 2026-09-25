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
    body_aspect: tuple[float, float] | None = None  # fitted body L, W as fraction of spec (stage 1)

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
            "body_aspect": self.body_aspect,
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
            "reproj_rms": self.reproj_rms, "line_rms_deg": self.line_rms_deg, "body_aspect": self.body_aspect,
            "verticals": self.verticals, "notes": self.notes,
        }

    @classmethod
    def from_json(cls, d: dict) -> "FitResult":
        return cls(Pose(np.array(d["R"]), np.array(d["t"])), Intrinsics(*d["K"]), Box(*d["box"]),
                   tuple(d["image_size"]), d["method"], d["uncovered"], d["excess"], d["status"],
                   d.get("reproj_rms"), d.get("verticals", "plumb"), d.get("notes", []),
                   d.get("line_rms_deg"), d.get("body_aspect"))

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
    """Maps a parameter vector to (Box, Pose, Intrinsics).

    x = (yaw, elev, roll, 10 ln d, a, b[, 10 ln f][, 10 ln kx, 10 ln ky])
    free_aspect lets the box's L and W scale independently (H fixed), so the
    pose can be fitted to the product BODY whose proportions differ from the
    spec box whenever handles or lid overhangs count toward the spec dims.
    """

    def __init__(self, box: Box, K0: Intrinsics, verticals: str, free_focal=False, free_aspect=False,
                 fixed_angles=None):
        self.box, self.K0, self.verticals = box, K0, verticals
        self.free_focal, self.free_aspect = free_focal, free_aspect
        self.fixed_angles = fixed_angles  # (yaw, elev, roll): then x = (10 ln d, a, b)
        if fixed_angles is not None:
            self.steps = np.array([0.5, 0.5, 0.5])
        else:
            self.steps = np.array([3.0, 3.0, 1.0, 0.5, 0.5, 0.5] + ([1.0] if free_focal else [])
                                  + ([0.5, 0.5] if free_aspect else []))

    def full(self, x) -> np.ndarray:
        return np.array([*self.fixed_angles, *x]) if self.fixed_angles is not None else np.asarray(x)

    def box_at(self, x) -> Box:
        if not self.free_aspect or self.fixed_angles is not None:
            return self.box
        kx, ky = np.exp(np.asarray(x[-2:]) / 10.0)
        return Box(self.box.sx * kx, self.box.sy * ky, self.box.sz)

    def aspect(self, x) -> tuple[float, float]:
        if not self.free_aspect or self.fixed_angles is not None:
            return 1.0, 1.0
        kx, ky = np.exp(np.asarray(x[-2:]) / 10.0)
        return float(kx), float(ky)

    def camera(self, x) -> tuple[Pose, Intrinsics]:
        v = self.full(x)
        yaw, elev, roll, logd, a, b = v[:6]
        K0 = self.K0
        if self.free_focal and self.fixed_angles is None:
            K0 = Intrinsics(float(np.exp(v[6] / 10.0)), K0.cx, K0.cy)
        return camera_from_params(yaw, elev, roll, float(np.exp(logd / 10.0)), a, b,
                                  K0, self.box.center, self.verticals)

    def polygon(self, x) -> Polygon:
        pose, K = self.camera(x)
        return box_polygon(self.box_at(x), pose, K)

    def init(self, yaw, elev, roll, target_bounds, f=None) -> np.ndarray:
        """Distance and shift so the projected box bbox matches target bounds."""
        x0, y0, x1, y1 = target_bounds
        fpx = f or self.K0.f
        tail = [10 * np.log(fpx)] if self.free_focal else []
        tail += [0.0, 0.0] if self.free_aspect else []
        d, a, b = 4 * self.box.diagonal * fpx / self.K0.f, 0.0, 0.0
        for _ in range(10):
            uv = project(self.box.array, *self.camera([yaw, elev, roll, 10 * np.log(d), a, b, *tail]))
            d *= np.ptp(uv[:, 1]) / (y1 - y0)
            uv = project(self.box.array, *self.camera([yaw, elev, roll, 10 * np.log(d), a, b, *tail]))
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


def _nm(cost, x0, steps, s, tol):
    n = len(x0)
    simplex = np.vstack([x0] + [x0 + np.eye(n)[i] * steps[i] * s for i in range(n)])
    return minimize(cost, x0, method="Nelder-Mead",
                    options={"initial_simplex": simplex, "xatol": tol, "fatol": tol ** 2, "maxiter": 6000})


def _polish(cost, best, steps, rounds):
    # Nelder-Mead collapses its simplex on this piecewise-smooth area cost; restarting
    # from a fresh simplex around the incumbent is the standard remedy
    for _ in range(rounds):
        improved = False
        for s in (1.0, 0.3, 0.1):
            r = _nm(cost, best.x, steps, s, 1e-6)
            if r.fun < best.fun - 1e-9:
                best, improved = r, True
        if not improved:
            break
    return best


# ------------------------------------------------------------------ fit

def fit(sil: Silhouette, box: Box, K0: Intrinsics, profile, fit_cfg, image_size: tuple[int, int],
        evidence: Evidence | None = None) -> FitResult:
    """Two stages.

    1. ANGLES from the product body: containment of the core silhouette (handles
       and other thin protrusions opened away) with the box's L/W proportions
       free, plus any line evidence. Handles can no longer tilt the box.
    2. Default: the fitted BODY box is the result; annotations hug the product.
       With fit.dims_include_protrusions the spec-size box is instead placed
       (angles fixed) to contain the full outline, handles included.
    Corners, when given, are used at the spec box size in both stages.
    """
    evidence = evidence or Evidence()
    free_f = bool(evidence.lines)
    cnames = [n for n in evidence.corners if n in box.vertices]
    if set(evidence.corners) - set(cnames):
        raise ValueError(f"unknown corner names {sorted(set(evidence.corners) - set(cnames))}; valid: {box.names}")
    free_aspect = fit_cfg.body_aspect_free and not cnames
    mid, lo, hi, half = _angle_box(profile)
    scale = max(image_size) / 2000.0
    flo, fhi = (np.array(profile.camera.focal_bounds_at_2000) * scale if free_f else (K0.f, K0.f))
    tol = fit_cfg.hull_tolerance_px * scale
    core = sil.core_hull.buffer(-tol)
    full = sil.hull.buffer(-tol)
    cobj = np.array([box.vertices[n] for n in cnames]).reshape(-1, 3)
    cimg = np.array([evidence.corners[n] for n in cnames]).reshape(-1, 2)
    px_unit = 0.005 * max(image_size)
    klo, khi = np.log(fit_cfg.body_aspect_bounds)

    def evidence_terms(pose, K, t):
        if evidence.lines:
            t["line_sin_rms"] = float(np.sqrt(np.mean(line_residuals(pose, K, evidence.lines, profile.axis_map) ** 2)))
        if len(cnames):
            err = project(cobj, pose, K) - cimg
            t["corner_rms_px"] = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))
        return t

    def evidence_cost(t):
        return (fit_cfg.w_lines * t.get("line_sin_rms", 0.0) ** 2
                + fit_cfg.w_corners * (t.get("corner_rms_px", 0.0) / px_unit) ** 2)

    # ---- stage 1: angles (+ focal) from the body
    m1 = Model(box, K0, profile.camera.verticals, free_focal=free_f, free_aspect=free_aspect)

    def cost1(x):
        ang = x[:3]
        out = np.clip((ang - hi) / half, 0, None) + np.clip((lo - ang) / half, 0, None)
        pen = 1e3 * float(np.sum(out ** 2))
        if free_f:
            f = np.exp(x[6] / 10.0)
            pen += 1e3 * (max(0.0, np.log(f / fhi)) + max(0.0, np.log(flo / f))) ** 2
        if free_aspect:
            k = np.asarray(x[-2:]) / 10.0
            pen += 1e3 * float(np.sum(np.clip(k - khi, 0, None) ** 2 + np.clip(klo - k, 0, None) ** 2))
            # mild pull toward spec proportions: body only departs when the outline insists
            pen += fit_cfg.w_aspect_prior * float(np.sum(k ** 2)) / 0.01
        try:
            pose, K = m1.camera(x)
            unc, exc = coverage(core, box_polygon(m1.box_at(x), pose, K))
        except ValueError:
            return 1e6
        t = evidence_terms(pose, K, {})
        prior = float(np.sum(((ang - mid) / half) ** 2))
        return (fit_cfg.w_uncovered * unc + fit_cfg.w_excess * exc + fit_cfg.w_prior * prior + pen
                + evidence_cost(t))

    # screen many cheap starts, then Nelder-Mead from the best few: in plumb mode
    # elevation trades off against the principal-point shift, which leaves local minima
    rng = np.random.default_rng(0)
    starts = [(mid, None)]
    if free_f:
        starts += [(mid, float(flo)), (mid, float(np.sqrt(flo * fhi))), (mid, float(fhi))]
    for _ in range(fit_cfg.screen_starts):
        f = float(np.exp(rng.uniform(np.log(flo), np.log(fhi)))) if free_f else None
        starts.append((lo + rng.random(3) * (hi - lo), f))
    x0s = sorted((m1.init(*ang, core.bounds, f) for ang, f in starts), key=cost1)
    best = min((_nm(cost1, x0, m1.steps, 1.0, 1e-4) for x0 in x0s[: fit_cfg.restarts]), key=lambda r: r.fun)
    best = _polish(cost1, best, m1.steps, fit_cfg.polish_rounds)
    x1 = best.x
    pose1, K1 = m1.camera(x1)
    kx, ky = m1.aspect(x1)

    if fit_cfg.dims_include_protrusions:
        # ---- stage 2: place the SPEC box (angles fixed) around the full outline
        target = full
        m2 = Model(box, K1, profile.camera.verticals, fixed_angles=tuple(x1[:3]))

        def cost2(x):
            try:
                pose, K = m2.camera(x)
                unc, exc = coverage(target, box_polygon(box, pose, K))
            except ValueError:
                return 1e6
            return fit_cfg.w_uncovered * unc + fit_cfg.w_excess * exc + evidence_cost(evidence_terms(pose, K, {}))

        y0 = Model(box, K1, profile.camera.verticals).init(*x1[:3], target.bounds)[3:6]
        r2 = _polish(cost2, _nm(cost2, y0, m2.steps, 1.0, 1e-5), m2.steps, fit_cfg.polish_rounds)
        pose, K = m2.camera(r2.x)
        out_box = box
    else:
        # the lines are drawn off the BODY box, which hugs the product like a designer's lines;
        # labels still print the spec values verbatim
        target, pose, K, out_box = core, pose1, K1, m1.box_at(x1)
    unc, exc = coverage(target, box_polygon(out_box, pose, K))
    t = evidence_terms(pose, K, {})

    res = FitResult(pose, K, out_box, image_size, "assisted" if evidence else "auto", unc, exc, "ok",
                    reproj_rms=t.get("corner_rms_px"), verticals=profile.camera.verticals)
    res.line_rms_deg = float(np.degrees(np.arcsin(min(1.0, t["line_sin_rms"])))) if "line_sin_rms" in t else None
    res.body_aspect = (round(kx, 3), round(ky, 3))
    if unc >= fit_cfg.max_uncovered:
        res.notes.append(f"uncovered {unc:.3f} >= max {fit_cfg.max_uncovered}")
    elo, ehi = profile.expected_excess
    if not (elo <= exc <= ehi):
        res.notes.append(f"excess {exc:.3f} outside expected [{elo}, {ehi}]")
    if res.line_rms_deg is not None and res.line_rms_deg > fit_cfg.max_line_deg:
        res.notes.append(f"line evidence off by {res.line_rms_deg:.2f} deg rms > {fit_cfg.max_line_deg}")
    if res.reproj_rms is not None and res.reproj_rms > fit_cfg.max_corner_px * scale:
        res.notes.append(f"corner reprojection {res.reproj_rms:.1f}px rms; re-click corners")
    if res.notes:
        res.status = "needs_manual"
    ang = x1[:3]
    if np.any(np.abs(ang - lo) < 0.25) or np.any(np.abs(ang - hi) < 0.25):
        res.notes.append("pose hit a fit_bounds limit; check nominal_pose")
    if not evidence and profile.camera.calibrated is False:
        res.notes.append("focal uncalibrated and no line evidence: receding-edge angles are a guess")
    return res


# ------------------------------------------------------------------ automatic line evidence

FACE_AXIS = {"front": "front_face", "back": "front_face", "left": "side_face", "right": "side_face"}


def detect_lines(sil: Silhouette, res: FitResult, axis_map, fit_cfg) -> dict[str, list]:
    """Long straight product edges, assigned to L or W.

    A segment only counts if it lies on a visible VERTICAL face of the current box
    (front face -> its horizontal axis, end face -> the other). The lid top is
    excluded: it carries both directions, and molded lid details are rarely square.
    Accepted only within auto_line_max_deg of the direction the current fit predicts.
    """
    import cv2
    from shapely.geometry import LineString

    W, H = res.image_size
    diag = float(np.hypot(W, H))
    raw = cv2.HoughLinesP((sil.edges * 255).astype(np.uint8), 1, np.pi / 720, threshold=80,
                          minLineLength=int(fit_cfg.auto_line_min_len * diag), maxLineGap=6)
    if raw is None:
        return {}
    segs = raw.reshape(-1, 4).astype(float)
    uv = res.box.project(res.pose, res.K)
    faces = {}
    for f in res.box.visible_faces(res.pose):
        if f in FACE_AXIS:
            from .box import FACES
            faces[f] = MultiPoint([tuple(uv[v]) for v in FACES[f][1]]).convex_hull.buffer(0.01 * diag)
    KR = res.K.K @ res.pose.R
    lines: dict[str, list] = {}
    for x0, y0, x1, y1 in segs:
        seg = LineString([(x0, y0), (x1, y1)])
        face = next((f for f, poly in faces.items() if poly.contains(seg)), None)
        if face is None:
            continue
        axis = getattr(axis_map, FACE_AXIS[face])
        m = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        v = KR @ axis_vector(axis, axis_map)
        pd = v[:2] - m * v[2]
        d = np.array([x1 - x0, y1 - y0])
        sin = abs(d[0] * pd[1] - d[1] * pd[0]) / (np.linalg.norm(d) * np.linalg.norm(pd))
        if np.degrees(np.arcsin(min(1.0, sin))) < fit_cfg.auto_line_max_deg:
            lines.setdefault(axis, []).append(((x0, y0), (x1, y1)))
    return lines


def fit_with_auto_lines(sil: Silhouette, box: Box, K0: Intrinsics, profile, fit_cfg, image_size,
                        evidence: Evidence | None = None) -> FitResult:
    """fit(), then refit with detected product edges added as line evidence."""
    res = fit(sil, box, K0, profile, fit_cfg, image_size, evidence)
    if not fit_cfg.auto_lines:
        return res
    found = detect_lines(sil, res, profile.axis_map, fit_cfg)
    n = sum(len(v) for v in found.values())
    if n == 0:
        res.notes.append("auto lines: none found")
        return res
    merged = Evidence({k: list(v) for k, v in (evidence.lines if evidence else {}).items()},
                      dict(evidence.corners) if evidence else {})
    for k, v in found.items():
        merged.lines.setdefault(k, []).extend(v)
    res2 = fit(sil, box, K0, profile, fit_cfg, image_size, merged)
    # trim: drop detected lines the refit disagrees with (molded details that aren't square), refit once
    keep: dict[str, list] = {}
    for axis, segs in found.items():
        for seg in segs:
            r = abs(line_residuals(res2.pose, res2.K, {axis: [seg]}, profile.axis_map)[0])
            if np.degrees(np.arcsin(min(1.0, r))) <= fit_cfg.max_line_deg * 0.5:
                keep.setdefault(axis, []).append(seg)
    if keep != found and sum(len(v) for v in keep.values()):
        found = keep
        merged = Evidence({k: list(v) for k, v in (evidence.lines if evidence else {}).items()},
                          dict(evidence.corners) if evidence else {})
        for k, v in found.items():
            merged.lines.setdefault(k, []).extend(v)
        res2 = fit(sil, box, K0, profile, fit_cfg, image_size, merged)
    counts = ", ".join(f"{k}:{len(v)}" for k, v in sorted(found.items()))
    if res2.status == "ok" and res2.excess <= max(res.excess * 1.25, res.excess + 0.01):
        res2.method = "assisted+auto" if evidence else "auto+lines"
        res2.notes = [n for n in res2.notes if "focal uncalibrated" not in n]
        res2.notes.append(f"auto lines used ({counts})")
        return res2
    res.notes.append(f"auto lines rejected ({counts}): refit excess {res2.excess:.3f}, status {res2.status}")
    return res
