"""Product silhouette from the photo's alpha (studio shots arrive pre-keyed).
White-threshold fallback exists only for fit.require_alpha: false. No ML."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import MultiPoint, Polygon


@dataclass
class Silhouette:
    mask: np.ndarray  # uint8 {0,1}, working resolution
    hull: Polygon
    core_hull: Polygon  # hull after opening away thin protrusions (handles)
    contour: Polygon
    edges: np.ndarray  # uint8 {0,1} Canny edges inside the mask

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.hull.bounds

    def side_clutter(self, side: str, band: float = 0.12) -> float:
        """Edge density + silhouette protrusion in the outer band on one side.

        High where the height line would sit next to handles, latches, texture.
        """
        x0, y0, x1, y1 = self.bbox
        w = x1 - x0
        bx0, bx1 = (x0, x0 + band * w) if side == "left" else (x1 - band * w, x1)
        cols = slice(int(bx0), int(np.ceil(bx1)))
        rows = slice(int(y0), int(np.ceil(y1)))
        m = self.mask[rows, cols]
        if m.sum() == 0:
            return 0.0
        density = float(self.edges[rows, cols].sum() / m.sum())
        # ragged side = mask fills the band unevenly (protrusions)
        ragged = float(1.0 - m.mean())
        return density + 0.5 * ragged


def extract(img: Image.Image, fit_cfg) -> Silhouette:
    a = np.asarray(img.convert("RGBA"))
    alpha = a[..., 3]
    if alpha.min() < 250:
        m = (alpha >= fit_cfg.alpha_threshold).astype(np.uint8)
    else:
        gray = cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY)
        m = (gray < fit_cfg.white_threshold).astype(np.uint8)
    k = np.ones((fit_cfg.morph_px, fit_cfg.morph_px), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        raise ValueError("no product found in image (check threshold / alpha)")
    m = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(float)
    contour = Polygon(c).buffer(0)
    hull = MultiPoint([tuple(p) for p in c]).convex_hull

    gray = cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY)
    edges = (cv2.Canny(gray, 60, 160) > 0).astype(np.uint8) * cv2.erode(m, k)
    return Silhouette(mask=m, hull=hull, core_hull=_core(m, hull, fit_cfg.core_open_ratio),
                      contour=contour, edges=edges)


def _core(m: np.ndarray, hull: Polygon, ratio: float) -> Polygon:
    """Body without handles: morphological opening with a disk ~ratio x bbox diagonal."""
    x0, y0, x1, y1 = hull.bounds
    d = int(round(ratio * np.hypot(x1 - x0, y1 - y0))) | 1
    if d < 3:
        return hull
    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(opened)
    if n < 2:
        return hull
    core = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    cnts, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(float)
    return MultiPoint([tuple(p) for p in c]).convex_hull
