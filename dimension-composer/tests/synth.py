"""Synthetic products with known pose: a box-shaped union of parts whose
bounding box is exactly the spec box (lid overhang, inset body, corner feet)."""
import cv2
import numpy as np
from PIL import Image

from dimcomp.geometry.box import Box
from dimcomp.geometry.camera import project


def parts(box: Box):
    L, W, H = box.sx, box.sy, box.sz
    return [
        # (x0, x1, y0, y1, z0, z1)
        (-L / 2, L / 2, -W / 2, W / 2, 0.82 * H, H),                          # lid, full footprint
        (-0.45 * L, 0.45 * L, -0.44 * W, 0.44 * W, 0.04 * H, 0.82 * H),       # inset body
        *[(sx * L / 2 - (sx > 0) * 0.12 * L, sx * L / 2 + (sx < 0) * 0.12 * L,
           sy * W / 2 - (sy > 0) * 0.12 * W, sy * W / 2 + (sy < 0) * 0.12 * W, 0, 0.04 * H)
          for sx in (-1, 1) for sy in (-1, 1)],                                 # feet at the corners
    ]


def render(box: Box, pose, K, size=(2000, 1600)) -> Image.Image:
    mask = np.zeros(size[::-1], np.uint8)
    for x0, x1, y0, y1, z0, z1 in parts(box):
        pts = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
        uv = project(pts, pose, K)
        hull = cv2.convexHull(np.round(uv).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
    rgba = np.zeros((*mask.shape, 4), np.uint8)
    rgba[..., :3] = 90
    rgba[..., 3] = mask
    return Image.fromarray(rgba, "RGBA")
