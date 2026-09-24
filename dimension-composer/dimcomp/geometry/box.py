"""Named bounding box: 8 vertices, 12 edges, 6 faces.

Vertex names: {bottom|top}_{front|back}_{left|right}; front = -y, left = -x.
Edge names: bottom_front, top_left, ... for horizontal edges (level_face), and
front_left, back_right, ... for vertical edges.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .camera import Intrinsics, Pose, faces_camera, project

LEVEL = {"bottom": 0.0, "top": 1.0}
DEPTH = {"front": -0.5, "back": 0.5}
SIDE = {"left": -0.5, "right": 0.5}

FACES = {
    # name: (normal, vertex names in loop order)
    "front": ((0, -1, 0), ["bottom_front_left", "bottom_front_right", "top_front_right", "top_front_left"]),
    "back": ((0, 1, 0), ["bottom_back_right", "bottom_back_left", "top_back_left", "top_back_right"]),
    "left": ((-1, 0, 0), ["bottom_back_left", "bottom_front_left", "top_front_left", "top_back_left"]),
    "right": ((1, 0, 0), ["bottom_front_right", "bottom_back_right", "top_back_right", "top_front_right"]),
    "top": ((0, 0, 1), ["top_front_left", "top_front_right", "top_back_right", "top_back_left"]),
    "bottom": ((0, 0, -1), ["bottom_front_left", "bottom_back_left", "bottom_back_right", "bottom_front_right"]),
}

# profile vocabulary -> face whose bottom edge it names
BOTTOM_EDGE_FACE = {"front": "front", "back": "back", "left_side": "left", "right_side": "right"}


def _edges() -> dict[str, tuple[str, str]]:
    e = {}
    for lvl in LEVEL:
        for d in DEPTH:
            e[f"{lvl}_{d}"] = (f"{lvl}_{d}_left", f"{lvl}_{d}_right")
        for s in SIDE:
            e[f"{lvl}_{s}"] = (f"{lvl}_front_{s}", f"{lvl}_back_{s}")
    for d in DEPTH:
        for s in SIDE:
            e[f"{d}_{s}"] = (f"bottom_{d}_{s}", f"top_{d}_{s}")
    return e


EDGES = _edges()


@dataclass(frozen=True)
class Box:
    """Extents along world x (sx), y (sy), z (sz), inches."""
    sx: float
    sy: float
    sz: float

    @classmethod
    def from_spec(cls, dims, axis_map) -> "Box":
        d = {"L": dims.L, "W": dims.W, "H": dims.H}
        return cls(d[axis_map.front_face], d[axis_map.side_face], d["H"])

    @cached_property
    def vertices(self) -> dict[str, np.ndarray]:
        v = {}
        for lvl, z in LEVEL.items():
            for d, y in DEPTH.items():
                for s, x in SIDE.items():
                    v[f"{lvl}_{d}_{s}"] = np.array([x * self.sx, y * self.sy, z * self.sz])
        return v

    @property
    def names(self) -> list[str]:
        return list(self.vertices)

    @property
    def array(self) -> np.ndarray:
        return np.stack(list(self.vertices.values()))

    @property
    def center(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.sz / 2])

    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm([self.sx, self.sy, self.sz]))

    def edge(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        a, b = EDGES[name]
        return self.vertices[a], self.vertices[b]

    def face_center(self, face: str) -> np.ndarray:
        return np.mean([self.vertices[n] for n in FACES[face][1]], axis=0)

    def face_normal(self, face: str) -> np.ndarray:
        return np.array(FACES[face][0], float)

    def visible_faces(self, pose: Pose) -> list[str]:
        return [f for f in FACES if faces_camera(self.face_center(f), self.face_normal(f), pose)]

    def visible_edges(self, pose: Pose) -> set[str]:
        vis = set(self.visible_faces(pose))
        out = set()
        for name, (a, b) in EDGES.items():
            for f in vis:
                vs = FACES[f][1]
                if a in vs and b in vs:
                    out.add(name)
        return out

    def project(self, pose: Pose, K: Intrinsics) -> dict[str, np.ndarray]:
        uv = project(self.array, pose, K)
        return dict(zip(self.names, uv))
