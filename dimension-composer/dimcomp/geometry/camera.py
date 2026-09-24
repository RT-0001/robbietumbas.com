"""Pinhole camera: intrinsics, pose, projection, backface test.

Conventions
-----------
World (product) frame, inches: origin at bottom-center of the bounding box,
x along the front face, y into the scene (front face at -y), z up.
Camera frame (OpenCV): x right, y down, z forward.
Pose maps world -> camera: X_c = R @ X_w + t.

Parametric pose (yaw, pitch, roll, distance, tx, ty) orbits the camera around
a target point: yaw=0 looks square at the front face, yaw<0 moves the camera
toward -x (left end visible), pitch>0 raises the camera. tx/ty shift the
object in the camera plane (inches), so the optical axis need not hit the
target.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class Intrinsics:
    f: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.0]])

    @classmethod
    def from_profile(cls, cam, width: int, height: int) -> "Intrinsics":
        f = cam.focal_px_at_2000 * max(width, height) / 2000.0
        if cam.principal_point == "center":
            cx, cy = width / 2.0, height / 2.0
        else:
            cx, cy = cam.principal_point[0] * width, cam.principal_point[1] * height
        return cls(f, cx, cy)

    def scaled(self, s: float) -> "Intrinsics":
        return Intrinsics(self.f * s, self.cx * s, self.cy * s)


@dataclass(frozen=True)
class Pose:
    R: np.ndarray  # 3x3 world->camera
    t: np.ndarray  # (3,)

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    def to_camera(self, pts: np.ndarray) -> np.ndarray:
        return np.asarray(pts, float) @ self.R.T + self.t


def rotation_from_angles(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])
    # unit vector from target to camera
    u = np.array([np.sin(yaw) * np.cos(pitch), -np.cos(yaw) * np.cos(pitch), np.sin(pitch)])
    fwd = -u
    right = np.cross(fwd, UP)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    c, s = np.cos(roll), np.sin(roll)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return Rz @ R


def pose_from_params(yaw_deg: float, pitch_deg: float, roll_deg: float, distance: float,
                     tx: float = 0.0, ty: float = 0.0, target=(0.0, 0.0, 0.0)) -> Pose:
    R = rotation_from_angles(yaw_deg, pitch_deg, roll_deg)
    C = np.asarray(target, float) - distance * R[2]
    t = -R @ C + np.array([tx, ty, 0.0])
    return Pose(R, t)


def camera_from_params(yaw_deg: float, elev_deg: float, roll_deg: float, distance: float,
                       a: float, b: float, K0: Intrinsics, target, verticals: str) -> tuple[Pose, Intrinsics]:
    """Shared 6-DOF parameterization for fitting.

    converge: camera pitched down at the target (pinhole, verticals converge);
              a, b translate the object in the camera plane (inches).
    plumb:    optical axis horizontal (shift lens / Upright-corrected verticals);
              the camera sits at elevation elev_deg and the principal point is
              shifted by (a, b) inches-at-target-depth.
    """
    if verticals == "converge":
        return pose_from_params(yaw_deg, elev_deg, roll_deg, distance, a, b, target), K0
    yaw, elev = np.radians([yaw_deg, elev_deg])
    u = np.array([np.sin(yaw) * np.cos(elev), -np.cos(yaw) * np.cos(elev), np.sin(elev)])
    C = np.asarray(target, float) + distance * u
    R = rotation_from_angles(yaw_deg, 0.0, roll_deg)
    depth = distance * np.cos(elev)
    K = Intrinsics(K0.f, K0.cx + a * K0.f / depth, K0.cy + b * K0.f / depth)
    return Pose(R, -R @ C), K


def elevation_deg(pose: Pose, target) -> float:
    """Camera elevation above the target, independent of where the optical axis points."""
    v = pose.center - np.asarray(target, float)
    return float(np.degrees(np.arcsin(v[2] / np.linalg.norm(v))))


def angles_from_rotation(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of rotation_from_angles (yaw, pitch, roll in degrees)."""
    fwd = R[2]
    u = -fwd
    pitch = np.degrees(np.arcsin(np.clip(u[2], -1, 1)))
    yaw = np.degrees(np.arctan2(u[0], -u[1]))
    R0 = rotation_from_angles(yaw, pitch, 0.0)
    Rz = R @ R0.T
    roll = np.degrees(np.arctan2(Rz[1, 0], Rz[0, 0]))
    return float(yaw), float(pitch), float(roll)


def project(pts: np.ndarray, pose: Pose, K: Intrinsics) -> np.ndarray:
    """(N,3) world -> (N,2) pixels."""
    Xc = pose.to_camera(np.atleast_2d(pts))
    z = Xc[:, 2:3]
    if np.any(z <= 1e-9):
        raise ValueError("point behind camera")
    return np.hstack([K.f * Xc[:, 0:1] / z + K.cx, K.f * Xc[:, 1:2] / z + K.cy])


def unproject(uv: np.ndarray, depth: np.ndarray, pose: Pose, K: Intrinsics) -> np.ndarray:
    """(N,2) pixels + camera-z depth -> (N,3) world."""
    uv = np.atleast_2d(uv)
    depth = np.asarray(depth, float).reshape(-1, 1)
    Xc = np.hstack([(uv[:, 0:1] - K.cx) / K.f * depth, (uv[:, 1:2] - K.cy) / K.f * depth, depth])
    return (Xc - pose.t) @ pose.R


def local_scale(pt: np.ndarray, pose: Pose, K: Intrinsics) -> float:
    """Pixels per inch at a world point (for a small fronto-parallel segment)."""
    return K.f / float(pose.to_camera(np.atleast_2d(pt))[0, 2])


def faces_camera(face_center: np.ndarray, normal: np.ndarray, pose: Pose) -> bool:
    return float(np.dot(normal, pose.center - face_center)) > 0
