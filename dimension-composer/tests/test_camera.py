import cv2
import numpy as np
import pytest

from dimcomp.geometry.box import Box
from dimcomp.geometry.camera import (Intrinsics, angles_from_rotation, camera_from_params, pose_from_params,
                                     project, unproject)

K = Intrinsics(4200.0, 1000.0, 1000.0)
PTS = np.random.default_rng(1).uniform(-10, 10, (50, 3)) + [0, 0, 8]


@pytest.mark.parametrize("yaw,pitch,roll", [(-35, 18, 0), (20, 5, 2), (0, 30, -3)])
def test_round_trip(yaw, pitch, roll):
    pose = pose_from_params(yaw, pitch, roll, 80.0, 1.5, -2.0, (0, 0, 8))
    uv = project(PTS, pose, K)
    depth = pose.to_camera(PTS)[:, 2]
    np.testing.assert_allclose(unproject(uv, depth, pose, K), PTS, atol=1e-9)
    assert np.allclose(angles_from_rotation(pose.R), (yaw, pitch, roll), atol=1e-9)


def test_matches_opencv():
    pose = pose_from_params(-30, 15, 1, 70.0, 0.5, 0.2, (0, 0, 8))
    rvec, _ = cv2.Rodrigues(pose.R)
    ref, _ = cv2.projectPoints(PTS, rvec, pose.t, K.K, None)
    np.testing.assert_allclose(project(PTS, pose, K), ref[:, 0, :], atol=1e-6)


def test_plumb_keeps_verticals_vertical_and_shows_top():
    box = Box(20, 14, 16)
    pose, K2 = camera_from_params(-35, 18, 0, 90, 0, 0, K, box.center, "plumb")
    uv = box.project(pose, K2)
    for e in ("front_left", "front_right", "back_left", "back_right"):
        a, b = uv[f"bottom_{e.split('_')[0]}_{e.split('_')[1]}"], uv[f"top_{e.split('_')[0]}_{e.split('_')[1]}"]
        assert abs(a[0] - b[0]) < 1e-6
    assert "top" in box.visible_faces(pose)
    pose_c, _ = camera_from_params(-35, 18, 0, 90, 0, 0, K, box.center, "converge")
    a, b = box.project(pose_c, K)["bottom_front_right"], box.project(pose_c, K)["top_front_right"]
    assert abs(a[0] - b[0]) > 1.0  # converging verticals lean


def test_backface():
    box = Box(20, 14, 16)
    pose = pose_from_params(-35, 18, 0, 80, target=box.center)
    assert set(box.visible_faces(pose)) == {"front", "left", "top"}
    pose = pose_from_params(35, 18, 0, 80, target=box.center)
    assert set(box.visible_faces(pose)) == {"front", "right", "top"}
