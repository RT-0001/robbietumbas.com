import numpy as np
import pytest

from dimcomp.config.models import FitCfg, Profile
from dimcomp.geometry import silhouette
from dimcomp.geometry.box import Box
from dimcomp.geometry.camera import Intrinsics, angles_from_rotation, camera_from_params, elevation_deg
from dimcomp.geometry.fit import Evidence, fit
from synth import render

BOX = Box(20.55, 14.23, 16.7)
SIZE = (2000, 1600)


def profile(verticals, nominal=(-30, 16)):
    return Profile.model_validate({
        "name": "t", "camera": {"focal_px_at_2000": 6000, "verticals": verticals},
        "nominal_pose": {"yaw_deg": nominal[0], "pitch_deg": nominal[1]},
        "fit_bounds": {"yaw_deg": [-10, 10], "pitch_deg": [-8, 8], "roll_deg": [-3, 3]},
        "visible_bottom_edges": ["front", "left_side"], "expected_excess": [0.0, 0.4]})


def truth(verticals, yaw, elev, f=6000.0):
    K0 = Intrinsics(f, SIZE[0] / 2, SIZE[1] / 2)
    d = 95.0 * f / 6000.0
    # plumb: shift the principal point so the product is framed (a real shift lens / crop)
    b = -d * np.sin(np.radians(elev)) - 1.0 if verticals == "plumb" else -1.0
    return camera_from_params(yaw, elev, 0.0, d, 0.5, b, K0, BOX.center, verticals)


@pytest.mark.parametrize("verticals,yaw,elev", [("plumb", -34, 14), ("converge", -27, 19)])
def test_auto_fit_recovers_pose(verticals, yaw, elev):
    pose, K = truth(verticals, yaw, elev)
    sil = silhouette.extract(render(BOX, pose, K, SIZE), FitCfg())
    K0 = Intrinsics(6000.0, SIZE[0] / 2, SIZE[1] / 2)
    res = fit(sil, BOX, K0, profile(verticals), FitCfg(), SIZE)
    y, _, r = angles_from_rotation(res.pose.R)
    assert res.status == "ok", res.qa()
    assert abs(y - yaw) < 1.0
    assert abs(elevation_deg(res.pose, BOX.center) - elev) < 1.0
    assert abs(r) < 1.0
    d_true = np.linalg.norm(pose.center - BOX.center)
    d_fit = np.linalg.norm(res.pose.center - BOX.center)
    assert abs(d_fit / d_true - 1) < 0.01  # 1% scale


def test_line_evidence_recovers_focal():
    """Wrong focal in the profile; two W lines + one L line pin the true one."""
    pose, K = truth("plumb", -36, 12, f=11000.0)
    sil = silhouette.extract(render(BOX, pose, K, SIZE), FitCfg())
    from dimcomp.geometry.camera import project
    def seg(a, b):
        uv = project(np.array([a, b]), pose, K)
        return (tuple(uv[0]), tuple(uv[1]))
    L2, W2 = BOX.sx / 2, BOX.sy / 2
    ev = Evidence(lines={
        "W": [seg((-L2, -W2, 16), (-L2, W2, 16)), seg((-L2, -W2, 3), (-L2, W2, 3))],
        "L": [seg((-L2, -W2, 14), (L2, -W2, 14))],
    })
    K0 = Intrinsics(5000.0, SIZE[0] / 2, SIZE[1] / 2)  # deliberately wrong
    res = fit(sil, BOX, K0, profile("plumb", (-32, 15)), FitCfg(), SIZE, ev)
    assert res.line_rms_deg < 0.2
    assert abs(res.K.f / 11000.0 - 1) < 0.05
    assert abs(angles_from_rotation(res.pose.R)[0] - (-36)) < 1.0


def test_silhouette_alone_cannot_pin_focal():
    """Documents the M1 finding: containment cost is nearly flat in focal."""
    pose, K = truth("plumb", -36, 12, f=11000.0)
    sil = silhouette.extract(render(BOX, pose, K, SIZE), FitCfg())
    excess = []
    for f in (5000.0, 11000.0, 30000.0):
        K0 = Intrinsics(f, SIZE[0] / 2, SIZE[1] / 2)
        excess.append(fit(sil, BOX, K0, profile("plumb", (-32, 15)), FitCfg(restarts=2), SIZE).excess)
    assert max(excess) - min(excess) < 0.05
