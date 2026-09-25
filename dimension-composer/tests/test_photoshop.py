"""The generated .jsx runs end to end against a Node mock of Photoshop's DOM.
That covers syntax, control flow, geometry and API usage; not Photoshop's rendering."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from dimcomp.layout.scene import Params, build_layout
from dimcomp.render.photoshop import LIB, to_jsx

NODE = shutil.which("node")
MOCK = Path(__file__).with_name("ps_mock.js")


def test_builder_is_es3():
    """ExtendScript is ES3: no arrow functions, let/const, template strings, JSON, forEach/map."""
    src = re.sub(r"//.*", "", LIB.read_text())
    for pat in (r"=>", r"\blet\b", r"\bconst\b", r"`", r"\bJSON\.", r"\.forEach\(", r"\.map\(", r"\.filter\(", r"\bclass\b"):
        assert not re.search(pat, src), pat


def run(ctx, tmp_path, fonts="gibson"):
    lay = build_layout(ctx, Params())
    jsx = tmp_path / "b.jsx"
    jsx.write_text(to_jsx(lay.scene, "out", ctx.cfg.root))
    r = subprocess.run([NODE, str(MOCK), str(jsx), fonts], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return lay, json.loads(r.stdout)


def flat(tree, path=""):
    for n in tree:
        if "group" in n:
            yield from flat(n["children"], path + n["group"] + "/")
        else:
            yield path + n["name"], n


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_builds_full_layer_tree(ctx, tmp_path):
    lay, out = run(ctx, tmp_path)
    assert out["alerts"][0].startswith("Built out.psd"), out["alerts"]
    assert out["saved"] and out["prefs"] == {"rulerUnits": "IN", "typeUnits": "PT"}  # user prefs restored
    layers = dict(flat(out["tree"]))
    for name in ("dimensions/dim_L/dim_L_label", "dimensions/dim_L/dim_L_line", "dimensions/dim_W/dim_W_line",
                 "dimensions/dim_H/dim_H_line", "callout_cans/can", "callout_cans/cans_num", "product"):
        assert name in layers, name
    assert layers["dimensions/dim_L/dim_L_label"]["font"] == "Gibson-SemiBold"
    assert layers["callout_interior/interior_label"]["font"] == "Gibson-Regular"
    assert layers["dimensions/dim_L/dim_L_line"]["subpaths"] == 2  # split around the label
    # product lands on the scene's alpha box
    target = next(l for l in lay.scene["layers"] if l["id"] == "product")["alpha_bbox"]
    got = layers["product"]["bounds"]
    assert abs(got[0] - target[0]) <= 1 and abs(got[1] - target[1]) <= 1 and abs(got[2] - target[2]) <= 1


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_missing_gibson_is_reported(ctx, tmp_path):
    _, out = run(ctx, tmp_path, fonts="nogibson")
    assert "Gibson semibold is not active" in out["alerts"][0]


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_can_artwork_is_embedded_byte_exact(ctx, tmp_path, root):
    import base64
    _, out = run(ctx, tmp_path)
    written = base64.b64decode(out["assetsWritten"]["dimcomp_can.png"])
    assert written == (root / "assets" / "can.png").read_bytes()
