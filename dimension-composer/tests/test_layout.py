import numpy as np

from dimcomp.layout.cost import HARD, score
from dimcomp.layout.scene import Params, build_layout
from dimcomp.layout.search import search


def test_default_layout_is_clean(ctx):
    lay = build_layout(ctx, Params(label_t={"L": 0.5, "W": 0.5, "H": 0.4}))
    cost, br = score(ctx, lay)
    assert all(br[k]["raw"] == 0 for k in HARD), br
    # visual offsets: all three lines sit the same on-screen distance from the box
    offs = [d.dim.offset_px for d in lay.dims]
    assert np.ptp(offs) / np.mean(offs) < 0.02
    ids = [l["id"] for l in lay.scene["layers"]]
    assert ids[:2] == ["title", "product"] and {"dim_L", "dim_W", "dim_H"} <= set(ids)


def test_labels_are_spec_values_verbatim(ctx):
    lay = build_layout(ctx, Params())
    labels = {l["axis"]: l["label"]["text"] for l in lay.scene["layers"] if l["type"] == "dimension"}
    assert labels == {"L": "20.55”", "W": "14.23”", "H": "16.7”"}


def test_label_on_product_is_hard_violation(ctx):
    lay = build_layout(ctx, Params(offset_ratio=0.0))
    _, br = score(ctx, lay)
    assert br["hard_product_clearance"]["raw"] > 0


def test_height_goes_on_clean_side(ctx):
    assert ctx.clutter["right"] < ctx.clutter["left"]  # handle hangs on the left
    _, br = score(ctx, build_layout(ctx, Params(height_side="left")))
    assert br["height_side_clutter"]["raw"] > 0


def test_search_top3(ctx):
    ctx.cfg.style.search.samples = 60
    ctx.cfg.style.search.refine_top = 3
    top, stats = search(ctx)
    assert len(top) == 3
    assert [c.cost for c in top] == sorted(c.cost for c in top)
    assert all(sum(c.breakdown[k]["weighted"] for k in HARD) == 0 for c in top)
    assert top[0].params.height_side == "right"


def test_lines_follow_template_psd(root):
    """Yukon lines vs the designer's lines in the template PSD (4000px, halved)."""
    from dimcomp.config import resolve
    from dimcomp.pipeline import load_context
    c = load_context(resolve(root / "specs" / "yukon70.json"))
    lay = build_layout(c, Params(offset_ratio=0.06, group_scale=1.0, label_t={"L": 0.567, "W": 0.524, "H": 0.466}))
    psd = {"W": ((360.3, 3032.5), (1246.6, 3342.8)), "L": ((1614.6, 3364.6), (3441.1, 3018.6)),
           "H": ((3593, 2777.2), (3593, 1124.2))}
    for d in lay.dims:
        p = np.array(psd[d.dim.axis]) / 2
        a, b = d.p0, d.p1
        if np.linalg.norm(a - p[0]) > np.linalg.norm(b - p[0]):
            a, b = b, a
        u, v = (b - a) / np.linalg.norm(b - a), (p[1] - p[0]) / np.linalg.norm(p[1] - p[0])
        assert np.degrees(np.arccos(np.clip(u @ v, -1, 1))) < 5.0, d.dim.axis  # same direction
        assert np.linalg.norm(b - p[1]) < 60, d.dim.axis  # far/near ends land near the designer's
    ids = [l["id"] for l in lay.scene["layers"]]
    h = next(l for l in lay.scene["layers"] if l["id"] == "dim_H")
    assert "hang_x" in h["label"]  # numerals centered on the plumb line
