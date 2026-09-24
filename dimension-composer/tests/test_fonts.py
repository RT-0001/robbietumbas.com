from dimcomp.config.models import FontTok
from dimcomp.layout.text_metrics import measure_tok


def test_gibson_missing_is_flagged_and_sized_by_cap(cfg):
    tok = FontTok(family="Gibson", weight=600, cap_ratio=0.021)
    tb = measure_tok("20.55”", tok, 2000, cfg.font_dirs, {"Gibson": "Montserrat"})
    if tb.face.family == "Gibson":  # licensed files installed: nothing to flag
        assert not tb.substituted
        return
    assert tb.substituted and tb.face.family == "Montserrat" and tb.face.weight == 600
    assert tb.postscript == "Gibson-SemiBold"  # PSD still asks for the real face
    assert abs(tb.cap - 42.0) < 1e-6  # cap height honored regardless of face


def test_h_scale_condenses(cfg):
    a = measure_tok("38", FontTok(family="Gibson", weight=600, cap_ratio=0.058), 2000, cfg.font_dirs,
                    {"Gibson": "Montserrat"})
    b = measure_tok("38", FontTok(family="Gibson", weight=600, cap_ratio=0.058, h_scale=0.65), 2000,
                    cfg.font_dirs, {"Gibson": "Montserrat"})
    assert abs(b.advance / a.advance - 0.65) < 1e-9 and a.cap == b.cap


def test_scene_records_requested_face(ctx):
    from dimcomp.layout.scene import Params, build_layout
    lay = build_layout(ctx, Params())
    fonts = [l["label"]["font"] for l in lay.scene["layers"] if l["type"] == "dimension"]
    assert all(f["family"] == "Gibson" and f["postscript"].startswith("Gibson") for f in fonts)
