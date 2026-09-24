import pytest

from dimcomp.config import deep_merge, parse_set, resolve


def test_parse_set_and_merge():
    patch = parse_set(["search.samples=40", "annotations.ticks=arrows", "layout.optical_center_target=[0.5, 0.5]"])
    assert patch == {"search": {"samples": 40}, "annotations": {"ticks": "arrows"},
                     "layout": {"optical_center_target": [0.5, 0.5]}}
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}


def test_precedence_and_layers(root):
    r = resolve(root / "specs" / "00034966.json", session={"search": {"samples": 12}})
    assert r.style.search.samples == 12
    assert r.layers[0] == "styles/igloo_default.yaml" and r.layers[-1] == "session"
    assert r.profile.camera.verticals == "plumb"


def test_bad_patch_names_its_layer(root):
    with pytest.raises(ValueError, match="session"):
        resolve(root / "specs" / "00034966.json", session={"annotations": {"ticks": "squiggles"}})
