"""Studio photos arrive masked and pre-keyed as TIFF with transparency."""
import numpy as np
import pytest
import tifffile
from PIL import ImageCms

from dimcomp.geometry.photo import NoTransparencyError, load_photo

H, W = 40, 60


def product():
    rgb = np.zeros((H, W, 3), np.float64)
    rgb[...] = (0.2, 0.4, 0.6)
    a = np.zeros((H, W))
    a[10:30, 15:45] = 1.0
    a[10:30, 14] = 0.5  # soft edge
    return rgb, a


def write(path, arr, **kw):
    tifffile.imwrite(path, arr, compression="zlib", **kw)
    return path


def test_16bit_straight_alpha(tmp_path):
    rgb, a = product()
    arr = (np.dstack([rgb, a]) * 65535).round().astype(np.uint16)
    img, info = load_photo(write(tmp_path / "p.tif", arr, photometric="rgb", extrasamples=["unassalpha"]))
    px = np.asarray(img)
    assert info.bits == 16 and info.alpha == "straight"
    assert tuple(px[20, 30]) == (51, 102, 153, 255)
    assert px[0, 0, 3] == 0 and px[20, 14, 3] == 128


def test_premultiplied_alpha_is_unpremultiplied(tmp_path):
    rgb, a = product()
    arr = (np.dstack([rgb * a[..., None], a]) * 255).round().astype(np.uint8)
    img, info = load_photo(write(tmp_path / "p.tif", arr, photometric="rgb", extrasamples=["assocalpha"]))
    px = np.asarray(img).astype(int)
    assert info.alpha == "premultiplied"
    assert np.abs(px[20, 14, :3] - [51, 102, 153]).max() <= 2  # edge pixel color restored


def test_saved_mask_channel_counts_as_transparency(tmp_path):
    rgb, a = product()
    arr = (np.dstack([rgb, a]) * 255).round().astype(np.uint8)
    img, info = load_photo(write(tmp_path / "p.tif", arr, photometric="rgb", extrasamples=["unspecified"]))
    assert info.alpha == "mask_channel" and info.notes
    assert np.asarray(img)[0, 0, 3] == 0


def test_cmyk_converts_and_keeps_alpha(tmp_path):
    _, a = product()
    cmyk = np.zeros((H, W, 4))
    cmyk[..., 0] = 1.0  # pure cyan
    arr = (np.dstack([cmyk, a]) * 255).round().astype(np.uint8)
    img, info = load_photo(write(tmp_path / "p.tif", arr, photometric="separated", extrasamples=["unassalpha"]))
    r, g, b, al = np.asarray(img)[20, 30]
    assert info.color == "CMYK" and al == 255
    assert r < 60 and g > 150 and b > 150
    assert any("CMYK without ICC" in n for n in info.notes)


def test_icc_is_read(tmp_path):
    rgb, a = product()
    arr = (np.dstack([rgb, a]) * 255).round().astype(np.uint8)
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    _, info = load_photo(write(tmp_path / "p.tif", arr, photometric="rgb", extrasamples=["unassalpha"], iccprofile=icc))
    assert info.icc and "sRGB" in info.icc


def test_unkeyed_tiff_fails_loudly(tmp_path):
    rgb, _ = product()
    p = write(tmp_path / "flat.tif", (rgb * 255).astype(np.uint8), photometric="rgb")
    with pytest.raises(NoTransparencyError, match="Save Transparency"):
        load_photo(p)
    rgb, _ = product()
    opaque = (np.dstack([rgb, np.ones((H, W))]) * 255).astype(np.uint8)
    with pytest.raises(NoTransparencyError, match="not keyed"):
        load_photo(write(tmp_path / "opaque.tif", opaque, photometric="rgb", extrasamples=["unassalpha"]))
