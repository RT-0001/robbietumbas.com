"""Studio photo ingest: masked, pre-keyed TIFFs with transparency.

Handles what Photoshop actually writes: 8/16-bit, straight (unassociated) or
premultiplied (associated) alpha, a saved mask channel (unspecified extra
sample), CMYK, and embedded ICC profiles (converted to sRGB for web output).
The original file stays untouched; it is what the PSD export will place.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

PHOTO_EXTS = (".tif", ".tiff", ".png")

# TIFF ExtraSamples values
UNSPECIFIED, ASSOCIATED, UNASSOCIATED = 0, 1, 2


class NoTransparencyError(ValueError):
    pass


@dataclass
class PhotoInfo:
    path: str
    size: tuple[int, int]
    bits: int
    color: str  # RGB | CMYK | GRAY
    alpha: str  # straight | premultiplied | mask_channel | png | none
    icc: str | None
    notes: list[str] = field(default_factory=list)


def find_photo(root: Path, sku: str) -> Path:
    for ext in PHOTO_EXTS:
        p = root / "photos" / f"{sku}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"no photo for {sku} in {root / 'photos'} (looked for {', '.join(PHOTO_EXTS)})")


def _icc_desc(icc: bytes | None) -> str | None:
    if not icc:
        return None
    try:
        return ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc))).strip()
    except Exception:
        return "unreadable ICC"


def _to_srgb(img: Image.Image, icc: bytes | None, notes: list) -> Image.Image:
    """img is RGB or CMYK (8-bit); returns sRGB RGB."""
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            dst = ImageCms.createProfile("sRGB")
            return ImageCms.profileToProfile(img, src, dst, outputMode="RGB",
                                             renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC)
        except Exception as e:  # noqa: BLE001
            notes.append(f"ICC conversion failed ({e}); colors unconverted")
    elif img.mode == "CMYK":
        notes.append("CMYK without ICC profile: naive conversion, colors will be off")
    return img.convert("RGB")


def _load_tiff(path: Path) -> tuple[Image.Image, PhotoInfo]:
    import tifffile

    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]  # Photoshop writes the flattened composite first
        arr = page.asarray()
        tags = page.tags
        icc = tags["InterColorProfile"].value if "InterColorProfile" in tags else None
        extra = tags["ExtraSamples"].value if "ExtraSamples" in tags else ()
        extra = (extra,) if isinstance(extra, int) else tuple(extra)
        photometric = page.photometric.name
        bits = int(np.dtype(page.dtype).itemsize * 8)
    icc = bytes(icc) if icc is not None else None

    if page.planarconfig.name == "SEPARATE" and arr.ndim == 3 and arr.shape[0] < arr.shape[-1]:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = arr[..., None]
    # normalize to float 0..1
    if arr.dtype.kind == "f":
        x = np.clip(arr.astype(np.float32), 0, 1)
    else:
        x = arr.astype(np.float32) / np.iinfo(arr.dtype).max

    n_color = {"RGB": 3, "SEPARATED": 4, "MINISBLACK": 1, "MINISWHITE": 1}.get(photometric)
    if n_color is None:
        raise ValueError(f"{path.name}: unsupported TIFF photometric {photometric}")
    color, rest = x[..., :n_color], x[..., n_color:]
    if photometric == "MINISWHITE":
        color = 1 - color
    notes: list[str] = []

    if rest.shape[-1] == 0:
        raise NoTransparencyError(
            f"{path.name}: no alpha. Expected a pre-keyed TIFF with transparency "
            "(Photoshop: Save As TIFF with 'Save Transparency' / Layers checked, or keep the mask as an alpha channel)")
    kind = extra[0] if extra else UNSPECIFIED
    a = rest[..., 0]
    if rest.shape[-1] > 1:
        notes.append(f"{rest.shape[-1]} extra channels; using the first as transparency")
    if kind == ASSOCIATED:
        alpha_kind = "premultiplied"
        color = np.where(a[..., None] > 1e-6, color / np.maximum(a[..., None], 1e-6), 0)
        color = np.clip(color, 0, 1)
    elif kind == UNASSOCIATED:
        alpha_kind = "straight"
    else:
        alpha_kind = "mask_channel"
        notes.append("alpha is an unspecified extra channel (saved mask); treated as transparency")

    c8 = np.round(color * 255).astype(np.uint8)
    a8 = np.round(a * 255).astype(np.uint8)
    mode = {1: "L", 3: "RGB", 4: "CMYK"}[n_color]
    base = Image.fromarray(c8[..., 0] if n_color == 1 else c8, mode)
    rgb = _to_srgb(base, icc, notes) if mode != "L" else base.convert("RGB")
    rgba = rgb.convert("RGBA")
    rgba.putalpha(Image.fromarray(a8, "L"))
    if a8.min() > 250:
        raise NoTransparencyError(f"{path.name}: alpha channel is fully opaque; the photo is not keyed")
    info = PhotoInfo(str(path), rgba.size, bits, {"SEPARATED": "CMYK", "RGB": "RGB"}.get(photometric, "GRAY"),
                     alpha_kind, _icc_desc(icc), notes)
    return rgba, info


def load_photo(path: Path, require_alpha: bool = True) -> tuple[Image.Image, PhotoInfo]:
    """-> (8-bit sRGB RGBA image, info)."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            return _load_tiff(path)
        except NoTransparencyError:
            if require_alpha:
                raise
            img = Image.open(path)
            return img.convert("RGBA"), PhotoInfo(str(path), img.size, 8, img.mode, "none", None,
                                                  ["no transparency; silhouette from white threshold"])
    img = Image.open(path)
    icc = img.info.get("icc_profile")
    notes: list[str] = []
    alpha = img.getchannel("A") if "A" in img.getbands() else None
    if alpha is None and require_alpha:
        raise NoTransparencyError(f"{path.name}: no alpha channel")
    rgb = _to_srgb(img.convert("CMYK" if img.mode == "CMYK" else "RGB"), icc, notes) if icc else img.convert("RGB")
    rgba = rgb.convert("RGBA")
    if alpha is not None:
        rgba.putalpha(alpha)
    return rgba, PhotoInfo(str(path), rgba.size, 8, "RGB", "png" if alpha else "none", _icc_desc(icc), notes)
