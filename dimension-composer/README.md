# dimcomp: dimension image composer

Composes Amazon-style dimension infographics from **real product photos**. It adds the title, perspective-correct dimension lines, labels and callouts, and never alters product pixels. Built from `dimension-composer-plan.md`. Milestones M0–M3 are done, plus an MVP of M5: a Photoshop script that builds the layered PSD.

## MVP workflow

1. The designer sends a transparent copy of the masked product photo (a small PNG or TIFF is fine) plus the spec: title, L/W/H, interior dims, can count.
2. `dimcomp run specs/<sku>.json` produces three ranked candidates. Each has a PNG preview and a **`<sku>_candidate_NN.jsx`**.
3. In Photoshop: **File → Scripts → Browse…** → pick the `.jsx`. It asks for the full-res TIFF (or finds it next to the script) and builds `<sku>_dimensions_NN.psd` with:
   - the product as a smart object, scaled to the layout;
   - live **Gibson** text layers, found by name in Adobe Fonts;
   - dimension lines and the can as vector shape layers, grouped and named like the scene.

   Label gaps are cut from Photoshop's own measurement of the real Gibson text, so spacing is exact even though the previews use a stand-in font.

```
pip install -e '.[dev]'
dimcomp run    specs/yukon70.json                  # candidates + .jsx + report.json in out/yukon70/
dimcomp fit    specs/yukon70.json                  # pose only -> fits/<sku>.json + out/<sku>/debug_fit.png
dimcomp config specs/yukon70.json                  # resolved config (style <- profile <- sku <- --set)
dimcomp run    specs/yukon70.json --set search.pinned.height_side=left
pytest -q                                          # 30 tests; the .jsx is executed against a Node mock of Photoshop
```

## Template (styles/igloo_default.yaml)

The template is measured on the approved **70 QT Yukon** export (2000²), which is the current template, and cross-checked against the Trailmate 25:
- **Title:** Gibson SemiBold, cap height 0.032 of the canvas, centered at y 0.049.
- **Labels:** Gibson SemiBold, cap height 0.0195. Lines are 4px black, stopping 0.8 cap-heights short of the label.
- **Placement rule** (both approved images agree): the product is centered horizontally, and the bottom of the product-and-dims group sits at y **0.842**, just above the callout row. Scale is the largest that fits the safe area.
- **Corner:** the W and L lines are extended to their shared corner, then each stops the same distance short of it (`corner_gap_ratio`), so the gap is symmetric and deliberate.
- **Can callout:** positions and shape proportions are taken from the Yukon, and the can bleeds off the bottom edge. The can count shrinks to fit the can.
- **Interior dims:** centered, "INTERIOR DIMENSIONS…" in Regular and the values in SemiBold.

**These numbers are measured off exports.** Once the template PSD is available, its layer data should replace them.

## How the fit works (and why it differs from the plan)

- **Plumb verticals.** Both approved images have perfectly vertical edges (shift lens, a render, or Upright in post). So `camera.verticals: plumb` uses a horizontal optical axis, with `pitch_deg` meaning camera elevation and a fitted principal point.
- **Body first, handles ignored.** Stage 1 fits the angles to the *body*: the outline after opening away thin protrusions such as handles and wire bails, with the box's L/W proportions free. So a handle can't tilt the box. The lines are drawn off that body box, so they hug the product the way a designer's do; labels still print the spec values verbatim. Setting `fit.dims_include_protrusions: true` makes the spec-size box swallow handles instead.
- **Automatic line evidence.** An outline alone cannot pin the focal length. Across 5k–40k px, fit quality barely changes while the W line slope swings 2×. So the fit finds long straight product edges itself (panel lines, the base, lid edges), counts only those lying on a vertical face, assigns each to L or W, drops outliers, and refits with focal free. Hand-marked lines in `fits/<sku>_evidence.json` add to that.
- **Consistent camera.** Both coolers converge on yaw about -41° and focal about 11–14k px, as expected from one studio setup.
- **Visual offsets.** A 3D ground offset toward the camera is foreshortened by about sin(elevation). So offsets are specified on screen (a fraction of the projected box diagonal), and the 3D offset is solved per edge: the lines converge correctly and still look evenly spaced.
- **Hard layout rules:** labels and lines keep clear of the product, of each other (no crossing or crowding between dimensions), of the title, of the callouts, and of the safe-area margins. A candidate that breaks any of them never outranks a clean one.

## House assumptions

- **Photos** are masked, pre-keyed TIFFs with transparency (8/16-bit, straight or premultiplied alpha, a saved mask channel, CMYK, and ICC profiles converted to sRGB). A TIFF without transparency is an error that points at Photoshop's "Save Transparency" option.
- **Type** is all Gibson, SemiBold plus Regular. Gibson is licensed through Adobe Fonts, so only Photoshop has it. The `.jsx` sets real Gibson, while the previews use Montserrat sized by cap height, and `report.json` → `font_substitutions` flags that.

## Fixtures

| SKU | Photo | Fit |
|---|---|---|
| `yukon70` | cut from the approved Yukon export | automatic (5 detected lines) |
| `00034966` Trailmate 25 | cut from the approved Trailmate export | assisted: 1 hand-marked W line + 1 H line, plus detected lines |

## Open decisions

- **Handles:** do spec outer dims include the handles? This decides `fit.dims_include_protrusions`.
- **Template PSD:** needed to replace the measured numbers with exact layer values (can shape, stroke, tracking).
- **Canvas sizes** beyond 2000², **more angle profiles**, and an **approved-image library** for style learning (M6).

## Next

M4 tuning UI (drag labels; mark lines) → M6 style learning from approved images → M7 batch contact sheets.
