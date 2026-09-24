# dimcomp: dimension image composer

Composes Amazon-style dimension infographics from **real product photos**. It adds the title, perspective-correct dimension lines, labels and callouts, and never alters product pixels. Built from `dimension-composer-plan.md`; milestones **M0–M3** are done.

```
pip install -e '.[dev]'
dimcomp config specs/00034966.json                 # resolved config (style <- profile <- sku <- --set)
dimcomp fit    specs/00034966.json                 # pose fit -> fits/<sku>.json + out/<sku>/debug_fit.png
dimcomp run    specs/00034966.json                 # top-3 candidates (.svg/.png/.scene.json) + report.json
dimcomp run    specs/00034966.json --set search.pinned.height_side=left --set annotations.ticks=arrows
dimcomp batch  specs/
pytest -q
```

On the fixture, search takes about 4.5s for about 980 layout evaluations. Rendering takes about 4s. A fresh fit takes about 10s, and the result is cached in `fits/`.

## Pipeline

`spec + photo → silhouette → box fit (pose) → 3D annotations → LHS search + refine → scene.json → SVG/PNG`

| Module | Role |
|---|---|
| `config/` | pydantic models, deep-merge precedence, `--set a.b=c` session patches; a bad patch fails naming its layer |
| `geometry/camera.py` | pinhole camera, `plumb` / `converge` models, project/unproject, backface test |
| `geometry/box.py` | named vertices, edges and faces |
| `geometry/silhouette.py` | alpha or white threshold, hull, side-clutter score |
| `geometry/fit.py` | one objective: containment of the silhouette + optional line/corner evidence |
| `layout/annotations.py` | 3D dims → 2D, with offsets solved *visually* |
| `layout/scene.py` | params → canvas geometry + scene graph (single source of truth) |
| `layout/cost.py`, `search.py` | hard/soft terms, each reported raw and weighted; pins via `search.pinned` |
| `render/` | SVG, PNG (resvg, project fonts only), fit debug overlay |

## Where this deviates from the plan, and why

Everything below was measured on the approved Trailmate 25 image (`tests/fixtures/reference_00034966.jpg`). The fixture photo `photos/00034966.png` is the product cut out of that image.

1. **Verticals are plumb, so the camera model needed a `plumb` mode.** On the approved image, the silhouette's left edge sits at x=230 across 160px of height, and the front-left seam at x=423 drifts only 1px over 240px. A pinhole camera pitched down 18° would lean them about 19px. The shot was made with a shift lens, is a render, or had verticals corrected in post. In `camera.verticals: plumb`, the optical axis is horizontal, `pitch_deg` means camera elevation, and the principal point is fitted. Switching to plumb dropped the fit's excess from 0.105 to 0.076.
2. **Silhouette alone cannot fix focal length.** Across focal 5k–40k px, the best containment excess stays between 0.076 and 0.086. Over the same range, the receding (W) edge slope swings from 1.17 to 0.58, which is the difference between a right-looking and a wrong-looking W line. `test_silhouette_alone_cannot_pin_focal` pins this down.
3. **Manual mode uses lines, not just corners.** Bounding-box corners float in air on a rounded, lid-overhung cooler, so there is nothing to click. Straight features along an axis do exist: a gasket band, a lid edge, a seam. `fits/<sku>_evidence.json` takes `lines: {L|W|H: [[p,q],...]}` (plus optional `corners`). These feed the same objective as the auto fit, and focal becomes free. Synthetic test: from a deliberately wrong 5000px guess, focal is recovered within 5% and yaw within 1°. The fixture uses one W line (left gasket band, 0.7px residual over 168px) and one H line (the plumb seam).
4. **Offsets are visual, not 3D.** A ground-plane offset toward the camera is foreshortened by about sin(elevation) ≈ 0.23. With a shared 3D offset, the H line sat about 4× farther out than L and W, and the labels touched the product. `offset_ratio` is now the on-screen distance divided by the projected box diagonal. Each L/W edge solves for the 3D offset that lands there, so lines still converge correctly but look equally spaced (`offset_equality` ≈ 1e-5). The approved image measures about 0.06.
5. **H defaults to `screen_vertical`.** The approved image draws H plumb, spanning the projected box end on that side (y 209→794), not along a 3D edge. `annotations.height_mode: edge` gives the plan's version.
6. **Style defaults were measured, not guessed.** The plan's font ratios (title 0.042, label 0.030) and reserved zones match the approved image. The plan's implied fill did not: the approved group fills **0.85** of the safe area at group scale **0.94**. At 0.62, the product came out visibly small. Stroke is 0.0016 (measured), not 0.0012.

## Fixture fit: an open question

The assisted fit gives yaw -39°, focal ≈ 8.4k px, elevation 13°. Its box corner lands where the designer broke the W and L lines. But the face-width ratio on the photo suggests yaw ≈ -24°. One cropped composite can't settle this. A calibration shot per profile (plan §13) or two clean lines per axis would. The profile says `calibrated: false` until then.

## Open decisions (plan §13) and current stand-ins

- **Fonts:** Montserrat Medium/Bold/Regular and Oswald SemiBold (OFL, in `fonts/`) stand in for the brand faces. Swap `tokens.fonts.*.family` and drop the brand TTFs in `fonts/`.
- **Label format:** `{value:g}”`, taken from spec verbatim.
- **Callouts:** can count and interior dims, included only when present in the spec.
- **Still needed:** a calibration shot per profile, the number of angle profiles, whether outer dims include handles, canvas sizes beyond 2000², and an approved-image library (for M6).

## Next

M4 tuning UI (clicking evidence lines replaces clicking corners) → M5 PSD with native text layers → M6 style learning (`targets.*` and `ranges.*` above are hand-measured from one image; M6 automates that) → M7 batch contact sheets and the optional VLM tiebreaker.
