"""Pydantic models for spec, angle profile, and style config.

Every aesthetic number lives in Style; code reads it, never hardcodes it.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, model_validator

Axis = Literal["L", "W", "H"]
BottomEdge = Literal["front", "back", "left_side", "right_side"]
HeightEdge = Literal["left_silhouette", "right_silhouette"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- spec

class Dims(Strict):
    L: PositiveFloat
    W: PositiveFloat
    H: PositiveFloat


class Spec(Strict):
    sku: str
    title: str
    dims_in: Dims
    interior_dims_in: Dims | None = None
    can_count: int | None = Field(default=None, ge=1)
    angle_profile: str
    photo: str | None = None  # defaults to photos/<sku>.tif|.tiff|.png
    overrides: dict = Field(default_factory=dict)


# ---------------------------------------------------------------- profile

class CameraCfg(Strict):
    focal_px_at_2000: PositiveFloat
    # "center" or (x, y) as fractions of image width/height
    principal_point: Literal["center"] | tuple[float, float] = "center"
    # plumb: verticals are parallel in the image (shift lens, render, or Upright in post);
    # pitch_deg then means camera elevation, and the principal point is fitted.
    # converge: plain pinhole pitched down at the product.
    verticals: Literal["plumb", "converge"] = "plumb"
    calibrated: bool = False
    # search window for focal when line evidence is supplied
    focal_bounds_at_2000: tuple[float, float] = (2500, 40000)


class PoseDeg(Strict):
    yaw_deg: float
    pitch_deg: float
    roll_deg: float = 0.0


class FitBounds(Strict):
    yaw_deg: tuple[float, float] = (-10, 10)
    pitch_deg: tuple[float, float] = (-8, 8)
    roll_deg: tuple[float, float] = (-3, 3)


class AxisMap(Strict):
    front_face: Literal["L", "W"] = "L"
    side_face: Literal["L", "W"] = "W"

    @model_validator(mode="after")
    def _distinct(self):
        if self.front_face == self.side_face:
            raise ValueError("axis_map.front_face and side_face must differ")
        return self


class Profile(Strict):
    name: str
    camera: CameraCfg
    nominal_pose: PoseDeg
    fit_bounds: FitBounds = FitBounds()
    axis_map: AxisMap = AxisMap()
    visible_bottom_edges: list[BottomEdge]
    height_edge_candidates: list[HeightEdge] = ["left_silhouette", "right_silhouette"]
    # fraction of the projected box NOT covered by product; outside => needs_manual
    expected_excess: tuple[float, float] = (0.0, 0.40)
    style_overrides: dict = Field(default_factory=dict)


# ---------------------------------------------------------------- style

class FontTok(Strict):
    family: str
    weight: int = 400
    # size by CAP HEIGHT / canvas width (font-independent: a stand-in font matches visually);
    # size_ratio (em / canvas width) is accepted as an alternative
    cap_ratio: PositiveFloat | None = None
    size_ratio: PositiveFloat | None = None
    tracking: float = 0.0  # em units
    h_scale: PositiveFloat = 1.0  # Photoshop horizontal scale (e.g. condensed "38")
    postscript: str | None = None  # name the PSD text layer should request; default <Family>-<Weight>

    @model_validator(mode="after")
    def _one_size(self):
        if (self.cap_ratio is None) == (self.size_ratio is None):
            raise ValueError("give exactly one of cap_ratio / size_ratio")
        return self


class Fonts(Strict):
    title: FontTok
    label: FontTok
    callout_num: FontTok  # cap_ratio is the MAX; shrinks to fit the can
    callout_word: FontTok
    callout_value: FontTok
    callout_text: FontTok
    callout_text_bold: FontTok


class Colors(Strict):
    line: str = "#1A1A1A"
    text: str = "#111111"
    callout_text: str = "#333333"
    can_fill: str = "#BBBCBE"
    can_neck: str = "#A5A6A7"
    can_tab: str = "#BDBFC2"


class Tokens(Strict):
    fonts: Fonts
    colors: Colors = Colors()
    stroke_ratio: PositiveFloat = 0.0016
    label_gap_capheights: float = 0.5
    number_format: str = "{value:g}”"


class Rect(Strict):
    x: float
    y: float
    w: PositiveFloat
    h: PositiveFloat


class Band(Strict):
    top: float
    height: PositiveFloat


class Margins(Strict):
    left: float
    right: float
    top: float
    bottom: float


class Canvas(Strict):
    w: int = 2000
    h: int = 2000
    bg: str = "#E9EAEC"


class Placement(Strict):
    """Template rule seen in every approved image: product centered, group bottom on a fixed line."""
    product_center_x: float = 0.495
    group_bottom: float = 0.842


class CansGeom(Strict):
    center_x: float = 0.0868
    head_cy: float = 0.844  # "HOLDS UP TO" cap center
    can_top: float = 0.866  # top of the pull tab
    can_w: float = 0.0785
    num_cy: float = 0.940
    word_cy: float = 0.978
    num_fill: float = 0.85  # number width <= this x can width
    bleed: bool = True  # can runs off the bottom edge


class InteriorGeom(Strict):
    cx: float = 0.5
    l1_cy: float = 0.916
    l2_cy: float = 0.946


class Layout(Strict):
    title_band: Band
    safe_area: Margins
    reserved_zones: dict[str, Rect] = Field(default_factory=dict)
    optical_center_target: tuple[float, float] = (0.5, 0.53)
    legibility_min_cap_px: float = 36
    callouts: dict[str, str] = Field(default_factory=dict)  # callout id -> reserved zone
    placement: Placement | None = None  # None: center the group in the safe area
    cans: CansGeom = CansGeom()
    interior: InteriorGeom = InteriorGeom()


class Annotations(Strict):
    clearance_ratio: float = 0.012
    ticks: Literal["none", "ticks", "arrows"] = "none"
    tick_len_ratio: float = 0.008
    extension_lines: Literal["on", "off", "auto"] = "auto"
    # "screen_vertical": H is drawn plumb, spanning the projected box end on that side
    #   (what hand-made Amazon images do). "edge": the projected 3D vertical edge.
    height_mode: Literal["screen_vertical", "edge"] = "screen_vertical"
    # W and L lines stop short of their shared virtual corner by this much each
    # (fraction of projected box diagonal). None: each line ends at its own box corner.
    corner_gap_ratio: float | None = 0.055


class FitCfg(Strict):
    require_alpha: bool = True  # studio photos arrive masked + pre-keyed; no alpha is an error
    white_threshold: int = 245
    alpha_threshold: int = 128
    morph_px: int = 5
    max_uncovered: float = 0.01
    working_long_edge: int = 2000
    restarts: int = 6  # Nelder-Mead runs, from the best screened starts
    screen_starts: int = 60
    polish_rounds: int = 4
    w_uncovered: float = 40.0
    w_excess: float = 1.0
    w_prior: float = 0.02
    w_lines: float = 3000.0  # (sin err)^2 weight; 0.1 deg ~ 1% excess (lines are precise evidence)
    w_corners: float = 0.01  # per (0.5%-of-long-edge px)^2
    max_line_deg: float = 1.0
    hull_tolerance_px: float = 1.5  # at 2000px long edge
    # handles / wire bails / thin protrusions are opened away before fitting angles
    core_open_ratio: float = 0.05  # opening kernel diameter / silhouette bbox diagonal
    body_aspect_free: bool = True
    body_aspect_bounds: tuple[float, float] = (0.6, 1.15)  # body L, W vs spec
    w_aspect_prior: float = 0.005  # a 10% proportion change costs about as much as 0.5% excess
    auto_lines: bool = True  # detect straight product edges and use them as line evidence
    auto_line_min_len: float = 0.08  # of image diagonal
    auto_line_max_deg: float = 6.0
    dims_include_protrusions: bool = False  # False: lines hug the body box; True: spec box swallows handles
    max_corner_px: float = 8.0  # at 2000px long edge


class Ranges(Strict):
    offset_ratio: tuple[float, float] = (0.035, 0.10)  # of projected box diagonal
    label_t: tuple[float, float] = (0.35, 0.65)
    height_side: list[Literal["left", "right"]] = ["left", "right"]
    group_scale: tuple[float, float] = (0.85, 0.98)
    group_nudge: tuple[float, float] = (-0.02, 0.02)
    extension_lines: list[bool] = [False, True]


class Search(Strict):
    samples: int = 300
    refine_top: int = 10
    refine_rounds: int = 4
    top_n: int = 3
    seed: int = 7
    min_param_distance: float = 0.08
    ranges: Ranges = Ranges()
    pinned: dict = Field(default_factory=dict)  # param -> fixed value


class Weights(Strict):
    offset_equality: float = 3.0
    offset_target: float = 1.0
    label_centering: float = 0.5
    whitespace_balance: float = 2.0
    optical_center: float = 1.5
    fill: float = 1.0
    height_side_clutter: float = 1.5
    extension_lines: float = 0.3
    style_deviation: float = 0.0
    hard: float = 1000.0


class Targets(Strict):
    offset_ratio: float = 0.06
    fill: float = 0.85  # group bbox area / safe-area area


class Style(Strict):
    name: str
    fonts_dir: str = "fonts"
    font_search_dirs: list[str] = Field(default_factory=list)
    # family -> stand-in used when the licensed files are missing; flagged in report.json
    font_fallbacks: dict[str, str] = Field(default_factory=dict)
    canvas: Canvas = Canvas()
    tokens: Tokens
    layout: Layout
    annotations: Annotations = Annotations()
    fit: FitCfg = FitCfg()
    search: Search = Search()
    weights: Weights = Weights()
    targets: Targets = Targets()
