"""Fade Restoration: the restoration matrix a Slide renders with, and the per-image estimate
of the dye survivals."""

import hashlib
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.kernel.system.i18n import mark, tr
from negpy.features.exposure.normalization import (
    LogNegativeBounds,
    effective_crosstalk_matrix,
    measure_neutral_axis_from_log,
)
from negpy.features.process.capture_color import camera_matrix_parts, is_source_profile
from negpy.features.process.logic import effective_highlight_reconstruction
from negpy.features.process.models import ProcessConfig
from negpy.features.process.path import RenderPath, render_path

#: Below this density spread a channel's neutral axis is too flat to be signal.
SPREAD_FLOOR = 0.02
#: Spreads within this fraction of the largest agree: no evidence of differential fade.
AGREEMENT_TOLERANCE = 0.05
#: Bound on an implied ratio, a reciprocal pair; the factor stays well conditioned.
RATIO_BOUNDS = (0.2, 5.0)
#: Bound on `fade_ratio_r`, a fraction of the original density, so it cannot exceed 1.0.
RED_SURVIVAL_BOUNDS = (0.05, 1.0)
#: Percentile of each channel's extremes for the frame's own bounds, read first.
COLOR_EXTREME_PCT = 0.00001
#: Below this confidence the first reading holds colored content, not grays.
MIN_CONFIDENCE = 0.2
#: The fallback bounds' percentile: a scan whose extreme cells are noise or scanner zeros sets no bounds.
ROBUST_EXTREME_PCT = 0.1
#: The fallback's own gate, stricter because its sector-free pass reads more colored content.
ROBUST_MIN_CONFIDENCE = 0.4
#: A cell at or below this log transmittance holds a decoded zero, not a density.
CLIPPED_LOG = -5.0
#: Percentile of the analysis grid that Remove Stain reads as each channel's white.
WHITE_PCT = 99.9
#: To suggest Remove Stain, the brightest channel's white is within this density of full scale.
STAIN_FULL_SCALE_TOLERANCE = 0.2
#: To suggest Remove Stain, blue's white is at least this density below both red's and green's.
STAIN_BLUE_MIN = 0.08
#: Remove Stain puts the white at least this log density below full scale, so clear areas do not clip.
FADE_WHITE_HEADROOM = 0.1


def resolve_fade_matrix(ratio_r: float, ratio_g: float, ratio_b: float) -> np.ndarray:
    """Restoration operator for a faded dye set: the inverse of diag(ar, ar*ag, ar*ab).

    Not row-normalized: fade changes each layer's neutral density, and that is the signal.
    `ratio_g` and `ratio_b` are survival ratios to red; `ratio_r` is red's absolute survival.
    """
    ar, ag, ab = float(ratio_r), float(ratio_g), float(ratio_b)
    return np.diag([1.0 / ar, 1.0 / (ar * ag), 1.0 / (ar * ab)])


def fade_matrix(process: ProcessConfig) -> Optional[np.ndarray]:
    """The restoration matrix, or None at unit survival or on a negative. A Positive slide takes
    it too, since a scanner file opens Positive."""
    ratios = (process.fade_ratio_r, process.fade_ratio_g, process.fade_ratio_b)
    if render_path(process) is RenderPath.PRINT or all(r == 1.0 for r in ratios):
        return None
    return resolve_fade_matrix(*ratios)


def fade_white_space(
    cam_xyz: Optional[Sequence[Sequence[float]]],
    camera_wb: Optional[Sequence[float]],
    crosstalk: Optional[np.ndarray] = None,
    highlight_mode: int = 0,
) -> str:
    """The space a camera RAW's log white is measured in: its white-balanced camera channels
    (`density_channels`) after the *crosstalk* unmix, named by the as-shot white balance, the unmix
    and the decode's *highlight_mode*, which rebuilds the clipped pixels the white is read from. The
    primaries after them do not name it. "" for a source with no camera matrix, whose white is the
    file's own full scale."""
    if cam_xyz is None or is_source_profile(cam_xyz):
        return ""
    gains, primaries = camera_matrix_parts(cam_xyz, camera_wb)
    if primaries is None:
        return ""
    named = b"" if gains is None else np.round(np.diag(gains).astype(np.float64), 4).tobytes()
    if crosstalk is not None:
        named += b"unmix" + np.round(np.asarray(crosstalk, dtype=np.float64), 4).tobytes()
    named += b"highlight" + int(highlight_mode).to_bytes(1, "little")
    return hashlib.sha1(_WHITE_SPACE_CHANNELS + named).hexdigest()[:12]


def render_white_space(
    process: ProcessConfig, process_mode: Optional[str], cam_xyz: Optional[Sequence[Sequence[float]]], camera_wb: Optional[Sequence[float]]
) -> str:
    """`fade_white_space` of a render with *process*: its crosstalk unmix and highlight mode included,
    "" on a negative, which takes no fade."""
    if render_path(process) is RenderPath.PRINT:
        return ""
    crosstalk = effective_crosstalk_matrix(process, process_mode)
    return fade_white_space(cam_xyz, camera_wb, crosstalk, effective_highlight_reconstruction(process))


# Names the channels in `fade_white_space`, so a tag hashed any other way reads as stale.
_WHITE_SPACE_CHANNELS = b"camera-gains"


class FadeDensity(NamedTuple):
    """`fade_density`'s result."""

    matrix: Optional[np.ndarray]
    anchor: Optional[tuple[float, float, float]]


def fade_density(process: ProcessConfig, process_mode: Optional[str], white_space: str = "") -> FadeDensity:
    """The log-density matrix a render applies, `F @ U` (the fade after the crosstalk unmix), and
    the anchor it scales about. A stored white w, read after U, makes the matrix the affine
    `F (U L - w) + t` about `a = U^-1 w`. The anchor is a camera RAW's a, else None
    (docs/PIPELINE.md, Fade restoration)."""
    unmix = effective_crosstalk_matrix(process, process_mode)
    fade = fade_matrix(process)
    matrix = unmix if fade is None else (fade if unmix is None else fade @ unmix)
    terms = _white_terms(process, white_space, fade is not None)
    if terms is None:
        return FadeDensity(matrix, None)
    white, target = terms
    a = white if unmix is None else _unmixed_point(unmix, white)
    linear = np.eye(3) if matrix is None else matrix
    affine = np.hstack([linear, (target - linear @ a)[:, None]])
    return FadeDensity(affine, (float(a[0]), float(a[1]), float(a[2])) if white_space else None)


def _unmixed_point(unmix: np.ndarray, point: np.ndarray) -> np.ndarray:
    """The log point that *unmix* maps to *point*; *point* itself when *unmix* has no inverse."""
    try:
        return np.linalg.solve(np.asarray(unmix, dtype=np.float64), point)
    except np.linalg.LinAlgError:
        return point


def unmix_with_fade(process: ProcessConfig, process_mode: Optional[str], white_space: str = "") -> Optional[np.ndarray]:
    """`fade_density`'s matrix."""
    return fade_density(process, process_mode, white_space).matrix


def _white_terms(process: ProcessConfig, white_space: str, fade_acts: bool) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """(white, target) of the stored white's affine, both after the crosstalk unmix, or None when no
    white acts. *fade_acts*: Fade Restoration applies. Crosstalk alone keeps a neutral neutral, so a
    camera RAW needs no anchor for it."""
    white = _fade_white(process, white_space)
    if white is None:
        return None
    if process.fade_white_anchor:
        return white, np.full(3, _white_target(white))
    if white_space and fade_acts:
        level = np.asarray(density_anchor(white, False), dtype=np.float64)
        return level, level
    return None


def density_anchor(white: Sequence[float], remove_stain: bool) -> tuple[float, float, float]:
    """The log point a camera RAW's fade scales density about: each channel's white with Remove Stain,
    else the white's cleanest channel on all three. A neutral point keeps a gray neutral when the white
    is a colored highlight, which is all a slide shows when it holds no clear film."""
    if remove_stain:
        return (float(white[0]), float(white[1]), float(white[2]))
    level = float(max(white))
    return (level, level, level)


def _fade_white(process: ProcessConfig, white_space: str = "") -> Optional[np.ndarray]:
    """The stored per-channel white (log transmittance) when it applies to this render, else None.
    A white measured in another space than the render's reads the wrong channels, so it does not apply."""
    if render_path(process) is RenderPath.PRINT or process.fade_white_space != white_space:
        return None
    white = np.array([process.fade_white_r, process.fade_white_g, process.fade_white_b], dtype=np.float64)
    return white if bool(np.any(white != 0.0)) else None


def _white_target(white: np.ndarray) -> float:
    """The cleanest channel's white, capped FADE_WHITE_HEADROOM below full scale, so a dim scan keeps its exposure."""
    return min(float(white.max()), -FADE_WHITE_HEADROOM)


def fade_white_target(process: ProcessConfig, white_space: str = "") -> Optional[float]:
    """The neutral log level Remove Stain maps the stored white to in this render, else None."""
    white = _fade_white(process, white_space) if process.fade_white_anchor else None
    return None if white is None else _white_target(white)


def fade_white_stale(process: ProcessConfig, white_space: str) -> bool:
    """Whether a camera RAW's fade needs its white measured again: Fade Restoration or Remove Stain
    acts, and the stored white was measured in another space (or never)."""
    if not white_space or render_path(process) is RenderPath.PRINT or process.fade_white_space == white_space:
        return False
    return process.fade_white_anchor or fade_matrix(process) is not None


#: A white a render measured for itself: (r, g, b, fade_white_space).
MeasuredWhite = tuple[float, float, float, str]


def resolve_fade_white(process: ProcessConfig, white_space: str, grid: Callable[[], ImageBuffer]) -> Optional[MeasuredWhite]:
    """The white a render measures on the Estimate *grid* when the stored one is stale, else None.
    Each engine measures it before the density stage, so a frame renders alike everywhere."""
    if not fade_white_stale(process, white_space):
        return None
    r, g, b = measure_white(grid()) or (0.0, 0.0, 0.0)
    return (r, g, b, white_space)


def with_fade_white(process: ProcessConfig, white: Optional[MeasuredWhite]) -> ProcessConfig:
    """*process* carrying a white `resolve_fade_white` measured; unchanged with none."""
    if white is None:
        return process
    return replace(process, fade_white_r=white[0], fade_white_g=white[1], fade_white_b=white[2], fade_white_space=white[3])


#: The return shape of `measure_neutral_axis_from_log`: midtone, shadow, highlight or None, confidence.
NeutralAxisRefs = tuple[tuple[float, float, float], tuple[float, float, float], object, float]


def measure_neutral_axis_ratios(grid: ImageBuffer) -> tuple[Optional[NeutralAxisRefs], str]:
    """(refs, reason) from the transparency meter's grid, metered against this frame's own
    bounds: the fixed transfer window finds no grays in a faded slide. `refs` is None, with a
    reason, when no trustworthy neutral axis exists."""
    return neutral_axis_from_grid(grid)


def neutral_axis_from_grid(grid: ImageBuffer) -> tuple[Optional[NeutralAxisRefs], str]:
    """`measure_neutral_axis_ratios` on a prefiltered log grid."""
    refs = measure_neutral_axis_from_log(grid, _measured_slide_bounds(grid, COLOR_EXTREME_PCT), None, 0.0)
    if refs is not None and refs[3] >= MIN_CONFIDENCE:
        return refs, ""
    # A faded slide's grays carry the fade's cast, which can put them in the foliage or sky
    # sector; the unfiltered set is the last try.
    bounds = _measured_slide_bounds(grid, ROBUST_EXTREME_PCT)
    for exclude in (True, False):
        refs = measure_neutral_axis_from_log(grid, bounds, None, 0.0, exclude_subjects=exclude)
        if refs is not None and refs[3] >= ROBUST_MIN_CONFIDENCE:
            return refs, ""
    return None, tr("No reliable grays found on this frame.")


def _measured_slide_bounds(grid: ImageBuffer, extreme_pct: float = COLOR_EXTREME_PCT) -> LogNegativeBounds:
    """This frame's own bounds in the slide orientation (floor above ceiling): the luma span
    from the base clip, each channel's offset from its extreme percentiles. The negative
    analyzer does not fit, since its dense end is chroma-gated for an orange mask."""
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    cells = grid.reshape(-1, 3)
    unclipped = cells[(cells > CLIPPED_LOG).all(axis=1)]
    if len(unclipped):
        cells = unclipped

    def _pct(p: float) -> list:
        return [float(np.percentile(cells[:, ch], p)) for ch in range(3)]

    base = float(EXPOSURE_CONSTANTS["base_luma_clip"])
    luma_f, luma_c = _pct(100.0 - base), _pct(base)
    color_f, color_c = _pct(100.0 - extreme_pct), _pct(extreme_pct)
    mean_f, mean_c = sum(luma_f) / 3.0, sum(luma_c) / 3.0
    mid_f, mid_c = sorted(color_f)[1], sorted(color_c)[1]
    floors = tuple(mean_f + (color_f[ch] - mid_f) for ch in range(3))
    ceils = tuple(mean_c + (color_c[ch] - mid_c) for ch in range(3))
    return LogNegativeBounds(floors, ceils)


def measure_white(grid: ImageBuffer) -> Optional[tuple[float, float, float]]:
    """Each channel's white: the log transmittance of its brightest cells in the analysis grid, or
    None with no cell left. A cell at full scale in every channel is light past the film (a mount gap,
    a leak or a blown specular), and a decoded zero is no reading, so neither counts."""
    cells = grid.reshape(-1, 3)
    cells = cells[(cells > CLIPPED_LOG).all(axis=1) & (cells < 0.0).any(axis=1)]
    if not len(cells):
        return None
    r, g, b = (float(v) for v in np.percentile(cells, WHITE_PCT, axis=0))
    return r, g, b


def white_anchored_grid(grid: ImageBuffer, white: tuple[float, float, float]) -> np.ndarray:
    """The grid as density above each channel's white, the space Remove Stain renders in.
    A cell brighter than the white stops at full scale, as a decode of the divided scan would."""
    return np.minimum(grid - np.asarray(white, dtype=np.float32), 0.0).astype(np.float32)


def stain_suggested(white: tuple[float, float, float]) -> bool:
    """Whether the white reads as a yellow stain on clear film: the brightest channel near full
    scale, and blue below both red and green. A red cast leaves blue near green, and a dim scan
    with no white object never qualifies."""
    return max(white) >= -STAIN_FULL_SCALE_TOLERANCE and min(white[0], white[1]) - white[2] >= STAIN_BLUE_MIN


class EstimateMethod(StrEnum):
    """MIDTONE reads the midtone grays' own densities. SPREAD reads each channel's midtone-to-shadow
    spread, which cancels a per-channel density offset but follows a flat, noisy shadow."""

    MIDTONE = "midtone"
    SPREAD = "spread"


@dataclass(frozen=True)
class FadeEstimate:
    """Survivals from one Estimate. red is None when no estimate was made;
    reason is "" for a real estimate (see fade_ratios_from_neutral_axis). white is the frame's
    measured white, None when the frame is not a slide or has no cell to read it from."""

    red: Optional[float]
    green: float
    blue: float
    reason: str
    white: Optional[tuple[float, float, float]] = None
    #: The crossover table measured after these survivals (`measure_crossover`), None with none.
    crossover: Optional[tuple[float, ...]] = None


def fade_ratios_from_neutral_axis(refs: Optional[NeutralAxisRefs], method: str = EstimateMethod.SPREAD) -> tuple[float, float, str]:
    """(ratio_g, ratio_b, reason) from neutral references made by `measure_neutral_axis_ratios`.

    Ratios are relative to red, the fade matrix's reference channel. `reason` is "" for a real
    estimate, else the fail-closed condition that fired, with 1.0 returned for both."""
    if refs is None:
        return 1.0, 1.0, tr("No grays to measure.")
    mid, shadow = refs[0], refs[1]
    if method == EstimateMethod.MIDTONE:
        # The refs are log transmittance, so density is the negative.
        spreads = tuple(-float(mid[ch]) for ch in range(3))
    else:
        spreads = tuple(float(mid[ch]) - float(shadow[ch]) for ch in range(3))
    r, g, b = spreads
    abs_spreads = (abs(r), abs(g), abs(b))
    if min(abs_spreads) < SPREAD_FLOOR:
        return 1.0, 1.0, tr("The grays in one channel are too flat to measure.")
    if max(abs_spreads) - min(abs_spreads) <= AGREEMENT_TOLERANCE * max(abs_spreads):
        return 1.0, 1.0, tr("The three layers faded equally, so there is nothing to estimate.")
    ratio_g = g / r
    ratio_b = b / r
    lo, hi = RATIO_BOUNDS
    if not (lo <= ratio_g <= hi) or not (lo <= ratio_b <= hi):
        clamped_g = min(max(ratio_g, lo), hi)
        clamped_b = min(max(ratio_b, lo), hi)
        sentences = [
            tr(template, raw=f"{raw:.1f}", lo=f"{lo:g}", hi=f"{hi:g}", clamped=f"{c:.1f}")
            for template, raw, c in (
                (mark("Green read {raw}, outside {lo} to {hi}, so it is set to {clamped}."), ratio_g, clamped_g),
                (mark("Blue read {raw}, outside {lo} to {hi}, so it is set to {clamped}."), ratio_b, clamped_b),
            )
            if raw != c
        ]
        if ratio_g > hi or ratio_b > hi:
            sentences.append(tr("Red faded more than this range covers. Lower Red Survival by eye."))
        reason = " ".join(sentences)
        return clamped_g, clamped_b, reason
    return ratio_g, ratio_b, ""


def estimate_fade(grid: ImageBuffer, white_anchor: bool = False, camera_raw: bool = False) -> dict[EstimateMethod, FadeEstimate]:
    """Both methods' estimates from one measurement of the grays. `grid` is the transparency
    meter's grid with the crosstalk unmix, no fade and no Remove Stain (`TransparencyBaseProcessor`).
    Both methods set Red Survival so the best-kept layer lands at 1.0; a ratio above 1 says red
    faded most. With `white_anchor`, or on a `camera_raw` whose render scales density above its
    white (`unmix_with_fade`), the grays are read as density above the point the render scales about
    (`density_anchor`). Each estimate carries the crossover table measured after its own survivals."""
    white = measure_white(grid)
    if white_anchor and white is None:
        return {m: FadeEstimate(None, 1.0, 1.0, tr("No clear area to measure the stain on this frame.")) for m in EstimateMethod}
    anchored = white is not None and (white_anchor or camera_raw)
    refs, reason = neutral_axis_from_grid(white_anchored_grid(grid, density_anchor(white, white_anchor)) if anchored else grid)
    out = {}
    for method in EstimateMethod:
        if refs is None:
            out[method] = FadeEstimate(None, 1.0, 1.0, reason, white)
            continue
        g, b, why = fade_ratios_from_neutral_axis(refs, method)
        red = None
        if not (why and g == 1.0 and b == 1.0):
            lo, hi = RED_SURVIVAL_BOUNDS
            red = min(max(1.0 / max(g, b, 1.0), lo), hi)
        table = None
        if red is not None:
            table = measure_crossover(survival_grid(grid, red, g, b, white, white_anchor, camera_raw))
        out[method] = FadeEstimate(red, g, b, why, white, table)
    return out


# Crossover table: the per-density lean of the neutrals that the survivals leave. Ported from
# NegativeConverter's expired-film rescue (MIT, Copyright (c) 2025 lex; NOTICE.md) and moved from
# the display-encoded positive to log density after the fade matrix.

#: Band edges over the frame's own density range, dense end first.
CROSSOVER_BAND_EDGES = (0.0, 0.06, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0)
#: Percentiles of the weighted mean density that set the frame's own range.
CROSSOVER_RANGE_PCT = (0.5, 99.5)
#: Channel weights of the mean density. Equal weights keep each pixel's mean dye density, the
#: neutral the transfer curve renders; the dye channels carry no luma.
CROSSOVER_WEIGHTS = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
#: Channel spread (D) at half neutral weight: the source's 0.12 display units at a mid gray.
CROSSOVER_NEUTRAL_SPREAD = 0.25
#: Mean-shift radius (D, Chebyshev): the source's 0.06 display units at a mid gray.
CROSSOVER_CLUSTER_RADIUS = 0.12
#: Largest channel offset (D), about one stop: a larger residual says the survivals are wrong.
CROSSOVER_LIMIT = 0.3
#: A band needs this share of the total neutral weight and this much weight to count.
CROSSOVER_MIN_SHARE = 0.004
CROSSOVER_MIN_WEIGHT = 8.0
#: Below this density span the frame has no range to band.
CROSSOVER_MIN_SPAN = 0.05
CROSSOVER_MEAN_SHIFT_STEPS = 3
#: Bins of the per-pixel offset table between the outer band anchors.
CROSSOVER_BINS = 64
#: Values per band in a stored table: mean log, share, then the lean of each channel.
CROSSOVER_BAND_SIZE = 5


def _mean_log(cells: np.ndarray) -> np.ndarray:
    w = np.asarray(CROSSOVER_WEIGHTS, dtype=np.float64)
    return cells @ w


def measure_crossover(grid: ImageBuffer) -> Optional[tuple[float, ...]]:
    """The lean of the neutrals band by band on a log *grid* after the fade matrix, as a flat tuple of
    (mean log, share, lean r, g, b) per band, or None with too few cells. A lean is the band's
    neutral mean minus its mean log, so it sums to zero under CROSSOVER_WEIGHTS."""
    cells = np.asarray(grid, dtype=np.float64).reshape(-1, 3)
    cells = cells[(cells > CLIPPED_LOG).all(axis=1) & (cells < 0.0).any(axis=1)]
    if len(cells) < 64:
        return None
    mean = _mean_log(cells)
    lo, hi = (float(v) for v in np.percentile(mean, CROSSOVER_RANGE_PCT))
    span = max(hi - lo, CROSSOVER_MIN_SPAN)
    t = np.clip((mean - lo) / span, 0.0, 1.0)
    edges = np.asarray(CROSSOVER_BAND_EDGES[1:-1])
    band = np.searchsorted(edges, t, side="right")
    spread = (cells.max(axis=1) - cells.min(axis=1)) / CROSSOVER_NEUTRAL_SPREAD
    neutral = 1.0 / (1.0 + spread * spread)
    n_bands = len(CROSSOVER_BAND_EDGES) - 1
    weight = np.bincount(band, neutral, n_bands)
    total = float(weight.sum())
    if total <= 0.0:
        return None
    centre = np.full((n_bands, 3), np.nan)
    band_mean = np.full(n_bands, np.nan)
    for k in range(n_bands):
        if weight[k] > 0.0:
            sel = band == k
            centre[k] = (neutral[sel, None] * cells[sel]).sum(axis=0) / weight[k]
            band_mean[k] = float((neutral[sel] * mean[sel]).sum() / weight[k])
    for _ in range(CROSSOVER_MEAN_SHIFT_STEPS):
        live = ~np.isnan(centre[:, 0])
        d = np.abs(cells - np.where(live[band, None], centre[band], 0.0)).max(axis=1) / CROSSOVER_CLUSTER_RADIUS
        w = neutral / (1.0 + d * d)
        w = np.where(live[band], w, 0.0)
        shift_w = np.bincount(band, w, n_bands)
        for k in range(n_bands):
            if not live[k] or shift_w[k] < CROSSOVER_MIN_WEIGHT:
                continue
            sel = band == k
            centre[k] = (w[sel, None] * cells[sel]).sum(axis=0) / shift_w[k]
            band_mean[k] = float((w[sel] * mean[sel]).sum() / shift_w[k])
            weight[k] = shift_w[k]
    out: list[float] = []
    for k in range(n_bands):
        share = float(weight[k]) / total
        if np.isnan(centre[k, 0]) or share < CROSSOVER_MIN_SHARE or weight[k] < CROSSOVER_MIN_WEIGHT:
            continue
        # The band's mean log is the weighted mean of its centre, so the lean sums to zero exactly.
        level = float(_mean_log(centre[k][None, :])[0])
        lean = centre[k] - level
        out.extend(round(v, 5) for v in (level, share, *lean))
    return tuple(out) if out else None


def crossover_bands(table: Sequence[float]) -> np.ndarray:
    """A stored table as rows of (mean log, share, lean r, g, b), dense end first."""
    return np.asarray(table, dtype=np.float64).reshape(-1, CROSSOVER_BAND_SIZE)


def crossover_cast(table: Sequence[float]) -> np.ndarray:
    """The share-weighted mean lean of the bands: the overall cast."""
    rows = crossover_bands(table)
    return (rows[:, 1:2] * rows[:, 2:]).sum(axis=0) / max(float(rows[:, 1].sum()), 1e-12)


def _shape_cubic(xs: np.ndarray, ys: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson monotone cubic through (xs, ys), flat beyond the ends."""
    n = len(xs)
    if n == 1:
        return np.full_like(x, ys[0])
    h = np.diff(xs)
    delta = np.diff(ys) / h
    d = np.zeros(n)
    d[0], d[-1] = delta[0], delta[-1]
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] > 0.0:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    xc = np.clip(x, xs[0], xs[-1])
    i = np.clip(np.searchsorted(xs, xc, side="right") - 1, 0, n - 2)
    t = (xc - xs[i]) / h[i]
    t2, t3 = t * t, t * t * t
    return (2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h[i] * d[i] + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h[i] * d[i + 1]


class CrossoverOffsets(NamedTuple):
    """The per-pixel offset table: CROSSOVER_BINS rows of log offsets between mean logs lo and hi."""

    bins: np.ndarray
    lo: float
    hi: float


def crossover_offsets(table: Optional[Sequence[float]], crossover: float, neutralize: float = 0.0) -> Optional[CrossoverOffsets]:
    """The offsets a render adds in log, -(cast * neutralize + (lean - cast) * crossover), or None when
    nothing moves. Each row sums to zero under CROSSOVER_WEIGHTS and is scaled down as a whole to
    CROSSOVER_LIMIT, so the correction keeps the mean density and its own hue."""
    if not table or (crossover == 0.0 and neutralize == 0.0):
        return None
    rows = crossover_bands(table)
    cast = crossover_cast(table)
    xs = rows[:, 0]
    lo, hi = float(xs[0]), float(xs[-1])
    # Bands are dense end first, so the anchors already rise in mean log.
    x = np.linspace(lo, hi, CROSSOVER_BINS) if hi > lo else np.full(CROSSOVER_BINS, lo)
    lean = np.stack([_shape_cubic(xs, rows[:, 2 + c], x) for c in range(3)], axis=1)
    offset = -(cast * neutralize + (lean - cast) * crossover)
    offset -= _mean_log(offset)[:, None]
    peak = np.abs(offset).max(axis=1, keepdims=True)
    offset *= np.minimum(1.0, CROSSOVER_LIMIT / np.maximum(peak, 1e-12))
    if float(np.abs(offset).max()) <= 1e-4:
        return None
    return CrossoverOffsets(offset.astype(np.float32), lo, hi)


def apply_crossover(img_log: ImageBuffer, offsets: Optional[CrossoverOffsets]) -> ImageBuffer:
    """*img_log* plus the offset row its own mean log selects, linear between bins. Float32 in the
    shader's order (normalization.wgsl), so the engines agree."""
    if offsets is None:
        return img_log
    img = np.ascontiguousarray(img_log, dtype=np.float32)
    cells = img.reshape(-1, 3)
    m = (cells[:, 0] + cells[:, 1] + cells[:, 2]) * np.float32(1.0 / 3.0)
    lo, span = np.float32(offsets.lo), np.float32(offsets.hi - offsets.lo)
    if span > 0.0:
        pos = np.clip((m - lo) / span, np.float32(0.0), np.float32(1.0)) * np.float32(CROSSOVER_BINS - 1)
    else:
        pos = np.zeros_like(m)
    i0 = np.minimum(pos.astype(np.intp), CROSSOVER_BINS - 1)
    i1 = np.minimum(i0 + 1, CROSSOVER_BINS - 1)
    f = (pos - i0.astype(np.float32))[:, None]
    delta = offsets.bins[i0] * (np.float32(1.0) - f) + offsets.bins[i1] * f
    return (cells + delta).reshape(img.shape)


def crossover_uniform(offsets: Optional[CrossoverOffsets]) -> bytes:
    """The shader's table: (lo, hi, on, 0) then CROSSOVER_BINS rows of vec4, zeros when off."""
    if offsets is None:
        return bytes(16 * (CROSSOVER_BINS + 1))
    rows = np.zeros((CROSSOVER_BINS, 4), dtype=np.float32)
    rows[:, :3] = offsets.bins
    return np.array([offsets.lo, offsets.hi, 1.0, 0.0], dtype=np.float32).tobytes() + rows.tobytes()


def crossover_space(density: FadeDensity, white_space: str) -> str:
    """Names the log space a crossover table is measured in: the render's fade matrix, its anchor and
    the white space. A table tagged with another name was measured against other survivals."""
    named = b"" if density.matrix is None else np.round(np.asarray(density.matrix, dtype=np.float64), 5).tobytes()
    if density.anchor is not None:
        named += b"anchor" + np.round(np.asarray(density.anchor, dtype=np.float64), 5).tobytes()
    return hashlib.sha1(b"crossover" + named + white_space.encode()).hexdigest()[:12]


#: A table a render measured for itself: (table, fade_crossover_space).
MeasuredCrossover = tuple[Optional[tuple[float, ...]], str]


def resolve_crossover(
    process: ProcessConfig, space: str, grid: Callable[[], ImageBuffer], measure: bool = True
) -> tuple[Optional[tuple[float, ...]], Optional[MeasuredCrossover]]:
    """(table, measured): the table a render applies, and the one it measured on the after-fade *grid*
    when the stored one is stale, else None. Nothing is read with Crossover at 0. Without *measure*
    (a tile of an export) a stale table does not apply."""
    if render_path(process) is RenderPath.PRINT or process.fade_crossover == 0.0:
        return None, None
    if process.fade_crossover_space == space:
        return process.fade_crossover_table, None
    if not measure:
        return None, None
    table = measure_crossover(grid())
    return table, (table, space)


def with_fade_crossover(process: ProcessConfig, measured: Optional[MeasuredCrossover]) -> ProcessConfig:
    """*process* carrying a table `resolve_crossover` measured; unchanged with none."""
    if measured is None:
        return process
    return replace(process, fade_crossover_table=measured[0], fade_crossover_space=measured[1])


def survival_grid(
    grid: ImageBuffer,
    red: float,
    green: float,
    blue: float,
    white: Optional[tuple[float, float, float]],
    white_anchor: bool,
    camera_raw: bool,
) -> np.ndarray:
    """The Estimate *grid* through the fade a render with these survivals applies after the crosstalk unmix."""
    from negpy.features.exposure.normalization import unmix_log_image
    from negpy.features.process.models import ProcessMode

    space = "estimate" if camera_raw else ""
    w = white or (0.0, 0.0, 0.0)
    process = replace(
        ProcessConfig(),
        process_mode=ProcessMode.E6,
        fade_ratio_r=red,
        fade_ratio_g=green,
        fade_ratio_b=blue,
        fade_white_anchor=white_anchor,
        fade_white_r=w[0],
        fade_white_g=w[1],
        fade_white_b=w[2],
        fade_white_space=space,
    )
    density = fade_density(process, ProcessMode.E6, space)
    return np.asarray(unmix_log_image(np.ascontiguousarray(grid, dtype=np.float32), density.matrix, density.anchor))
