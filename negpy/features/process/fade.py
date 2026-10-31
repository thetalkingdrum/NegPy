"""Per-image estimate of the green and blue dye survival for Fade Restoration.

Neutral content constrains only the ratios between channels, so red's absolute survival is
never estimated.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

import negpy_core
import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.kernel.system.i18n import tr
from negpy.features.exposure.normalization import (
    LogNegativeBounds,
    measure_neutral_axis_from_log,
    prefilter_log_grid,
)
from negpy.features.process.capture_color import apply_camera_matrix
from negpy.features.process.models import ProcessConfig, ProcessMode

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
#: Percentile of the analysis grid that White Anchor reads as each channel's white.
WHITE_PCT = 99.9
#: To suggest White Anchor, the brightest channel's white is within this density of full scale.
STAIN_FULL_SCALE_TOLERANCE = 0.1
#: To suggest White Anchor, blue is the darkest channel at the white, by at least this density.
#: Fitted with the tolerance above on three sets of scans only (NORAD, SilverFast, Nikon Scan).
STAIN_BLUE_MIN = 0.2

#: The return shape of `measure_neutral_axis_from_log`: midtone, shadow, highlight or None, confidence.
NeutralAxisRefs = tuple[tuple[float, float, float], tuple[float, float, float], object, float]


def measure_neutral_axis_ratios(image: ImageBuffer, roi: Optional[tuple], analysis_buffer: float) -> tuple[Optional[NeutralAxisRefs], str]:
    """(refs, reason) from the raw capture, metered against this frame's own bounds: the fixed
    transfer window finds no grays in a faded slide. `refs` is None, with a reason, when no
    trustworthy neutral axis exists."""
    return neutral_axis_from_grid(prefilter_log_grid(image, roi, analysis_buffer))


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


def measure_white(grid: ImageBuffer) -> tuple[float, float, float]:
    """Each channel's white: the log transmittance of its brightest cells in the analysis grid."""
    r, g, b = (float(v) for v in np.percentile(grid.reshape(-1, 3), WHITE_PCT, axis=0))
    return r, g, b


def white_anchored_grid(grid: ImageBuffer, white: tuple[float, float, float]) -> np.ndarray:
    """The grid as density above each channel's white, the space White Anchor renders in.
    A cell brighter than the white stops at full scale, as a decode of the divided scan would."""
    return np.minimum(grid - np.asarray(white, dtype=np.float32), 0.0).astype(np.float32)


def stain_suggested(white: tuple[float, float, float]) -> bool:
    """Whether the white reads as a yellow stain on clear film: the brightest channel near full
    scale, and blue clearly the darkest. A dim scan with no white object never qualifies."""
    brightest = max(white)
    return brightest >= -STAIN_FULL_SCALE_TOLERANCE and white[2] <= min(white[0], white[1]) and brightest - white[2] >= STAIN_BLUE_MIN


class EstimateMethod(StrEnum):
    """MIDTONE reads the midtone grays' own densities. SPREAD reads each channel's midtone-to-shadow
    spread, which cancels a per-channel density offset but follows a flat, noisy shadow."""

    MIDTONE = "midtone"
    SPREAD = "spread"


@dataclass(frozen=True)
class FadeEstimate:
    """Survivals from one Estimate. red is None when the method leaves Red Survival as it is;
    reason is "" for a real estimate (see fade_ratios_from_neutral_axis). white is the frame's
    measured white, None when the frame is not a slide."""

    red: Optional[float]
    green: float
    blue: float
    reason: str
    white: Optional[tuple[float, float, float]] = None


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
        out = [(n, raw, c) for n, raw, c in ((tr("green"), ratio_g, clamped_g), (tr("blue"), ratio_b, clamped_b)) if raw != c]
        reason = "; ".join(
            tr(
                "Implied {channel} {raw} is outside {lo}–{hi}, set to {clamped}",
                channel=n,
                raw=f"{raw:.1f}",
                lo=f"{lo:g}",
                hi=f"{hi:g}",
                clamped=f"{c:.1f}",
            )
            for n, raw, c in out
        )
        if any(raw > hi for _n, raw, _c in out):
            reason += ". " + tr("Red has faded further than this range covers: lower Red Survival by eye")
        return clamped_g, clamped_b, reason
    return ratio_g, ratio_b, ""


def estimate_fade_ratios(
    image: ImageBuffer, process_mode: Optional[str], roi: Optional[tuple], analysis_buffer: float
) -> tuple[float, float, str]:
    """(ratio_g, ratio_b, reason) from the raw capture, failing closed to 1.0 with a reason.
    Runs on request, never during a render."""
    if process_mode is not None and str(process_mode) != str(ProcessMode.E6):
        return 1.0, 1.0, tr("This frame is not a slide.")
    refs, reason = measure_neutral_axis_ratios(image, roi, analysis_buffer)
    if refs is None:
        return 1.0, 1.0, reason
    return fade_ratios_from_neutral_axis(refs)


def estimate_fade(
    image: ImageBuffer,
    process_mode: Optional[str],
    roi: Optional[tuple],
    analysis_buffer: float,
    cam_matrix: Optional[np.ndarray] = None,
    white_anchor: bool = False,
) -> dict[EstimateMethod, FadeEstimate]:
    """Both methods' estimates from one measurement of the grays. MIDTONE also sets Red Survival so
    the best-kept layer lands at 1.0; a ratio above 1 says red faded most. `cam_matrix` takes a
    camera buffer into the working space the render applies the fade in. With `white_anchor`, the
    grays are read as density above the frame's white, as White Anchor renders them."""
    if process_mode is not None and str(process_mode) != str(ProcessMode.E6):
        reason = tr("This frame is not a slide.")
        return {m: FadeEstimate(None, 1.0, 1.0, reason) for m in EstimateMethod}
    if cam_matrix is not None:
        image = apply_camera_matrix(negpy_core.nan_to_num(np.ascontiguousarray(image, dtype=np.float32), 1e-6, 1.0, 1e-6), cam_matrix)
    grid = prefilter_log_grid(image, roi, analysis_buffer)
    white = measure_white(grid)
    refs, reason = neutral_axis_from_grid(white_anchored_grid(grid, white) if white_anchor else grid)
    out = {}
    for method in EstimateMethod:
        if refs is None:
            out[method] = FadeEstimate(None, 1.0, 1.0, reason, white)
            continue
        g, b, why = fade_ratios_from_neutral_axis(refs, method)
        red = None
        if method == EstimateMethod.MIDTONE and not (why and g == 1.0 and b == 1.0):
            lo, hi = RED_SURVIVAL_BOUNDS
            red = min(max(1.0 / max(g, b, 1.0), lo), hi)
        out[method] = FadeEstimate(red, g, b, why, white)
    return out


def fade_estimate_available(process: ProcessConfig) -> bool:
    """Whether the Estimate action can run: Transparency only."""
    return process.process_mode == ProcessMode.E6
