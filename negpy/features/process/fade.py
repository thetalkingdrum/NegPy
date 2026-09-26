"""Per-image dye-survival ratio estimation for Fade Restoration.

The six side-absorption ratios (`fade_delta`) are a property of the dye set, set once
per stock by a profile (services/assets/fade.py). The two survival *ratios* -- how much
this particular slide's green and blue layers have faded relative to red -- are a
property of one frame instead, and can only be estimated per image, which is what this
module does.

Red's own absolute survival (`fade_ratio_r`, `resolve_fade_matrix`'s third parameter) is
deliberately not estimated here: neutral image content only ever constrains a ratio
between two channels, never one channel's survival on its own, so no per-image
measurement can supply it. It stays at its slider default (or a physically-anchored
source, once one exists -- the rebate probe) rather than being guessed from the frame.

Reads off Cast Removal's own neutral-detection algorithm (`measure_neutral_axis_from_log`
-- midtone + shadow references, chroma-gated, cross-checked on NegPy's GPU parity tests)
rather than blind percentile spans, but does its own measurement rather than reusing
Cast Removal's already-computed `neutral_axis_refs`: those are metered downstream of
whatever fade correction is *already* configured, which is the identity at the default
survival ratios (1.0, 1.0, 1.0) regardless of delta -- so at the moment this feature is
actually used, nothing has removed the dye set's own measurement mixing yet.
`fade_measurement_unmix` does that unmixing explicitly, independent of the current ratio
state.
"""

from typing import Optional

import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.exposure.normalization import (
    LogNegativeBounds,
    fade_measurement_unmix,
    measure_neutral_axis_from_log,
    prefilter_log_grid,
    unmix_log_image,
)
from negpy.features.process.models import ProcessConfig, ProcessMode

#: Below this density spread (midtone - shadow) a channel's neutral axis is too flat to
#: be signal rather than noise.
SPREAD_FLOOR = 0.02
#: Spreads within this fraction of the largest agree -- no evidence of differential fade.
AGREEMENT_TOLERANCE = 0.05
#: Sane bound on an implied ratio; outside this the estimate is clamped and reported.
#: Reciprocal pair (0.2 = 1/5.0) so neither direction is favored. Wider than the original
#: 0.5-2.0: real severely-faded slides push a pure diagonal (delta=0, no profile yet)
#: correction past a factor of two on green and/or blue, and the ill-conditioning guard
#: in resolve_fade_matrix (reported via fade_reject_reason) is the real backstop, not
#: this bound -- see FADE_CONDITION_LIMIT in features/exposure/normalization.py.
RATIO_BOUNDS = (0.2, 5.0)
#: Bound on fade_ratio_r, red's own absolute survival fraction -- not a reciprocal pair
#: like RATIO_BOUNDS, since this is a fraction of original density, not a ratio between
#: two channels: it cannot physically exceed 1.0 (a fading process does not add density
#: back), and 0.05 keeps the restoration matrix's overall gain from a bare slip of the
#: slider running away before FADE_CONDITION_LIMIT has a chance to catch it.
RED_SURVIVAL_BOUNDS = (0.05, 1.0)

#: (midtone RGB triple, shadow RGB triple, highlight RGB triple or None, confidence) --
#: the return shape of normalization.measure_neutral_axis_from_log.
NeutralAxisRefs = tuple[tuple[float, float, float], tuple[float, float, float], object, float]


def measure_neutral_axis_ratios(
    image: ImageBuffer, roi: Optional[tuple], analysis_buffer: float, delta: Optional[tuple]
) -> tuple[Optional[NeutralAxisRefs], str]:
    """(refs, reject_reason) from the raw capture: unmixed by the selected profile's delta
    (`fade_measurement_unmix`, neutral-preserving so the detector's fixed luma bands still
    find pixels) before Cast Removal's own two-point neutral detection runs on it, against
    bounds measured from this frame rather than E-6's fixed window. A genuinely faded
    slide's own density span is compressed by exactly the amount `fade_ratio_r` exists to
    restore (see its docstring in models.py); against a window sized for an undegraded
    slide, the detector's luma bands find no content past roughly fade_ratio_r < 0.85 and
    fail outright rather than return a degraded estimate, on precisely the slides that need
    the estimate most. Falls back to reading measured density
    directly when no profile is selected -- a real, if delta-biased, estimate beats none.
    reject_reason is set (refs is None) when delta is degenerate or the detector finds no
    trustworthy neutral axis."""
    grid = prefilter_log_grid(image, roi, analysis_buffer)
    if delta is not None:
        unmix = fade_measurement_unmix(delta)
        if unmix is None:
            return None, "the dye-set side-absorption profile is degenerate — check the delta values"
        grid = unmix_log_image(grid, unmix)
    bounds = _measured_slide_bounds(grid)
    refs = measure_neutral_axis_from_log(grid, bounds, None, 0.0)
    if refs is None:
        return None, "no trustworthy neutral axis found on this frame"
    return refs, ""


def _measured_slide_bounds(grid: ImageBuffer) -> LogNegativeBounds:
    """This frame's own bounds in the slide orientation (floor above ceiling): the luma span
    from the base clip, each channel's offset from its extreme percentiles. The negative
    analyzer does not fit, since its dense end is chroma-gated for an orange mask."""
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    def _pct(p: float) -> list:
        return [float(np.percentile(grid[:, :, ch], p)) for ch in range(3)]

    base = float(EXPOSURE_CONSTANTS["base_luma_clip"])
    luma_f, luma_c = _pct(100.0 - base), _pct(base)
    color_f, color_c = _pct(100.0 - 0.00001), _pct(0.00001)
    mean_f, mean_c = sum(luma_f) / 3.0, sum(luma_c) / 3.0
    mid_f, mid_c = sorted(color_f)[1], sorted(color_c)[1]
    floors = tuple(mean_f + (color_f[ch] - mid_f) for ch in range(3))
    ceils = tuple(mean_c + (color_c[ch] - mid_c) for ch in range(3))
    return LogNegativeBounds(floors, ceils)


def fade_ratios_from_neutral_axis(refs: Optional[NeutralAxisRefs]) -> tuple[float, float, str]:
    """(ratio_g, ratio_b, reason) from two neutral references (midtone, shadow) already
    unmixed by `fade_measurement_unmix` -- `refs` must come from
    `measure_neutral_axis_ratios`, not from a render's own `neutral_axis_refs` metric,
    which is metered before that unmix.

    A channel's midtone-to-shadow spread is proportional to its survival fraction. Red is
    the fade matrix's reference channel (Cast Removal's is green: `neutral_axis_affine`
    lands red/blue on green's refs instead), so ratios here are spread-relative-to-red, not
    green.

    reason is "" for a real estimate, otherwise which fail-closed condition fired -- a
    silent identity is indistinguishable from a broken feature. A legitimately
    monochromatic slide reads as faded here; there is no way around that from image
    statistics alone, which is why the estimate is a suggestion, not a lock."""
    if refs is None:
        return 1.0, 1.0, "no neutral axis available"
    mid, shadow = refs[0], refs[1]
    spreads = tuple(float(mid[ch]) - float(shadow[ch]) for ch in range(3))
    r, g, b = spreads
    abs_spreads = (abs(r), abs(g), abs(b))
    if min(abs_spreads) < SPREAD_FLOOR:
        return 1.0, 1.0, f"a channel's neutral-axis spread is below the {SPREAD_FLOOR:g}-density noise floor"
    if max(abs_spreads) - min(abs_spreads) <= AGREEMENT_TOLERANCE * max(abs_spreads):
        return 1.0, 1.0, "channel spreads agree — no evidence of differential fade"
    ratio_g = g / r
    ratio_b = b / r
    lo, hi = RATIO_BOUNDS
    if not (lo <= ratio_g <= hi) or not (lo <= ratio_b <= hi):
        clamped_g = min(max(ratio_g, lo), hi)
        clamped_b = min(max(ratio_b, lo), hi)
        return clamped_g, clamped_b, f"implied ratio outside {lo:g}–{hi:g} — clamped"
    return ratio_g, ratio_b, ""


def estimate_fade_ratios(
    image: ImageBuffer, process_mode: Optional[str], roi: Optional[tuple], analysis_buffer: float, delta: Optional[tuple] = None
) -> tuple[float, float, str]:
    """Estimate (ratio_g, ratio_b) from the raw capture, or fail closed to (1.0, 1.0)
    with a reason. Never runs during render -- an explicit "Estimate" action bakes the
    result into ProcessConfig, following the sensor-calibration precedent
    (`sensor.measure_capture` / `build_sensor_matrix`).

    `delta` should be the currently selected fade profile's, if any: the estimate then
    depends on that profile's unmix, so a profile change invalidates a previous estimate."""
    if process_mode is not None and str(process_mode) != str(ProcessMode.E6):
        return 1.0, 1.0, "not a transparency"
    refs, reason = measure_neutral_axis_ratios(image, roi, analysis_buffer, delta)
    if refs is None:
        return 1.0, 1.0, reason
    return fade_ratios_from_neutral_axis(refs)


def fade_estimate_available(process: ProcessConfig) -> bool:
    """Whether the Estimate action can run at all; see estimate_fade_ratios for the
    per-image conditions under which it still reports no evidence."""
    return process.process_mode == ProcessMode.E6
