"""Per-image dye-survival ratio estimation for Fade Restoration.

The six side-absorption ratios (`fade_delta`) are a property of the dye set, set once
per stock by a profile (services/assets/fade.py). The two survival ratios -- how much
this particular slide's green and blue layers have faded relative to red -- are a
property of one frame instead, and can only be estimated per image. Mirrors
features/process/sensor.py in shape: a measure function, a build function, and a
reason string for the fail-closed cases, baked into ProcessConfig by an explicit
action rather than resolved during render (see IMPLEMENT_FADE_AUTO.md §2-3).
"""

from typing import Optional

import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.exposure.normalization import (
    fade_side_absorption_unmix,
    percentile_from_sorted,
    prefilter_log_grid,
    sorted_channel_grid,
    unmix_log_image,
)
from negpy.features.process.models import ProcessConfig, ProcessMode

#: Below this density span (P1-P99) a channel's spread is noise, not fade signal.
SPAN_FLOOR = 0.05
#: Spans within this fraction of the largest agree -- no evidence of differential fade.
AGREEMENT_TOLERANCE = 0.05
#: Sane bound on an implied ratio; outside this the estimate is clamped and reported.
#: Reciprocal pair (0.2 = 1/5.0) so neither direction is favored. Wider than the original
#: 0.5-2.0: real severely-faded slides push a pure diagonal (delta=0, no profile yet)
#: correction past a factor of two on green and/or blue, and the ill-conditioning guard
#: in resolve_fade_matrix (reported via fade_reject_reason) is the real backstop, not
#: this bound -- see FADE_CONDITION_LIMIT in features/exposure/normalization.py.
RATIO_BOUNDS = (0.2, 5.0)


def measure_channel_spans(
    image: ImageBuffer, roi: Optional[tuple], analysis_buffer: float, delta: Optional[tuple] = None
) -> tuple[float, float, float]:
    """Per-channel P1-P99 density span of the raw capture -- read once, before any
    fade composition, so the estimate cannot be derived from data its own correction
    has already touched. Independent of `LogNegativeBounds`: those are measured
    post-unmix (the feedback loop this avoids) and, on the default E-6 transfer path,
    not measured at all (`transfer_bounds` is a fixed, content-blind window with no
    per-channel span to read).

    `delta`, when given, unmixes the grid by the dye set's own side-absorption matrix
    first (`fade_side_absorption_unmix`): a channel's span in *measured* density is not
    proportional to its survival fraction, because the dye set's side absorption mixes
    the channels -- it biases every ratio toward 1.0, worse the more the slide has
    actually faded. Reading spans in *concentration* space (after unmixing) removes that
    bias. Falls back to reading measured density directly when no profile is selected."""
    grid = prefilter_log_grid(image, roi, analysis_buffer)
    unmix = fade_side_absorption_unmix(delta) if delta is not None else None
    if unmix is not None:
        grid = unmix_log_image(grid, unmix)
    sorted_grid = sorted_channel_grid(grid)
    lo = percentile_from_sorted(sorted_grid, 1.0)
    hi = percentile_from_sorted(sorted_grid, 99.0)
    spans = np.abs(hi - lo)
    return (float(spans[0]), float(spans[1]), float(spans[2]))


def fade_ratios_from_spans(spans: tuple[float, float, float]) -> tuple[float, float, str]:
    """(ratio_g, ratio_b, reason) from per-channel spans, ignoring the small
    off-diagonal terms (`D'_c ≈ α_c · D_c`). reason is "" for a real estimate,
    otherwise which fail-closed condition fired -- a silent identity is
    indistinguishable from a broken feature. A legitimately monochromatic slide
    (Daniell's own example is a sunset) reads as faded here; there is no way around
    that from image statistics alone, which is why the estimate is a suggestion, not
    a lock (see IMPLEMENT_FADE_AUTO.md §3)."""
    r, g, b = spans
    if min(r, g, b) < SPAN_FLOOR:
        return 1.0, 1.0, f"a channel span is below the {SPAN_FLOOR:g}-density noise floor"
    if max(r, g, b) - min(r, g, b) <= AGREEMENT_TOLERANCE * max(r, g, b):
        return 1.0, 1.0, "channel spans agree — no evidence of differential fade"
    ratio_g, ratio_b = g / r, b / r
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
    depends on that profile (see measure_channel_spans), so a profile change should be
    treated as invalidating a previous estimate."""
    if process_mode is not None and str(process_mode) != str(ProcessMode.E6):
        return 1.0, 1.0, "not a transparency"
    spans = measure_channel_spans(image, roi, analysis_buffer, delta)
    return fade_ratios_from_spans(spans)


def fade_estimate_available(process: ProcessConfig) -> bool:
    """Whether the Estimate action can run at all; see estimate_fade_ratios for the
    per-image conditions under which it still reports no evidence."""
    return process.process_mode == ProcessMode.E6
