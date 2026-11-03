"""Tone NegPy: NegativeConverter's automatic tone for a faded slide, read in density and written into
the slide's own Print Density and ISO-R Grade sliders, so no hidden curve acts."""

import math
from dataclasses import replace
from typing import Any, Mapping, Optional

import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.process.models import FadeABTone
from negpy.features.process.path import RenderPath, render_path
from negpy.kernel.system.i18n import tr

#: Percentiles of the mean density for the dense end, the median and the light end: theirs.
TONE_PCT = (99.5, 50.0, 0.5)
#: Their display midtone after levels.
TONE_MIDTONE = 0.42
#: Their contrast: 20 + (1 - range) * 30, held to 10..45, as a fraction of the S-curve.
TONE_CONTRAST = (20.0, 30.0, 10.0, 45.0)
#: Their brightness, -100 log2 of the display gamma that puts the median at TONE_MIDTONE, held to this range.
TONE_BRIGHTNESS = (-25.0, 60.0)
#: Their S-curve, 0.5 + tanh((x - 0.5) * 4) / (2 tanh 2), has this slope at the midtone.
TONE_S_SLOPE = 2.0 / math.tanh(2.0)
#: Below this density span a frame has no range to read.
TONE_MIN_SPAN = 0.05
#: The exposure fields Tone NegPy writes and gives back, in the order `fade_tone_saved` holds them.
TONE_FIELDS = ("density", "grade", "auto_exposure", "auto_normalize_contrast")


def tone_points(density: ImageBuffer) -> Optional[tuple[float, float, float]]:
    """(dense, median, light) of the per-cell mean of a capture-density grid (the transfer curve's
    input, D = range * normalized log), at TONE_PCT; None with too few cells or no range."""
    cells = np.asarray(density, dtype=np.float64).reshape(-1, 3)
    cells = cells[np.isfinite(cells).all(axis=1)]
    if len(cells) < 64:
        return None
    dense, median, light = (float(v) for v in np.percentile(cells.mean(axis=1), TONE_PCT))
    if dense - light < TONE_MIN_SPAN:
        return None
    return dense, median, light


def fade_tone_points(log_grid: ImageBuffer, process: Any) -> Optional[tuple[float, float, float]]:
    """`tone_points` of the meter grid after the fade and Crossover, through the slide's window with
    its White and Black Point, as the transfer curve reads it."""
    from negpy.features.exposure.normalization import LogNegativeBounds, normalize_log_image
    from negpy.features.transparency.logic import TRANSFER_DENSITY_RANGE, transfer_bounds, transfer_point_offsets

    floors, ceils = transfer_bounds()
    wp3, bp3 = transfer_point_offsets(process)
    bounds = LogNegativeBounds(tuple(f + w for f, w in zip(floors, wp3)), tuple(c + b for c, b in zip(ceils, bp3)))
    return tone_points(np.asarray(normalize_log_image(log_grid, bounds)) * TRANSFER_DENSITY_RANGE)


def _display(density: float, positive_source: bool) -> float:
    """The display value of a neutral at output *density* on the transfer curve, through the working TRC."""
    from negpy.features.transparency.logic import TRANSFER_CONSTANTS, display_rendering
    from negpy.kernel.image.logic import working_oetf_encode

    linear = 10.0 ** (-density)
    if not positive_source:
        gain = 2.0 ** float(TRANSFER_CONSTANTS["transfer_baseline_ev"])
        linear = float(display_rendering(np.array([gain * linear], dtype=np.float32))[0])
    return float(working_oetf_encode(np.array([[[linear] * 3]], dtype=np.float32))[0, 0, 0])


def display_density(display: float, positive_source: bool) -> float:
    """The output density at which the transfer curve shows a neutral at *display*."""
    lo, hi = 0.0, 6.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _display(mid, positive_source) > display else (lo, mid)
    return 0.5 * (lo + hi)


def midtone_density(leveled: float, positive_source: bool) -> float:
    """Their midtone move on our transfer curve: the median, shown at *leveled* output density, takes
    the display gamma toward TONE_MIDTONE that their brightness range allows."""
    shown = min(max(_display(leveled, positive_source), 0.02), 0.98)
    brightness = min(max(-100.0 * math.log2(math.log(TONE_MIDTONE) / math.log(shown)), TONE_BRIGHTNESS[0]), TONE_BRIGHTNESS[1])
    return display_density(shown ** (2.0 ** (-brightness / 100.0)), positive_source)


def tone_contrast(span: float) -> float:
    """The contrast factor about the pivot: levels from the span onto the window, times the midtone
    slope of their S-curve at the strength their missing-range rule gives."""
    from negpy.features.transparency.logic import TRANSFER_DENSITY_RANGE

    base, per_range, lo, hi = TONE_CONTRAST
    fraction = min(span / TRANSFER_DENSITY_RANGE, 1.0)
    strength = min(max(base + (1.0 - fraction) * per_range, lo), hi) / 100.0
    return TRANSFER_DENSITY_RANGE / span * (1.0 + strength * (TONE_S_SLOPE - 1.0))


def tone_values(points: tuple[float, float, float], positive_source: bool) -> dict[str, float]:
    """Print Density and ISO-R Grade for *points*: `tone_contrast` about the Grade pivot with the
    light end at white, then the median at `midtone_density`, held to the sliders' ranges."""
    from negpy.features.transparency.logic import TRANSFER_CONSTANTS

    dense, median, light = points
    pivot = float(TRANSFER_CONSTANTS["transfer_contrast_pivot"])
    lo, hi = float(EXPOSURE_CONSTANTS["iso_r_min"]), float(EXPOSURE_CONSTANTS["iso_r_max"])
    grade = min(max(float(TRANSFER_CONSTANTS["transfer_grade_ref"]) / tone_contrast(dense - light), lo), hi)
    contrast = float(TRANSFER_CONSTANTS["transfer_grade_ref"]) / grade
    # D' = pivot + (D - offset - pivot) * contrast. Levels put the light end at 0.
    target = midtone_density((median - light) * contrast, positive_source)
    offset = median - pivot - (target - pivot) / contrast
    density = 1.0 - offset / (float(TRANSFER_CONSTANTS["transfer_density_stops"]) * math.log10(2.0))
    return {"density": min(max(density, 0.0), 2.0), "grade": grade}


def fade_tone_changes(config: Any, tone: FadeABTone, metrics: Mapping[str, Any]) -> tuple[Optional[Any], str]:
    """(config, reason): *config* with Tone set to *tone*. NegPy writes `tone_values` into the Tone
    sliders and keeps the values it replaced in `fade_tone_saved`; any other choice gives them back
    first, so Off restores the sliders exactly. None with a reason when NegPy has no reading."""
    process, exposure = config.process, config.exposure
    saved = process.fade_tone_saved
    if saved is not None and process.fade_ab_tone == FadeABTone.NEGPY:
        exposure = replace(exposure, **dict(zip(TONE_FIELDS, saved)))
        saved = None
    if tone == FadeABTone.NEGPY:
        points = metrics.get("fade_tone_points")
        if points is None:
            return None, tr("No range to read on this frame.")
        saved = tuple(getattr(exposure, f) for f in TONE_FIELDS)
        values = tone_values(points, render_path(process) is RenderPath.POSITIVE)
        exposure = replace(exposure, **values, auto_exposure=False, auto_normalize_contrast=False)
    process = replace(process, fade_ab_tone=tone, fade_tone_saved=saved)
    return replace(config, process=process, exposure=exposure), ""
