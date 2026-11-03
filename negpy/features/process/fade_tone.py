"""Tone NegPy and Adaptive: an automatic tone for a faded slide, written into the slide's own Print
Density and ISO-R Grade sliders, so no hidden curve acts. NegPy fits the two sliders so the transfer
curve shows the frame's tones as NegativeConverter's levels, brightness and S-curve would; Adaptive
maps the frame's own density span onto a print range with its median at a midtone."""

import math
from dataclasses import replace
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.process.models import FadeABTone
from negpy.features.process.path import RenderPath, render_path
from negpy.kernel.system.i18n import tr

#: Percentiles of the mean density the fit reads, light end first: their 0.5 and 99.5 % levels
#: points, their median and the quantiles between, so the fit weighs tones as the frame holds them.
TONE_PCT = (0.5, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 99.5)
#: Their display midtone after levels.
TONE_MIDTONE = 0.42
#: Their contrast: 20 + (1 - range) * 30, held to 10..45, in percent of the S-curve.
TONE_CONTRAST = (20.0, 30.0, 10.0, 45.0)
#: Their brightness, -100 log2 of the display gamma that puts the median at TONE_MIDTONE, held to this range.
TONE_BRIGHTNESS = (-25.0, 60.0)
#: Below this density span a frame has no range to read.
TONE_MIN_SPAN = 0.05
#: Slider steps the fit searches: Print Density and ISO-R Grade.
TONE_DENSITY_STEP = 0.01
TONE_GRADE_STEP = 1.0
#: The exposure fields Tone NegPy writes and gives back, in the order `fade_tone_saved` holds them.
TONE_FIELDS = ("density", "grade", "auto_exposure", "auto_normalize_contrast")
#: Adaptive: the print density span the frame's 0.5 to 99.5 % span maps to: about the span a well-exposed
#: slide's picture holds.
ADAPTIVE_SPAN = 2.4
#: Adaptive: the contrast (transfer slope) it applies, at least the slide default (a frame that kept its
#: range keeps ISO-R 115) and at most 1.5 (ISO-R 77), so a frame that lost most of its range is not
#: stretched until its grain and stain grow into the picture.
ADAPTIVE_CONTRAST = (1.0, 1.5)
#: Adaptive: the print density of the frame's median, about a normally exposed slide's mid gray. At the
#: slide defaults (Print Density 1, ISO-R 115) the transfer curve prints each density as itself.
ADAPTIVE_MIDTONE = 1.0


def tone_points(density: ImageBuffer) -> Optional[tuple[float, ...]]:
    """The per-cell mean of a capture-density grid (the transfer curve's input, D = range * normalized
    log) at TONE_PCT, light end first; None with too few cells or no range."""
    cells = np.asarray(density, dtype=np.float64).reshape(-1, 3)
    cells = cells[np.isfinite(cells).all(axis=1)]
    if len(cells) < 64:
        return None
    points = tuple(float(v) for v in np.percentile(cells.mean(axis=1), TONE_PCT))
    if points[-1] - points[0] < TONE_MIN_SPAN:
        return None
    return points


def fade_tone_points(log_grid: ImageBuffer, process: Any) -> Optional[tuple[float, ...]]:
    """`tone_points` of the meter grid after the fade and Crossover, through the slide's window with
    its White and Black Point, as the transfer curve reads it."""
    from negpy.features.exposure.normalization import LogNegativeBounds, normalize_log_image
    from negpy.features.transparency.logic import TRANSFER_DENSITY_RANGE, transfer_bounds, transfer_point_offsets

    floors, ceils = transfer_bounds()
    wp3, bp3 = transfer_point_offsets(process)
    bounds = LogNegativeBounds(tuple(f + w for f, w in zip(floors, wp3)), tuple(c + b for c, b in zip(ceils, bp3)))
    return tone_points(np.asarray(normalize_log_image(log_grid, bounds)) * TRANSFER_DENSITY_RANGE)


def display(density: np.ndarray, positive_source: bool) -> np.ndarray:
    """The display value of a neutral at output *density* on the transfer curve, through the working
    TRC: the positive itself, or a camera RAW slide's baseline and display rendering."""
    from negpy.features.transparency.logic import TRANSFER_CONSTANTS, display_rendering
    from negpy.kernel.image.logic import working_oetf_encode

    linear = np.power(10.0, -np.maximum(np.asarray(density, dtype=np.float64), 0.0)).astype(np.float32)
    if not positive_source:
        linear = display_rendering(linear * np.float32(2.0 ** float(TRANSFER_CONSTANTS["transfer_baseline_ev"])))
    flat = np.ascontiguousarray(np.repeat(linear.reshape(-1, 1, 1), 3, axis=2))
    return np.asarray(working_oetf_encode(flat), dtype=np.float64)[:, 0, 0].reshape(np.shape(density))


def their_tone(x: np.ndarray, low: float, high: float, median: float) -> np.ndarray:
    """`defaultExpiredRescueParams` and the shared tone curve of `buildExpiredRescueStages` on display
    values *x*, with the frame's levels points and median in display."""
    from negpy.features.process.nc_rescue import _js_round, _soft_contrast

    span = max(0.02, high - low)
    leveled = min(max((median - low) / (high - low) if high - low >= 0.02 else median, 0.02), 0.98)
    brightness = _js_round(min(max(-100.0 * math.log2(math.log(TONE_MIDTONE) / math.log(leveled)), TONE_BRIGHTNESS[0]), TONE_BRIGHTNESS[1]))
    base, per_range, lo, hi = TONE_CONTRAST
    contrast = _js_round(min(max(base + (1.0 - (high - low)) * per_range, lo), hi)) / 100.0
    t = np.clip((np.asarray(x, dtype=np.float64) - low) / span, 0.0, 1.0) ** (2.0 ** (-brightness / 100.0))
    return np.clip(np.vectorize(lambda v: _soft_contrast(v, contrast))(t), 0.0, 1.0)


def tone_values(points: Sequence[float], positive_source: bool) -> dict[str, float]:
    """Print Density and ISO-R Grade, on the sliders' steps and ranges, whose transfer curve shows the
    frame's tones closest (least squares in display) to their tone curve applied to the frame as the
    slide's defaults show it."""
    from negpy.features.transparency.logic import TRANSFER_CONSTANTS

    d = np.asarray(points, dtype=np.float64)
    shown = display(d, positive_source)
    median = float(display(np.array([float(np.median(d))]), positive_source)[0])
    target = their_tone(shown, float(shown[-1]), float(shown[0]), median)
    pivot = float(TRANSFER_CONSTANTS["transfer_contrast_pivot"])
    ref = float(TRANSFER_CONSTANTS["transfer_grade_ref"])
    per_density = float(TRANSFER_CONSTANTS["transfer_density_stops"]) * math.log10(2.0)
    grades = np.arange(
        float(EXPOSURE_CONSTANTS["iso_r_min"]), float(EXPOSURE_CONSTANTS["iso_r_max"]) + 0.5 * TONE_GRADE_STEP, TONE_GRADE_STEP
    )
    densities = np.round(np.arange(0.0, 2.0 + 0.5 * TONE_DENSITY_STEP, TONE_DENSITY_STEP), 2)
    contrast = ref / grades[:, None, None]
    offset = ((1.0 - densities) * per_density)[None, :, None]
    out = display(pivot + (d[None, None, :] - offset - pivot) * contrast, positive_source)
    error = ((out - target[None, None, :]) ** 2).sum(axis=2)
    gi, di = np.unravel_index(int(np.argmin(error)), error.shape)
    return {"density": float(densities[di]), "grade": float(grades[gi])}


def adaptive_values(points: Sequence[float], positive_source: bool) -> dict[str, float]:
    """Print Density and ISO-R Grade, on the sliders' steps and ranges, that map the frame's 0.5 to
    99.5 % density span onto ADAPTIVE_SPAN within ADAPTIVE_CONTRAST, and print its median at
    ADAPTIVE_MIDTONE. *positive_source* is unused: the fit is in density on both paths."""
    from negpy.features.transparency.logic import TRANSFER_CONSTANTS

    d = np.asarray(points, dtype=np.float64)
    ref = float(TRANSFER_CONSTANTS["transfer_grade_ref"])
    pivot = float(TRANSFER_CONSTANTS["transfer_contrast_pivot"])
    per_density = float(TRANSFER_CONSTANTS["transfer_density_stops"]) * math.log10(2.0)
    lo_c, hi_c = ADAPTIVE_CONTRAST
    contrast = min(max(ADAPTIVE_SPAN / max(float(d[-1] - d[0]), TONE_MIN_SPAN), lo_c), hi_c)
    lo, hi = float(EXPOSURE_CONSTANTS["iso_r_min"]), float(EXPOSURE_CONSTANTS["iso_r_max"])
    grade = min(max(round(ref / contrast / TONE_GRADE_STEP) * TONE_GRADE_STEP, lo), hi)
    offset = float(np.median(d)) - pivot - (ADAPTIVE_MIDTONE - pivot) / (ref / grade)
    density = min(max(round((1.0 - offset / per_density) / TONE_DENSITY_STEP) * TONE_DENSITY_STEP, 0.0), 2.0)
    return {"density": round(density, 2), "grade": float(grade)}


#: The Tone choices that write the sliders, and their fit.
TONE_FITS = {FadeABTone.NEGPY: tone_values, FadeABTone.ADAPTIVE: adaptive_values}


def _blend(start: dict[str, float], fit: dict[str, float], strength: float) -> dict[str, float]:
    s = min(max(float(strength), 0.0), 1.0)
    density = start["density"] + s * (fit["density"] - start["density"])
    grade = start["grade"] + s * (fit["grade"] - start["grade"])
    return {
        "density": round(round(density / TONE_DENSITY_STEP) * TONE_DENSITY_STEP, 2),
        "grade": float(round(grade / TONE_GRADE_STEP) * TONE_GRADE_STEP),
    }


def fade_tone_changes(
    config: Any, tone: FadeABTone, metrics: Mapping[str, Any], strength: Optional[float] = None
) -> tuple[Optional[Any], str]:
    """(config, reason): *config* with Tone set to *tone* at *strength* (None keeps the frame's Tone
    Strength). NegPy and Adaptive write their fit into the Tone sliders, moved from the values they
    replace by the strength, and keep those values in `fade_tone_saved`; any other choice gives them
    back first, so Off restores the sliders exactly. None with a reason when the frame has no reading."""
    from negpy.features.transparency.logic import transfer_shown_values

    process, exposure = config.process, config.exposure
    strength = process.fade_tone_strength if strength is None else float(strength)
    saved = process.fade_tone_saved
    if saved is not None and process.fade_ab_tone in TONE_FITS:
        exposure = replace(exposure, **dict(zip(TONE_FIELDS, saved)))
        saved = None
    if tone in TONE_FITS:
        points = metrics.get("fade_tone_points")
        if points is None:
            return None, tr("No range to read on this frame.")
        saved = tuple(getattr(exposure, f) for f in TONE_FIELDS)
        fit = TONE_FITS[tone](points, render_path(process) is RenderPath.POSITIVE)
        # The values the sliders print now, an Auto Density or Auto Grade included.
        shown = transfer_shown_values(exposure, metrics)
        start = {f: float(shown.get(f, getattr(exposure, f))) for f in ("density", "grade")}
        values = fit if strength >= 1.0 else _blend(start, fit, strength)
        exposure = replace(exposure, **values, auto_exposure=False, auto_normalize_contrast=False)
    process = replace(process, fade_ab_tone=tone, fade_tone_saved=saved, fade_tone_strength=strength)
    return replace(config, process=process, exposure=exposure), ""
