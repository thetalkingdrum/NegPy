"""Tone and Auto Red: an automatic tone for a faded slide. Tone writes the slide's own Print Density
and ISO-R Grade sliders, so no hidden curve acts: it maps the frame's density span onto a print range
with its median at a midtone. Auto Red restores the span through Red Survival instead."""

import math
from dataclasses import replace
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.exposure.models import EXPOSURE_CONSTANTS
from negpy.features.process.path import RenderPath, render_path
from negpy.kernel.system.i18n import tr

#: Percentiles of the mean density the fit reads, light end first: their 0.5 and 99.5 % levels
#: points, their median and the quantiles between, so the fit weighs tones as the frame holds them.
TONE_PCT = (0.5, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 99.5)
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
#: Auto Red: the lowest Red Survival it sets. A slide that never held a full range (fog, a dull day) reads
#: as a heavy fade, so the stretch stops here.
AUTO_RED_MIN = 0.4
#: Auto Red: the density span it maps the frame's 0.5 to 99.5 % span onto, below ADAPTIVE_SPAN because a
#: stretch by Red Survival also steepens the transfer curve's shoulder and toe.
AUTO_RED_SPAN = 1.6
#: Auto Red: the lowest density its shift leaves the frame's 0.5 % point at. The points are grid-cell means,
#: so the brightest pixels sit above them.
AUTO_RED_LIGHT = 0.2


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


def survival_points(
    grid: ImageBuffer,
    process: Any,
    survivals: tuple[float, float, float],
    white_anchor: bool,
    camera_raw: bool,
    white_space: str,
    shift: float = 0.0,
) -> Optional[tuple[float, ...]]:
    """`fade_tone_points` of the Estimate *grid* after *survivals* and *shift*, as a render with them
    reads it. The crossover table keeps each cell's mean density, so it does not move the points."""
    from negpy.features.process.fade import measure_white, survival_density, through_density

    density = survival_density(*survivals, measure_white(grid), white_anchor, camera_raw or bool(white_space), white_space, shift)
    return fade_tone_points(through_density(grid, density), process)


def auto_red(
    points_at: Callable[[float], Optional[Sequence[float]]], red: float, shift: float = 0.0
) -> tuple[Optional[dict[str, float]], str]:
    """(changes, reason): the Red Survival that maps the frame's 0.5 to 99.5 % density span onto
    AUTO_RED_SPAN, held to AUTO_RED_MIN and 1.0, and the density shift that keeps the median where it
    printed, cut so the 0.5 % point prints no lighter than AUTO_RED_LIGHT or than before. *points_at* gives
    the points at a red with *shift*. Every density scales by red / new red, so one reading finds it."""
    from negpy.features.process.fade import RED_SURVIVAL_BOUNDS

    points = points_at(red)
    if points is None or float(points[-1] - points[0]) < TONE_MIN_SPAN:
        return None, tr("No range to read on this frame.")
    span = float(points[-1] - points[0])
    new = round(min(max(float(red) * span / AUTO_RED_SPAN, AUTO_RED_MIN, RED_SURVIVAL_BOUNDS[0]), RED_SURVIVAL_BOUNDS[1]), 2)
    after = points_at(new)
    gain = 0.0
    if after is not None:
        gain = min(float(np.median(after)) - float(np.median(points)), float(after[0]) - min(AUTO_RED_LIGHT, float(points[0])))
    return {"fade_ratio_r": new, "fade_density_shift": round(float(shift) - gain, 4)}, ""


def _blend(start: dict[str, float], fit: dict[str, float], strength: float) -> dict[str, float]:
    s = min(max(float(strength), 0.0), 1.0)
    density = start["density"] + s * (fit["density"] - start["density"])
    grade = start["grade"] + s * (fit["grade"] - start["grade"])
    return {
        "density": round(round(density / TONE_DENSITY_STEP) * TONE_DENSITY_STEP, 2),
        "grade": float(round(grade / TONE_GRADE_STEP) * TONE_GRADE_STEP),
    }


def fade_tone_changes(config: Any, metrics: Mapping[str, Any], strength: float) -> tuple[Optional[Any], str]:
    """(config, reason): *config* with Tone at *strength*. Above 0 it writes `adaptive_values` into the
    Tone sliders, moved from the values they replace by the strength, and keeps those values in
    `fade_tone_saved`; 0 gives them back exactly. None with a reason when the frame has no reading."""
    from negpy.features.transparency.logic import transfer_shown_values

    process, exposure = config.process, config.exposure
    strength = min(max(float(strength), 0.0), 1.0)
    saved = process.fade_tone_saved
    if saved is not None:
        exposure = replace(exposure, **dict(zip(TONE_FIELDS, saved)))
        saved = None
    if strength > 0.0:
        points = metrics.get("fade_tone_points")
        if points is None:
            return None, tr("No range to read on this frame.")
        saved = tuple(getattr(exposure, f) for f in TONE_FIELDS)
        fit = adaptive_values(points, render_path(process) is RenderPath.POSITIVE)
        # The values the sliders print now, an Auto Density or Auto Grade included.
        shown = transfer_shown_values(exposure, metrics)
        start = {f: float(shown.get(f, getattr(exposure, f))) for f in ("density", "grade")}
        values = fit if strength >= 1.0 else _blend(start, fit, strength)
        exposure = replace(exposure, **values, auto_exposure=False, auto_normalize_contrast=False)
    process = replace(process, fade_tone_saved=saved, fade_tone_strength=strength)
    return replace(config, process=process, exposure=exposure), ""
