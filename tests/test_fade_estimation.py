"""Survival-ratio estimation for Fade Restoration (features/process/fade.py), reading
Cast Removal's two-point neutral-detection algorithm rather than blind percentile spans.
Mirrors tests/test_sensor_calibration.py in spirit: a measure step, a build step, and
the fail-closed reasons in between.

Verified premise: at the default survival ratios (1.0, 1.0), resolve_fade_matrix's
similarity-transform fix makes the render's own unmix exactly the identity regardless of
delta -- so at the moment Estimate is actually used, nothing has removed the dye set's own
measurement mixing yet. The estimator must do its own delta-unmix rather than assume the
render already did (test_delta_unmix_is_required_even_at_default_ratios pins this).
"""

import numpy as np

from negpy.features.exposure.normalization import fade_measurement_unmix
from negpy.features.process.fade import (
    RATIO_BOUNDS,
    SPREAD_FLOOR,
    estimate_fade_ratios,
    fade_ratios_from_neutral_axis,
    measure_neutral_axis_ratios,
)
from negpy.features.process.models import ProcessMode

_GENERIC_E6_DELTA = (0.0689, 0.0111, 0.2246, 0.0486, 0.0854, 0.1815)


def _synthetic_neutral_slide(
    ratio_g: float, ratio_b: float, delta: tuple, span: float = 3.0, size: int = 200, seed: int = 0, ratio_r: float = 1.0
) -> np.ndarray:
    """A synthetic (size, size, 3) linear capture: every pixel is neutral by construction
    (a single shared per-pixel factor t, scaled per channel by the true survival ratios --
    concentration-space densities), then mixed through the dye set's own side-absorption
    matrix S -- what a real scan of a faded, perfectly neutral gray card would read as
    measured density. `span` covers E-6's full transfer window so both the midtone and
    shadow luma bands have real pixels to find *when ratio_r == 1* -- `ratio_r` scales the
    whole capture's density down the same way a real fade_ratio_r < 1 does, shrinking the
    actual span well below `span` regardless of what `span` is set to."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.03, 0.97, (size, size)).astype(np.float64)
    d_gr, d_br, d_rg, d_bg, d_rb, d_gb = delta
    s_matrix = np.array([[1.0, d_gr, d_br], [d_rg, 1.0, d_bg], [d_rb, d_gb, 1.0]])
    concentration_log = np.stack([-t * span * ratio_r, -t * span * ratio_r * ratio_g, -t * span * ratio_r * ratio_b], axis=-1)
    measured_log = concentration_log @ s_matrix.T
    return np.power(10.0, measured_log).astype(np.float32)


def test_recovers_known_ratios_through_the_real_detector():
    """End-to-end: a synthetic slide with known concentration-space survival ratios and
    real generic-E6 delta, run through the actual production detector
    (measure_neutral_axis_from_log) via measure_neutral_axis_ratios, recovers the true
    ratios via fade_ratios_from_neutral_axis. This is the regression test for the row-sum
    correction: without it, the recovered ratios are biased toward 1.0 by roughly the
    row-sum ratio (~20% on green, for this delta)."""
    ratio_g_true, ratio_b_true = 0.6, 0.85
    image = _synthetic_neutral_slide(ratio_g_true, ratio_b_true, _GENERIC_E6_DELTA)

    ratio_g, ratio_b, reason = estimate_fade_ratios(image, ProcessMode.E6, None, 0.0, _GENERIC_E6_DELTA)
    assert reason == ""
    assert abs(ratio_g - ratio_g_true) < 1e-2
    assert abs(ratio_b - ratio_b_true) < 1e-2


def test_recovers_known_ratios_on_a_severely_faded_frame():
    """The regression test for the measured-bounds fix: against E-6's fixed 0..3 window,
    a frame whose own density span is compressed by a real fade_ratio_r (here 0.1, so the
    frame's actual span is a tenth of what the fixed window expects) found no content in
    the detector's midtone/shadow luma bands at all and failed closed to (1.0, 1.0) --
    exactly the slides fade_ratio_r exists to help. Bounds measured from the frame itself
    put the detector's bands back on real content, recovering the true ratios regardless."""
    ratio_g_true, ratio_b_true = 0.6, 0.85
    image = _synthetic_neutral_slide(ratio_g_true, ratio_b_true, _GENERIC_E6_DELTA, ratio_r=0.1)

    ratio_g, ratio_b, reason = estimate_fade_ratios(image, ProcessMode.E6, None, 0.0, _GENERIC_E6_DELTA)
    assert reason == ""
    assert abs(ratio_g - ratio_g_true) < 1e-2
    assert abs(ratio_b - ratio_b_true) < 1e-2


def test_delta_unmix_is_required_even_at_default_ratios():
    """The bug this whole design avoids: reading the neutral axis on the *measured*-density
    grid (no delta unmix) gives a materially biased answer, even though a render composed
    at the default ratios (1.0, 1.0) would show this exact grid -- resolve_fade_matrix's
    F = I there regardless of delta. Confirms delta=None (skip the unmix) reproduces the
    bias measure_neutral_axis_ratios's own delta-aware unmix fixes."""
    ratio_g_true, ratio_b_true = 0.6, 0.85
    image = _synthetic_neutral_slide(ratio_g_true, ratio_b_true, _GENERIC_E6_DELTA)

    ratio_g_biased, ratio_b_biased, reason = estimate_fade_ratios(image, ProcessMode.E6, None, 0.0, None)
    assert reason == ""
    assert ratio_g_biased - ratio_g_true > 0.1  # biased toward 1.0, not a rounding difference
    assert ratio_b_biased - ratio_b_true > 0.05


def test_row_sum_correction_is_not_negligible():
    """fade_measurement_unmix's row-normalization (needed to keep the neutral-axis
    detector's fixed luma bands working) introduces a per-channel bias that
    fade_ratios_from_neutral_axis must divide back out. Confirm the correction factor
    itself is a real, double-digit-percent effect for the shipped generic E6 delta, not
    something safe to drop as a simplification."""
    found = fade_measurement_unmix(_GENERIC_E6_DELTA)
    assert found is not None
    _unmix, row_sums = found
    green_factor = row_sums[1] / row_sums[0]
    assert abs(green_factor - 1.0) > 0.1


def test_documented_wrong_behavior_on_unequal_unfaded_spreads():
    """A legitimately unequal-spread scene with no fade at all still reads as non-unity.
    Documented wrong behavior -- a monochromatic slide defeats any estimator built on this
    assumption -- not a bug for a future pass to "fix"."""
    image = _synthetic_neutral_slide(1.6, 1.0, (0.0,) * 6)  # a genuinely more colorful green channel, zero delta
    ratio_g, ratio_b, reason = estimate_fade_ratios(image, ProcessMode.E6, None, 0.0, None)
    assert reason == ""
    assert ratio_g != 1.0


def test_fails_closed_when_spreads_agree():
    ratio_g, ratio_b, reason = fade_ratios_from_neutral_axis(((1.0, 1.01, 0.99), (0.0, 0.0, 0.0), None, 1.0), None)
    assert (ratio_g, ratio_b) == (1.0, 1.0)
    assert reason


def test_fails_closed_below_spread_floor():
    tiny = SPREAD_FLOOR / 2
    ratio_g, ratio_b, reason = fade_ratios_from_neutral_axis(((tiny, tiny, tiny), (0.0, 0.0, 0.0), None, 1.0), None)
    assert (ratio_g, ratio_b) == (1.0, 1.0)
    assert reason


def test_clamps_and_reports_an_out_of_bounds_ratio():
    lo, hi = RATIO_BOUNDS
    ratio_g, ratio_b, reason = fade_ratios_from_neutral_axis(((1.0, 10.0, 1.0), (0.0, 0.0, 0.0), None, 1.0), None)
    assert ratio_g == hi
    assert ratio_b == 1.0
    assert reason


def test_no_neutral_axis_found_fails_closed():
    ratio_g, ratio_b, reason = fade_ratios_from_neutral_axis(None, None)
    assert (ratio_g, ratio_b) == (1.0, 1.0)
    assert reason


def test_fails_closed_for_non_transparency():
    image = _synthetic_neutral_slide(0.7, 1.4, _GENERIC_E6_DELTA)
    ratio_g, ratio_b, reason = estimate_fade_ratios(image, ProcessMode.C41, None, 0.0, _GENERIC_E6_DELTA)
    assert (ratio_g, ratio_b) == (1.0, 1.0)
    assert reason


def test_degenerate_delta_reported_by_measurement_step():
    """A delta so extreme S itself is not invertible (or its row sums are ~0) is reported
    by measure_neutral_axis_ratios before any detection runs, not silently ignored."""
    degenerate_delta = (0.9995,) * 6
    refs, reason = measure_neutral_axis_ratios(np.full((32, 32, 3), 0.5, dtype=np.float32), None, 0.0, degenerate_delta)
    assert refs is None
    assert reason
