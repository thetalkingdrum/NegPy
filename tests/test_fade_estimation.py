"""Per-image survival-ratio estimation for Fade Restoration (features/process/fade.py).
Mirrors tests/test_sensor_calibration.py in spirit: a measure step, a build step, and
the fail-closed reasons in between.
"""

import numpy as np

from negpy.features.exposure.normalization import resolve_fade_matrix, unmix_log_image
from negpy.features.process.fade import (
    RATIO_BOUNDS,
    SPAN_FLOOR,
    estimate_fade_ratios,
    fade_ratios_from_spans,
)
from negpy.features.process.models import ProcessMode


def _synthetic_capture(span_r: float, span_g: float, span_b: float, seed: int = 0) -> np.ndarray:
    """A synthetic (64, 64, 3) linear capture whose per-channel P1-P99 log-density span
    is exactly (span_r, span_g, span_b): one shared per-pixel factor `t`, scaled per
    channel, so any two channels' spans are in the exact ratio of their span argument
    regardless of block-median or percentile sampling (both commute with a positive
    per-channel scale)."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 1.0, (64, 64)).astype(np.float64)
    log_r, log_g, log_b = -t * span_r, -t * span_g, -t * span_b
    image = np.stack([10.0**log_r, 10.0**log_g, 10.0**log_b], axis=-1)
    return image.astype(np.float32)


def _synthetic_measured_capture(ratio_g: float, ratio_b: float, delta: tuple, span: float = 1.0, seed: int = 0) -> np.ndarray:
    """A synthetic (64, 64, 3) linear capture simulating a real scan: concentration-space
    log-densities (equal unfaded spans, scaled by the given survival ratios) are mixed
    through the dye set's own side-absorption matrix S before being written out -- the
    measured-density domain measure_channel_spans actually reads. Unlike _synthetic_capture,
    a channel's *measured* span here is not simply proportional to its survival ratio,
    because S mixes the channels -- this is what the delta-aware unmix in
    measure_channel_spans must undo."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 1.0, (64, 64)).astype(np.float64)
    d_gr, d_br, d_rg, d_bg, d_rb, d_gb = delta
    s_matrix = np.array([[1.0, d_gr, d_br], [d_rg, 1.0, d_bg], [d_rb, d_gb, 1.0]])
    concentration_log = np.stack([-t * span, -t * span * ratio_g, -t * span * ratio_b], axis=-1)
    measured_log = concentration_log @ s_matrix.T
    return np.power(10.0, measured_log).astype(np.float32)


def test_unmix_by_delta_recovers_true_ratios_from_measured_density():
    """A channel's *measured* density span is not proportional to its survival ratio --
    the dye set's own side absorption mixes the channels and biases every ratio toward
    1.0 (under-reporting fade). Unmixing by the selected profile's delta first, as
    measure_channel_spans now does, recovers the true concentration-space ratios."""
    generic_e6_delta = (0.0689, 0.0111, 0.2246, 0.0486, 0.0854, 0.1815)
    ratio_g_true, ratio_b_true = 0.6, 0.6
    image = _synthetic_measured_capture(ratio_g_true, ratio_b_true, generic_e6_delta)

    ratio_g_unmixed, ratio_b_unmixed, reason = estimate_fade_ratios(image, ProcessMode.E6, None, 0.0, generic_e6_delta)
    assert reason == ""
    assert abs(ratio_g_unmixed - ratio_g_true) < 1e-2
    assert abs(ratio_b_unmixed - ratio_b_true) < 1e-2

    # Without delta, the same measured image reads as materially less faded -- the bias
    # this fix removes, not a hypothetical.
    ratio_g_biased, ratio_b_biased, reason_biased = estimate_fade_ratios(image, ProcessMode.E6, None, 0.0, None)
    assert reason_biased == ""
    assert ratio_g_biased - ratio_g_true > 0.1
    assert ratio_b_biased - ratio_b_true > 0.1


def test_round_trip_recovers_known_ratios_on_equal_span_source():
    """An "unfaded" source has equal spans across channels (Daniell's ideal-image
    assumption); applying a known fade (scaling green/blue spans by known ratios) and
    then estimating must recover those ratios."""
    ratio_g_true, ratio_b_true = 0.8, 1.3
    image = _synthetic_capture(1.2, 1.2 * ratio_g_true, 1.2 * ratio_b_true)
    ratio_g, ratio_b, reason = estimate_fade_ratios(image, ProcessMode.E6, None, 0.0)
    assert reason == ""
    assert abs(ratio_g - ratio_g_true) < 1e-3
    assert abs(ratio_b - ratio_b_true) < 1e-3


def test_documented_wrong_behavior_on_unequal_unfaded_spans():
    """A legitimately unequal-span scene with no fade at all still reads as non-unity.
    This is expected, documented wrong behavior (IMPLEMENT_FADE_AUTO.md §3) -- a
    monochromatic slide defeats any estimator built on this assumption -- not a bug for
    a future pass to "fix"."""
    image = _synthetic_capture(1.0, 1.6, 1.0)  # a genuinely more colorful green channel
    ratio_g, ratio_b, reason = estimate_fade_ratios(image, ProcessMode.E6, None, 0.0)
    assert reason == ""
    assert ratio_g != 1.0


def test_fails_closed_below_span_floor():
    ratio_g, ratio_b, reason = fade_ratios_from_spans((SPAN_FLOOR / 2, SPAN_FLOOR / 2, SPAN_FLOOR / 2))
    assert (ratio_g, ratio_b) == (1.0, 1.0)
    assert reason


def test_fails_closed_when_spans_agree():
    ratio_g, ratio_b, reason = fade_ratios_from_spans((1.0, 1.01, 0.99))
    assert (ratio_g, ratio_b) == (1.0, 1.0)
    assert reason


def test_clamps_and_reports_an_out_of_bounds_ratio():
    lo, hi = RATIO_BOUNDS
    ratio_g, ratio_b, reason = fade_ratios_from_spans((1.0, 10.0, 1.0))
    assert ratio_g == hi
    assert ratio_b == 1.0
    assert reason


def test_fails_closed_for_non_transparency():
    image = _synthetic_capture(1.0, 0.7, 1.4)
    ratio_g, ratio_b, reason = estimate_fade_ratios(image, ProcessMode.C41, None, 0.0)
    assert (ratio_g, ratio_b) == (1.0, 1.0)
    assert reason


def test_estimating_on_already_corrected_image_gives_a_different_answer():
    """Regression test for the feedback loop the estimator must avoid (IMPLEMENT_FADE_AUTO.md
    §2): estimating on an image the fade matrix has already touched gives a different --
    and wrong, if it were reused -- answer than estimating on the raw capture. This is why
    the estimate must always run on pre-composition data, never inside the render."""
    ratio_g_true, ratio_b_true = 0.7, 1.2
    image = _synthetic_capture(1.2, 1.2 * ratio_g_true, 1.2 * ratio_b_true)

    ratio_g_before, ratio_b_before, reason = estimate_fade_ratios(image, ProcessMode.E6, None, 0.0)
    assert reason == ""

    matrix = resolve_fade_matrix(1.0, ratio_g_before, ratio_b_before, None)
    img_log = np.log10(np.clip(image, 1e-6, 1.0))
    corrected_log = unmix_log_image(img_log, matrix)
    corrected_image = np.clip(10.0**corrected_log, 1e-6, 1.0).astype(np.float32)

    ratio_g_after, ratio_b_after, reason_after = estimate_fade_ratios(corrected_image, ProcessMode.E6, None, 0.0)
    assert abs(ratio_g_after - 1.0) < 1e-2  # the already-corrected image now reads as unfaded
    assert abs(ratio_b_after - 1.0) < 1e-2
    assert abs(ratio_g_after - ratio_g_before) > 0.1  # a materially different answer
