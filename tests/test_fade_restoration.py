"""Dye-fade restoration: a second 3x3 (`inv(F)`) composed into the crosstalk unmix
slot rather than run as its own stage. Mirrors tests/test_capture_unmix.py.
"""

import numpy as np

from negpy.features.exposure.normalization import (
    effective_crosstalk_matrix,
    resolve_crosstalk_matrix,
    resolve_fade_matrix,
)
from negpy.features.process.models import ProcessConfig, ProcessMode

_CROSSTALK = (1.0, -0.05, -0.02, -0.04, 1.0, -0.08, -0.01, -0.1, 1.0)
_ALPHA = (0.8, 0.6, 0.9)
_DELTA = (0.03, 0.02, 0.01, 0.04, 0.02, 0.03)  # (gr, br, rg, bg, rb, gb)


def _forward_fade_matrix(alpha, delta):
    """Independent transcription of the forward operator from IMPLEMENT_FADE_RESTORATION.md
    §3, built here rather than reused from production so a sign error in
    resolve_fade_matrix cannot cancel itself against this test."""
    ar, ag, ab = alpha
    d_gr, d_br, d_rg, d_bg, d_rb, d_gb = delta
    return np.array(
        [
            [ar, ag * d_gr, ab * d_br],
            [ar * d_rg, ag, ab * d_bg],
            [ar * d_rb, ag * d_gb, ab],
        ]
    )


def test_strength_zero_is_off():
    assert resolve_fade_matrix(0.0, _ALPHA, _DELTA) is None
    assert resolve_fade_matrix(1.0, None, _DELTA) is None
    assert resolve_fade_matrix(1.0, _ALPHA, None) is None


def test_fade_off_leaves_crosstalk_matrix_unchanged():
    """fade_strength = 0.0 must not perturb the crosstalk result, including when the
    crosstalk matrix itself is None (off)."""
    process_off_crosstalk = ProcessConfig(fade_strength=0.0, fade_alpha=_ALPHA, fade_delta=_DELTA)
    assert effective_crosstalk_matrix(process_off_crosstalk, ProcessMode.E6) is None

    process_on_crosstalk = ProcessConfig(
        crosstalk_strength=0.6,
        crosstalk_matrix=_CROSSTALK,
        crosstalk_process=ProcessMode.C41,
        fade_strength=0.0,
        fade_alpha=_ALPHA,
        fade_delta=_DELTA,
    )
    unmix_alone = resolve_crosstalk_matrix(0.6, _CROSSTALK)
    np.testing.assert_array_equal(effective_crosstalk_matrix(process_on_crosstalk, ProcessMode.C41), unmix_alone)


def test_identity_fade_composes_to_crosstalk_matrix():
    """delta = 0, alpha = 1: F is exactly the identity, so composing must reproduce the
    crosstalk matrix bit-for-bit (not merely close)."""
    process = ProcessConfig(
        crosstalk_strength=0.6,
        crosstalk_matrix=_CROSSTALK,
        crosstalk_process=ProcessMode.E6,
        fade_strength=1.0,
        fade_alpha=(1.0, 1.0, 1.0),
        fade_delta=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        fade_process=ProcessMode.E6,
    )
    unmix_alone = resolve_crosstalk_matrix(0.6, _CROSSTALK)
    composed = effective_crosstalk_matrix(process, ProcessMode.E6)
    assert composed is not None
    np.testing.assert_allclose(composed, unmix_alone, atol=1e-12)


def test_fade_matrix_round_trips_density():
    """Forward-fade then restore recovers the original density to numerical precision."""
    f = _forward_fade_matrix(_ALPHA, _DELTA)
    restore = resolve_fade_matrix(1.0, _ALPHA, _DELTA)
    assert restore is not None
    density = np.array([0.3, 0.9, 1.4])
    faded = f @ density
    restored = restore @ faded
    np.testing.assert_allclose(restored, density, atol=1e-10)


def test_fade_matrix_not_row_normalized():
    """Unlike resolve_crosstalk_matrix, a non-trivial fade matrix must not have unit row
    sums: fade changes each layer's neutral density, and normalizing that away would
    make the feature a no-op."""
    m = resolve_fade_matrix(1.0, _ALPHA, _DELTA)
    assert m is not None
    assert np.max(np.abs(np.sum(m, axis=1) - 1.0)) > 1e-6


def test_process_mode_gate_ignores_fade_for_mismatched_process():
    """A C-41 image with an E-6 fade profile selected: the fade gate must return None,
    leaving only the (mode-matched) crosstalk unmix."""
    process = ProcessConfig(
        crosstalk_strength=0.6,
        crosstalk_matrix=_CROSSTALK,
        crosstalk_process=ProcessMode.C41,
        fade_strength=1.0,
        fade_alpha=_ALPHA,
        fade_delta=_DELTA,
        fade_process=ProcessMode.E6,
    )
    unmix_alone = resolve_crosstalk_matrix(0.6, _CROSSTALK)
    np.testing.assert_array_equal(effective_crosstalk_matrix(process, ProcessMode.C41), unmix_alone)


def test_ill_conditioned_fade_falls_back_to_uncomposed_matrix():
    """A singular/ill-conditioned (alpha, delta) must refuse rather than invert into
    garbage: the composed result stays exactly the uncomposed crosstalk matrix."""
    # Uniform delta d gives F = (1-d)*I + d*J (J = all-ones): eigenvalues (1-d) [x2] and
    # (1+2d) [x1], so cond = (1+2d)/(1-d) -> 148 at d=0.98, comfortably past the guard.
    bad_delta = (0.98, 0.98, 0.98, 0.98, 0.98, 0.98)
    assert resolve_fade_matrix(1.0, (1.0, 1.0, 1.0), bad_delta) is None

    process = ProcessConfig(
        crosstalk_strength=0.6,
        crosstalk_matrix=_CROSSTALK,
        crosstalk_process=ProcessMode.E6,
        fade_strength=1.0,
        fade_alpha=(1.0, 1.0, 1.0),
        fade_delta=bad_delta,
        fade_process=ProcessMode.E6,
    )
    unmix_alone = resolve_crosstalk_matrix(0.6, _CROSSTALK)
    np.testing.assert_array_equal(effective_crosstalk_matrix(process, ProcessMode.E6), unmix_alone)


def test_effective_crosstalk_matrix_is_a_pure_function_of_config():
    """CPU and GPU both call effective_crosstalk_matrix directly with (process, mode) and
    no cached state; identical inputs must give bit-identical outputs across independent
    calls, which is what keeps the two engines from being able to disagree."""
    process = ProcessConfig(
        crosstalk_strength=0.6,
        crosstalk_matrix=_CROSSTALK,
        crosstalk_process=ProcessMode.E6,
        fade_strength=0.7,
        fade_alpha=_ALPHA,
        fade_delta=_DELTA,
        fade_process=ProcessMode.E6,
    )
    a = effective_crosstalk_matrix(process, ProcessMode.E6)
    b = effective_crosstalk_matrix(process, ProcessMode.E6)
    np.testing.assert_array_equal(a, b)


def test_fade_alpha_delta_coerced_to_tuple():
    """Config is hashed for the render cache; a list here would silently break it."""
    cfg = ProcessConfig(fade_alpha=list(_ALPHA), fade_delta=list(_DELTA))
    assert isinstance(cfg.fade_alpha, tuple)
    assert isinstance(cfg.fade_delta, tuple)
