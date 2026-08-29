"""Dye-fade restoration: a second 3x3 (`inv(F)`) composed into the crosstalk unmix
slot rather than run as its own stage. Mirrors tests/test_capture_unmix.py.
"""

import numpy as np

from negpy.features.exposure.normalization import (
    effective_crosstalk_matrix,
    fade_delta_conflict_reason,
    fade_reject_reason,
    resolve_crosstalk_matrix,
    resolve_fade_matrix,
)
from negpy.features.process.models import ProcessConfig, ProcessMode

_CROSSTALK = (1.0, -0.05, -0.02, -0.04, 1.0, -0.08, -0.01, -0.1, 1.0)
_RATIO_G, _RATIO_B = 0.75, 1.125  # = 0.6/0.8, 0.9/0.8 -- the ratios behind the old (0.8, 0.6, 0.9) alpha
_DELTA = (0.03, 0.02, 0.01, 0.04, 0.02, 0.03)  # (gr, br, rg, bg, rb, gb)


def _roc_coefficients(ar, ag, ab, delta):
    """The six ROC-style ratio coefficients from NEGPY_FADE_RESTORATION_DESIGN.md §1,
    computed independently of any production code -- this is what "only ratios matter"
    actually claims: c1..c6 are invariant to a uniform scale of (ar, ag, ab)."""
    d_gr, d_br, d_rg, d_bg, d_rb, d_gb = delta
    return (
        ag * d_gr / ar,  # c1 (R<-G)
        ab * d_br / ar,  # c5 (R<-B)
        ar * d_rg / ag,  # c4 (G<-R)
        ab * d_bg / ag,  # c3 (G<-B)
        ar * d_rb / ab,  # c2 (B<-R)
        ag * d_gb / ab,  # c6 (B<-G)
    )


def _forward_fade_matrix(ratio_g, ratio_b, delta):
    """Independent transcription of the forward operator from IMPLEMENT_FADE_RESTORATION.md
    §3 / IMPLEMENT_FADE_AUTO.md §1, with the red layer's survival pinned at 1.0 -- built
    here rather than reused from production so a sign error in resolve_fade_matrix cannot
    cancel itself against this test."""
    ar, ag, ab = 1.0, ratio_g, ratio_b
    d_gr, d_br, d_rg, d_bg, d_rb, d_gb = delta
    return np.array(
        [
            [ar, ag * d_gr, ab * d_br],
            [ar * d_rg, ag, ab * d_bg],
            [ar * d_rb, ag * d_gb, ab],
        ]
    )


def test_ratio_invariance():
    """The load-bearing claim behind the whole reparameterization (IMPLEMENT_FADE_AUTO.md
    §1): scaling all three survival fractions by an arbitrary factor leaves all six
    ROC-style coefficients unchanged, because every one of them is a ratio of alphas.
    This is why only two of the three survival fractions are real degrees of freedom."""
    ar, ag, ab = 0.8, 0.6, 0.9
    base = _roc_coefficients(ar, ag, ab, _DELTA)
    for k in (0.3, 1.0, 2.7, 10.0):
        scaled = _roc_coefficients(ar * k, ag * k, ab * k, _DELTA)
        np.testing.assert_allclose(scaled, base, atol=1e-12)


def test_strength_zero_is_off():
    assert resolve_fade_matrix(0.0, _RATIO_G, _RATIO_B, _DELTA) is None


def test_no_delta_treated_as_zero_side_absorption():
    """delta=None (no profile selected) must not turn fade off outright: the survival
    ratios alone still give a real, if purely diagonal, correction."""
    with_none = resolve_fade_matrix(1.0, _RATIO_G, _RATIO_B, None)
    with_zero = resolve_fade_matrix(1.0, _RATIO_G, _RATIO_B, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert with_none is not None
    np.testing.assert_array_equal(with_none, with_zero)
    assert not np.allclose(with_none, np.eye(3))  # a real diagonal correction, not a no-op


def test_fade_off_leaves_crosstalk_matrix_unchanged():
    """fade_strength = 0.0 must not perturb the crosstalk result, including when the
    crosstalk matrix itself is None (off)."""
    process_off_crosstalk = ProcessConfig(fade_strength=0.0, fade_ratio_g=_RATIO_G, fade_ratio_b=_RATIO_B, fade_delta=_DELTA)
    assert effective_crosstalk_matrix(process_off_crosstalk, ProcessMode.E6) is None

    process_on_crosstalk = ProcessConfig(
        crosstalk_strength=0.6,
        crosstalk_matrix=_CROSSTALK,
        crosstalk_process=ProcessMode.C41,
        fade_strength=0.0,
        fade_ratio_g=_RATIO_G,
        fade_ratio_b=_RATIO_B,
        fade_delta=_DELTA,
    )
    unmix_alone = resolve_crosstalk_matrix(0.6, _CROSSTALK)
    np.testing.assert_array_equal(effective_crosstalk_matrix(process_on_crosstalk, ProcessMode.C41), unmix_alone)


def test_identity_fade_composes_to_crosstalk_matrix():
    """delta = 0, ratios = 1: F is exactly the identity, so composing must reproduce the
    crosstalk matrix bit-for-bit (not merely close). This is also ProcessConfig's default
    fade_ratio_g/fade_ratio_b, so a bare "Strength up" with no profile or ratio touched
    stays a no-op."""
    process = ProcessConfig(
        crosstalk_strength=0.6,
        crosstalk_matrix=_CROSSTALK,
        crosstalk_process=ProcessMode.E6,
        fade_strength=1.0,
        fade_ratio_g=1.0,
        fade_ratio_b=1.0,
        fade_delta=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        fade_process=ProcessMode.E6,
    )
    unmix_alone = resolve_crosstalk_matrix(0.6, _CROSSTALK)
    composed = effective_crosstalk_matrix(process, ProcessMode.E6)
    assert composed is not None
    assert unmix_alone is not None
    np.testing.assert_allclose(composed, unmix_alone, atol=1e-12)


def test_fade_matrix_round_trips_density():
    """Forward-fade then restore recovers the original density to numerical precision."""
    f = _forward_fade_matrix(_RATIO_G, _RATIO_B, _DELTA)
    restore = resolve_fade_matrix(1.0, _RATIO_G, _RATIO_B, _DELTA)
    assert restore is not None
    density = np.array([0.3, 0.9, 1.4])
    faded = f @ density
    restored = restore @ faded
    np.testing.assert_allclose(restored, density, atol=1e-10)


def test_fade_matrix_not_row_normalized():
    """Unlike resolve_crosstalk_matrix, a non-trivial fade matrix must not have unit row
    sums: fade changes each layer's neutral density, and normalizing that away would
    make the feature a no-op."""
    m = resolve_fade_matrix(1.0, _RATIO_G, _RATIO_B, _DELTA)
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
        fade_ratio_g=_RATIO_G,
        fade_ratio_b=_RATIO_B,
        fade_delta=_DELTA,
        fade_process=ProcessMode.E6,
    )
    unmix_alone = resolve_crosstalk_matrix(0.6, _CROSSTALK)
    np.testing.assert_array_equal(effective_crosstalk_matrix(process, ProcessMode.C41), unmix_alone)


def test_ill_conditioned_fade_falls_back_to_uncomposed_matrix():
    """A singular/ill-conditioned (ratios, delta) must refuse rather than invert into
    garbage: the composed result stays exactly the uncomposed crosstalk matrix."""
    # Uniform delta d gives F = (1-d)*I + d*J (J = all-ones): eigenvalues (1-d) [x2] and
    # (1+2d) [x1], so cond = (1+2d)/(1-d) -> 148 at d=0.98, comfortably past the guard.
    bad_delta = (0.98, 0.98, 0.98, 0.98, 0.98, 0.98)
    assert resolve_fade_matrix(1.0, 1.0, 1.0, bad_delta) is None

    process = ProcessConfig(
        crosstalk_strength=0.6,
        crosstalk_matrix=_CROSSTALK,
        crosstalk_process=ProcessMode.E6,
        fade_strength=1.0,
        fade_ratio_g=1.0,
        fade_ratio_b=1.0,
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
        fade_ratio_g=_RATIO_G,
        fade_ratio_b=_RATIO_B,
        fade_delta=_DELTA,
        fade_process=ProcessMode.E6,
    )
    a = effective_crosstalk_matrix(process, ProcessMode.E6)
    b = effective_crosstalk_matrix(process, ProcessMode.E6)
    np.testing.assert_array_equal(a, b)


def test_fade_delta_coerced_to_tuple():
    """Config is hashed for the render cache; a list here would silently break it."""
    cfg = ProcessConfig(fade_delta=list(_DELTA))
    assert isinstance(cfg.fade_delta, tuple)


def test_active_crosstalk_and_fade_delta_do_not_double_correct():
    """A crosstalk profile and fade_delta describe the same physical quantity -- a dye
    set's inherent side absorption. Both active for the same mode must not compose that
    absorption twice: fade_delta is dropped (survival ratios still apply)."""
    process = ProcessConfig(
        crosstalk_strength=1.0,
        crosstalk_matrix=_CROSSTALK,
        crosstalk_process=ProcessMode.E6,
        fade_strength=1.0,
        fade_ratio_g=_RATIO_G,
        fade_ratio_b=_RATIO_B,
        fade_delta=_DELTA,
        fade_process=ProcessMode.E6,
    )
    assert fade_delta_conflict_reason(process, ProcessMode.E6)

    composed = effective_crosstalk_matrix(process, ProcessMode.E6)
    ratios_only = resolve_fade_matrix(1.0, _RATIO_G, _RATIO_B, None)
    unmix_alone = resolve_crosstalk_matrix(1.0, _CROSSTALK)
    assert composed is not None and ratios_only is not None and unmix_alone is not None
    np.testing.assert_allclose(composed, ratios_only @ unmix_alone, atol=1e-12)

    # Without an active crosstalk profile there is no conflict, so the same fade_delta
    # is not dropped: the otherwise-identical config keeps its full (delta + ratios) factor.
    no_crosstalk = ProcessConfig(
        fade_strength=1.0, fade_ratio_g=_RATIO_G, fade_ratio_b=_RATIO_B, fade_delta=_DELTA, fade_process=ProcessMode.E6
    )
    full_fade = resolve_fade_matrix(1.0, _RATIO_G, _RATIO_B, _DELTA)
    no_crosstalk_composed = effective_crosstalk_matrix(no_crosstalk, ProcessMode.E6)
    assert fade_delta_conflict_reason(no_crosstalk, ProcessMode.E6) == ""
    assert no_crosstalk_composed is not None and full_fade is not None
    np.testing.assert_allclose(no_crosstalk_composed, full_fade, atol=1e-12)


def _crosstalk_active_config(**fade_kwargs) -> ProcessConfig:
    return ProcessConfig(
        crosstalk_strength=1.0,
        crosstalk_matrix=_CROSSTALK,
        crosstalk_process=ProcessMode.E6,
        fade_process=ProcessMode.E6,
        **fade_kwargs,
    )


def test_no_conflict_when_fade_delta_is_zero_or_off():
    """The guard only fires on a real conflict: an all-zero or absent fade_delta, or
    fade_strength = 0, must not trip it even with an active same-mode crosstalk profile."""
    assert fade_delta_conflict_reason(_crosstalk_active_config(fade_strength=1.0, fade_delta=None), ProcessMode.E6) == ""
    assert fade_delta_conflict_reason(_crosstalk_active_config(fade_strength=1.0, fade_delta=(0.0,) * 6), ProcessMode.E6) == ""
    assert fade_delta_conflict_reason(_crosstalk_active_config(fade_strength=0.0, fade_delta=_DELTA), ProcessMode.E6) == ""


def test_ill_conditioned_reject_reason_is_reported():
    """Defect 2: the ill-conditioned refusal must not be silent -- fade_reject_reason
    reports why, using the same guard resolve_fade_matrix applies."""
    assert fade_reject_reason(0.0, 1.0, 1.0, _DELTA) == ""  # off is not a rejection
    assert fade_reject_reason(1.0, 1.0, 1.0, (0.0,) * 6) == ""  # identity: nothing to reject

    bad_delta = (0.98, 0.98, 0.98, 0.98, 0.98, 0.98)
    reason = fade_reject_reason(1.0, 1.0, 1.0, bad_delta)
    assert reason
    assert resolve_fade_matrix(1.0, 1.0, 1.0, bad_delta) is None


def test_fade_profile_excluded_from_base_cache_key():
    """fade_profile is a display label; only the baked fade_delta should invalidate the
    base-stage cache (mirrors crosstalk_profile). Adding it back would force a re-render
    on every profile-name change with no numeric effect."""
    from negpy.domain.models import WorkspaceConfig
    from negpy.services.rendering.engine import DarkroomEngine

    engine = DarkroomEngine()
    img = np.random.rand(64, 64, 3).astype(np.float32)
    settings = WorkspaceConfig.from_flat_dict(
        {
            "process_mode": ProcessMode.E6,
            "fade_strength": 0.5,
            "fade_ratio_g": _RATIO_G,
            "fade_ratio_b": _RATIO_B,
            "fade_delta": list(_DELTA),
            "fade_process": ProcessMode.E6,
            "fade_profile": "Profile A",
        }
    )
    engine.process(img, settings, source_hash="file1")
    assert engine.cache.base is not None
    base_data = engine.cache.base.data

    settings2 = WorkspaceConfig.from_flat_dict({**settings.to_dict(), "fade_profile": "Profile B"})
    engine.process(img, settings2, source_hash="file1")
    assert engine.cache.base is not None
    assert engine.cache.base.data is base_data
