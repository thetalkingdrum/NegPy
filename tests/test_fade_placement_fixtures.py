"""Synthetic fade fixtures: a forward model and a scene, independent of the app's own
`_fade_forward_matrix` (features/exposure/normalization.py), used to validate that
matrix against a from-scratch construction and to build test images with known ground
truth. The only pipeline-level validation of the fade math that exists outside the app's
own unit tests -- kept here rather than in a throwaway experiment script so it survives
whatever else changes around it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GENERIC_E6_DELTA = (0.0689, 0.0111, 0.2246, 0.0486, 0.0854, 0.1815)  # (gr, br, rg, bg, rb, gb)
KODACHROME_DELTA = (0.0770, 0.0004, 0.0558, 0.0260, 0.0393, 0.2278)  # a differently-shaped dye set


def side_absorption_matrix(delta: tuple) -> np.ndarray:
    d_gr, d_br, d_rg, d_bg, d_rb, d_gb = delta
    return np.array([[1.0, d_gr, d_br], [d_rg, 1.0, d_bg], [d_rb, d_gb, 1.0]])


def forward_fade_matrix(ratio_g: float, ratio_b: float, delta: tuple) -> np.ndarray:
    """F = S @ diag(1, ratio_g, ratio_b) @ inv(S), applied directly to a *scan's* measured
    density (not a hypothetical concentration starting point): `clean_slide()` is already
    "a scan of unfaded film", i.e. measured density with S's mixing baked in, the same way
    a real scan would be. This is the similarity-transform form `_fade_forward_matrix` in
    normalization.py uses, not the naive `S @ diag(1, ag, ab)` -- that form equals S itself
    (not the identity) at ratio_g == ratio_b == 1.0, which fails "a control passes through
    unchanged". This form is exactly the identity at ratio 1, for any S."""
    s_matrix = side_absorption_matrix(delta)
    return s_matrix @ np.diag([1.0, ratio_g, ratio_b]) @ np.linalg.inv(s_matrix)


def clean_slide(size: int = 256, seed: int = 0) -> np.ndarray:
    """A synthetic 'unfaded slide scan': a full-range luma gradient (so both the
    detector-quality bands and an affine fit have midtone/shadow content to work with)
    plus smooth, low-frequency per-channel chroma variation standing in for scene
    content -- not a flat gray wedge. Returns linear RGB in (0, 1]."""
    yy, xx = (np.mgrid[0:size, 0:size].astype(np.float64)) / size
    t = xx  # bright (left) to dark (right), full E-6 transfer range

    def field(seed_offset: int) -> np.ndarray:
        rng2 = np.random.default_rng(seed + seed_offset)
        f = np.zeros((size, size))
        for _ in range(4):
            fx, fy = rng2.uniform(0.5, 3.0, 2)
            phase = rng2.uniform(0, 2 * np.pi)
            amp = rng2.uniform(0.3, 1.0) / 4
            f += amp * np.sin(2 * np.pi * (fx * xx + fy * yy) + phase)
        return f

    span = 2.9  # TRANSFER_DENSITY_RANGE
    chroma_amp = 0.08
    log_r = -(t * span) + chroma_amp * field(1)
    log_g = -(t * span) + chroma_amp * field(2)
    log_b = -(t * span) + chroma_amp * field(3)
    image = np.stack([10.0**log_r, 10.0**log_g, 10.0**log_b], axis=-1)
    return np.clip(image, 1e-6, 1.0).astype(np.float64)


@dataclass
class Fixture:
    name: str
    ratio_g: float
    ratio_b: float
    delta: tuple
    stain: tuple  # additive offset in img_log units, applied after the matrix
    clean: np.ndarray
    faded_log: np.ndarray


def make_fixture(name: str, ratio_g: float, ratio_b: float, delta: tuple, stain: tuple, seed: int = 0) -> Fixture:
    clean = clean_slide(seed=seed)
    clean_log = np.log10(clean)
    forward = forward_fade_matrix(ratio_g, ratio_b, delta)
    faded_log = clean_log @ forward.T + np.asarray(stain)
    return Fixture(name=name, ratio_g=ratio_g, ratio_b=ratio_b, delta=delta, stain=stain, clean=clean, faded_log=faded_log)


def fixture_set() -> list[Fixture]:
    z = (0.0, 0.0, 0.0)
    return [
        make_fixture("control (no fade, no stain)", 1.0, 1.0, GENERIC_E6_DELTA, z, seed=0),
        make_fixture("mild fade, no stain", 0.85, 0.9, GENERIC_E6_DELTA, z, seed=1),
        make_fixture("heavy fade, no stain", 0.35, 0.5, GENERIC_E6_DELTA, z, seed=2),
        make_fixture("mild fade, heavy stain", 0.85, 0.9, GENERIC_E6_DELTA, (0.15, 0.05, -0.10), seed=3),
        make_fixture("differential fade, with stain", 0.4, 0.95, KODACHROME_DELTA, (0.08, -0.03, 0.06), seed=4),
    ]


# --------------------------------------------------------------------------- invariants


def test_forward_matrix_is_identity_at_unity_ratios():
    for delta in (GENERIC_E6_DELTA, KODACHROME_DELTA, (0.3, 0.1, 0.2, 0.15, 0.25, 0.05)):
        forward = forward_fade_matrix(1.0, 1.0, delta)
        assert np.allclose(forward, np.eye(3), atol=1e-12)


def test_control_fixture_matches_clean_reference():
    control = fixture_set()[0]
    assert np.allclose(control.faded_log, np.log10(control.clean), atol=1e-12)


def test_forward_matrix_matches_the_production_similarity_transform():
    from negpy.features.exposure.normalization import _fade_forward_matrix

    for ratio_g, ratio_b in ((0.85, 0.9), (0.35, 0.5), (0.4, 0.95)):
        expected = _fade_forward_matrix(1.0, ratio_g, ratio_b, GENERIC_E6_DELTA)
        assert expected is not None
        actual = forward_fade_matrix(ratio_g, ratio_b, GENERIC_E6_DELTA)
        assert np.allclose(actual, expected, atol=1e-10)


def test_fixture_set_covers_stained_and_unstained_cases():
    names = [fx.name for fx in fixture_set()]
    assert any("stain" in n and "no stain" not in n for n in names)
    assert any("no stain" in n for n in names)
