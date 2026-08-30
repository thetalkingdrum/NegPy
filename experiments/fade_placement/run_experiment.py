"""EXPERIMENT_PLAN_placement.md: three placement variants, one shared synthetic
fixture set, measured against known ground truth.

Not part of the app or the test suite -- a throwaway prototype script, per the plan's
own framing. Run with: uv run python experiments/fade_placement/run_experiment.py

## What this tests

Fade Restoration composes `inv(F)` into the crosstalk-unmix slot, upstream of Cast
Removal's own per-channel affine (features/exposure/processor.py). Two open questions
from PLAN_after_cast_removal.md:

  B. A stained slide's density feeds the fade matrix *before* Cast Removal's offset
     cleans it up: physically it should be `F^-1(D - o)`, not `F^-1(D) - o`. Does the
     order actually matter?
  C. Cast Removal currently meters the *fade-corrected* film, so changing fade changes
     what Cast Removal fits. Is that the right domain for it to operate in, or should
     fade move downstream, next to the affine?

## Method

A synthetic "clean slide" (smooth low-frequency scene content, not a flat wedge) is
faded through the real forward model (`F = S @ diag(1, ratio_g, ratio_b)`, `S` from a
shipped delta) plus a known additive per-channel stain offset in img_log space. Each
variant restores it and gets a *best-possible* affine fit directly against the true
clean reference (least squares, not a neutral-pixel heuristic) -- this isolates the
placement question from neutral-detector quality, which is a separate concern already
covered by the production Cast Removal tests.

Recovery error is measured in density (|img_log_restored - img_log_clean|) after each
variant's full pipeline (matrix + affine, in whichever order that variant specifies).
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
    density (not to a hypothetical concentration starting point): `clean_slide()` is
    already "a scan of unfaded film", i.e. measured density with S's mixing baked in, the
    same way a real scan would be. This is the similarity-transform form
    `_fade_forward_matrix` in normalization.py uses, not the plan's literal
    `S @ diag(1, ag, ab)` -- that form is S itself (not the identity) at ratio_g ==
    ratio_b == 1.0, which would fail the control fixture's "passes through unchanged"
    requirement. Checked directly: this form is exactly the identity at ratio 1, any S."""
    s_matrix = side_absorption_matrix(delta)
    return s_matrix @ np.diag([1.0, ratio_g, ratio_b]) @ np.linalg.inv(s_matrix)


# --------------------------------------------------------------------------- fixtures


def clean_slide(size: int = 256, seed: int = 0) -> np.ndarray:
    """A synthetic 'unfaded slide scan': a full-range luma gradient (so both the
    detector-quality bands and this script's own affine fit have midtone/shadow content
    to work with) plus smooth, low-frequency per-channel chroma variation standing in
    for scene content -- not a flat gray wedge. Returns linear RGB in (0, 1]."""
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


# --------------------------------------------------------------------------- variants


def fit_affine_to_reference(restored_log: np.ndarray, reference_log: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel least-squares (gain, offset) landing restored_log on reference_log --
    the best possible affine, an upper bound on what a real neutral-detector-driven Cast
    Removal could achieve. Isolates the placement question from detector quality."""
    gains, offsets = [], []
    for ch in range(3):
        x = restored_log[..., ch].ravel()
        y = reference_log[..., ch].ravel()
        design = np.stack([x, np.ones_like(x)], axis=1)
        gain, offset = np.linalg.lstsq(design, y, rcond=None)[0]
        gains.append(gain)
        offsets.append(offset)
    return np.array(gains), np.array(offsets)


def apply_affine(img_log: np.ndarray, gains: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return img_log * gains + offsets


@dataclass
class VariantResult:
    restored_log: np.ndarray
    affine_gains: np.ndarray
    affine_offsets: np.ndarray


def variant_a_baseline(fx: Fixture) -> VariantResult:
    """Current placement: fade matrix first, no offset handling, affine fits whatever's
    left. F^-1(D), then affine(that)."""
    forward = forward_fade_matrix(fx.ratio_g, fx.ratio_b, fx.delta)
    matrix_restored = fx.faded_log @ np.linalg.inv(forward).T
    gains, offsets = fit_affine_to_reference(matrix_restored, np.log10(fx.clean))
    return VariantResult(apply_affine(matrix_restored, gains, offsets), gains, offsets)


def variant_b_upstream_offset(fx: Fixture) -> VariantResult:
    """Subtract the (assumed known) stain before the matrix: F^-1(D - o), then affine
    fits the remaining residual. Best case for the offset -- tests whether the ordering
    itself matters, not whether the offset can be estimated."""
    forward = forward_fade_matrix(fx.ratio_g, fx.ratio_b, fx.delta)
    de_stained = fx.faded_log - np.asarray(fx.stain)
    matrix_restored = de_stained @ np.linalg.inv(forward).T
    gains, offsets = fit_affine_to_reference(matrix_restored, np.log10(fx.clean))
    return VariantResult(apply_affine(matrix_restored, gains, offsets), gains, offsets)


def variant_c_downstream(fx: Fixture) -> VariantResult:
    """Fade matrix moved next to the affine: affine fits the *raw faded* capture first
    (breaking the coupling where Cast Removal meters the fade-corrected film), then the
    matrix restores what's left. affine(D), then F^-1(that)."""
    gains, offsets = fit_affine_to_reference(fx.faded_log, np.log10(fx.clean))
    pre_corrected = apply_affine(fx.faded_log, gains, offsets)
    forward = forward_fade_matrix(fx.ratio_g, fx.ratio_b, fx.delta)
    restored_log = pre_corrected @ np.linalg.inv(forward).T
    return VariantResult(restored_log, gains, offsets)


VARIANTS = {
    "A baseline": variant_a_baseline,
    "B upstream offset": variant_b_upstream_offset,
    "C fade downstream": variant_c_downstream,
}


# --------------------------------------------------------------------------- measure


def recovery_error(result: VariantResult, fx: Fixture) -> tuple[float, float]:
    diff = np.abs(result.restored_log - np.log10(fx.clean))
    return float(diff.max()), float(diff.mean())


def affine_movement(result: VariantResult) -> float:
    """How far the fitted affine had to move from the identity -- how much work landed
    in the 'cheap' stage vs the physically motivated matrix."""
    gain_dev = np.abs(result.affine_gains - 1.0).max()
    offset_dev = np.abs(result.affine_offsets).max()
    return float(gain_dev + offset_dev)


def main() -> None:
    fixtures = fixture_set()
    rows = []
    for fx in fixtures:
        for variant_name, variant_fn in VARIANTS.items():
            result = variant_fn(fx)
            max_err, mean_err = recovery_error(result, fx)
            movement = affine_movement(result)
            rows.append((fx.name, variant_name, max_err, mean_err, movement))

    header = f"{'fixture':<32} {'variant':<20} {'max err':>10} {'mean err':>10} {'affine moved':>13}"
    print(header)
    print("-" * len(header))
    for fixture_name, variant_name, max_err, mean_err, movement in rows:
        print(f"{fixture_name:<32} {variant_name:<20} {max_err:>10.5f} {mean_err:>10.5f} {movement:>13.5f}")

    print()
    print("Control fixture (no fade, no stain) sanity check -- every variant should be ~0:")
    for fixture_name, variant_name, max_err, mean_err, _m in rows:
        if fixture_name.startswith("control"):
            status = "OK" if max_err < 1e-6 else "FAIL -- variant does not pass the control through unchanged"
            print(f"  {variant_name:<20} max_err={max_err:.2e}  {status}")


if __name__ == "__main__":
    main()
