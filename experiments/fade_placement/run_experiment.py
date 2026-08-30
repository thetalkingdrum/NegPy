"""EXPERIMENT_PLAN_placement.md: three placement variants, one shared synthetic
fixture set, measured against known ground truth.

Not part of the app -- a throwaway prototype script, per the plan's own framing. The
fixture generator it depends on lives in tests/test_fade_placement_fixtures.py instead,
since it is real pipeline-level validation and this variant-comparison scaffolding is
not. Run with: uv run python experiments/fade_placement/run_experiment.py

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

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.test_fade_placement_fixtures import (  # noqa: E402
    Fixture,
    fixture_set,
    forward_fade_matrix,
)

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
