"""Phase 3 numbers on the synthetic films of tests/fade_crossover_scene.py: scanner zero, tone and uneven fog.

uv run python experiments/fade_crossover/phase3.py [zero] [tone] [fog]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

from fade_crossover_scene import (  # noqa: E402
    CROSSOVER,
    REFERENCE,
    build_scene,
    coolscan,
    estimate,
    linear_fade,
    plustek,
    ramp_chroma,
    region_lab,
    render,
    scan_image,
    survival_changes,
    to_lab,
)
from negpy.features.process.fade import crossover_bands, crossover_lean_fit as lean_fit  # noqa: E402

ZEROS = ("full_scale", "clear_film", "clear_neutral")
FADES = (("crossover fade", CROSSOVER), ("linear fade 1.0/0.7/0.5", linear_fade(REFERENCE, (1.0, 0.7, 0.5))))
SCANNERS = (("Plustek-like", plustek), ("Coolscan-like", coolscan))


def scores(lab: np.ndarray, ref_lab: np.ndarray, scene) -> tuple[float, float, float, float]:
    ramp = ramp_chroma(lab, scene)
    de = [float(np.linalg.norm(region_lab(lab, m) - region_lab(ref_lab, m))) for m in scene.patches.values()]
    return max(a for a, _ in ramp), float(np.mean([b for _, b in ramp])), float(np.mean(de)), max(de)


def zero_report() -> None:
    scene = build_scene()
    print("# Scanner zero\n")
    for scanner_name, scanner in SCANNERS:
        ref_lab = to_lab(render(scan_image(scanner(REFERENCE), scene), dict(positive_source=True)))
        for fade_name, film in FADES:
            image = scan_image(scanner(film), scene)
            print(f"\n## {scanner_name}, {fade_name}\n")
            print("| zero | survivals r/g/b | lean slope r/g/b | residual | x | ramp C* max | ramp C* mean | patch dE mean | patch dE max |")
            print("|---|---|---|---|---|---|---|---|---|")
            tables = {}
            for zero in ZEROS:
                fields = dict(positive_source=True, fade_ab_scanner_zero=zero)
                est = estimate(image, **fields)
                tables[zero] = est.crossover
                if est.crossover is not None:
                    slope, residual = lean_fit(est.crossover)
                    lean = "/".join(f"{v:+.3f}" for v in slope) + f" | {residual:.3f}"
                else:
                    lean = "none | none"
                surv = f"{est.red or 1.0:.2f}/{est.green:.2f}/{est.blue:.2f}"
                for x in (0.0, 1.0):
                    s = scores(to_lab(render(image, survival_changes(est, x, **fields))), ref_lab, scene)
                    print(f"| {zero} | {surv} | {lean} | {x:g} | " + " | ".join(f"{v:.1f}" for v in s) + " |")
            for zero, table in tables.items():
                if table is None:
                    continue
                print(f"\n{zero} bands (mean log, share, lean r, g, b):\n")
                for row in crossover_bands(table):
                    print("    " + "  ".join(f"{v:+.3f}" for v in row))


def main(argv: list[str]) -> None:
    parts = set(argv) or {"zero", "tone", "fog"}
    if "zero" in parts:
        zero_report()
    if "tone" in parts:
        tone_report()
    if "fog" in parts:
        fog_report()


def tone_report() -> None:
    pass


def fog_report() -> None:
    pass


if __name__ == "__main__":
    main(sys.argv[1:])
