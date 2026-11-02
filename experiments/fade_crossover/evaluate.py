"""Print the crossover table's numbers on the synthetic films of tests/fade_crossover_scene.py.

uv run python experiments/fade_crossover/evaluate.py
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
    chroma,
    estimate,
    hue,
    linear_fade,
    ramp_chroma,
    region_lab,
    render,
    scan_image,
    survival_changes,
    to_lab,
)
from negpy.features.process.fade import crossover_bands  # noqa: E402

STRENGTHS = (0.0, 0.5, 1.0)


def _hue_delta(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def report(title: str, film, **scene_kw) -> None:
    scene = build_scene(**scene_kw)
    image = scan_image(film, scene)
    est = estimate(image)
    print(f"\n## {title}\n")
    print(f"Estimate: red {est.red}, green {est.green:.3f}, blue {est.blue:.3f} {est.reason!r}\n")
    if est.crossover is not None:
        print("| band mean log | share | lean R | lean G | lean B |\n|---|---|---|---|---|")
        for row in crossover_bands(est.crossover):
            print("| " + " | ".join(f"{v:+.3f}" for v in row) + " |")
        print()
    ref_lab = to_lab(render(scan_image(REFERENCE, scene), {}))
    labs = {x: to_lab(render(image, survival_changes(est, x))) for x in STRENGTHS}
    print("Neutral ramp C* per band, max / mean, darkest first:\n")
    print("| crossover | " + " | ".join(f"b{i}" for i in range(8)) + " |\n|---|" + "---|" * 8)
    for x, lab in labs.items():
        print(f"| {x} | " + " | ".join(f"{a:.1f} / {b:.1f}" for a, b in ramp_chroma(lab, scene)) + " |")
    print("\nPatches: C* and hue (deg) at each strength; drift is against crossover 0 and the unfaded film:\n")
    print("| patch | ref C*/h | x=0 C*/h | x=0.5 C*/h | x=1 C*/h | dC*(1-0) | dh(1-0) | dC*(1-ref) | dh(1-ref) |")
    print("|---|---|---|---|---|---|---|---|---|")
    regions = dict(scene.patches)
    if scene.subject is not None:
        regions["red subject"] = scene.subject
    if scene.groups is not None:
        regions["group big (magenta)"], regions["group small (green)"] = scene.groups
    for name, mask in regions.items():
        ref = region_lab(ref_lab, mask)
        vals = [region_lab(labs[x], mask) for x in STRENGTHS]
        cells = [f"{chroma(v):.1f} / {hue(v):.0f}" for v in (ref, *vals)]
        d0 = (chroma(vals[2]) - chroma(vals[0]), _hue_delta(hue(vals[2]), hue(vals[0])))
        dr = (chroma(vals[2]) - chroma(ref), _hue_delta(hue(vals[2]), hue(ref)))
        print(f"| {name} | " + " | ".join(cells) + f" | {d0[0]:+.1f} | {d0[1]:+.0f} | {dr[0]:+.1f} | {dr[1]:+.0f} |")


def main() -> None:
    report("Crossover fade (per-layer curves)", CROSSOVER)
    report("Crossover fade with a large red subject", CROSSOVER, subject=True)
    report("Linear fade only (survivals 1.0 / 0.7 / 0.5)", linear_fade(REFERENCE, (1.0, 0.7, 0.5)))
    report("Two opposite groups at one exposure, on the unfaded film", REFERENCE, opposite_groups=True)
    np.set_printoptions(precision=3)


if __name__ == "__main__":
    main()
