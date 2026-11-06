"""Phase 4 numbers on synthetic films: the cold lean of the gray finders. Its tone and fog parts measured
choices the app no longer has; RESULTS.md keeps their numbers.

uv run python experiments/fade_crossover/phase4.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fade_crossover_scene import (  # noqa: E402
    CROSSOVER,
    REFERENCE,
    estimate_grid,
    plustek,
    linear_fade,
    region_lab,
    render,
    scan_warm,
    survival_changes,
    to_lab,
    warm_scene,
)
from neutral_finders import Finder, check_replica, find  # noqa: E402

from negpy.features.process.fade import nc_neutral_refs  # noqa: E402

# ---------------------------------------------------------------------------------------------- neutrals

_scan = scan_warm


def lab_of_log(log: np.ndarray) -> np.ndarray:
    """CIELAB of a linear transmittance 10^log, as a neutral in the unfaded film would show it."""
    from negpy.kernel.image.logic import rgb_to_lab_working

    rgb = np.power(10.0, np.asarray(log, dtype=np.float64)).astype(np.float32).reshape(1, 1, 3)
    return rgb_to_lab_working(rgb).reshape(3)


FINDERS = (
    ("NegPy", Finder()),
    ("no subject sectors", Finder(subjects=False)),
    ("no chroma caps", Finder(caps=False)),
    ("raw-log chroma", Finder(raw=True)),
    ("soft weight 0.25 D", Finder(soft=0.25)),
    ("soft weight 0.12 D", Finder(soft=0.12)),
    ("8 bands", Finder(bands=8)),
    ("soft 0.25 D, 8 bands", Finder(soft=0.25, bands=8)),
)


LAYOUTS = (("wall", 0.0), ("wall", 1.0), ("wall", 2.0), ("sun", 0.0), ("sun", 1.0), ("sun", 2.0), ("bright", 1.0), ("sky", 1.0))


def nc_refs(grid: np.ndarray):
    refs, _ = nc_neutral_refs(grid)
    return None if refs is None else (np.asarray(refs[0]), np.asarray(refs[1]), None if refs[2] is None else np.asarray(refs[2]))


def reader(finder):
    """An `estimate_fade_with` gray reader: NegPy's own (with its fallbacks), NC's, or a `Finder`."""
    from negpy.features.process.fade import gray_refs

    if finder in ("negpy", "nc"):
        return lambda g: gray_refs(g, 1.0 if finder == "nc" else 0.0)
    if isinstance(finder, float):
        return lambda g: gray_refs(g, finder)

    def read(g):
        refs = find(g, finder)
        if refs is None:
            return None, "none"
        return (tuple(refs[0]), tuple(refs[1]), None if refs[2] is None else tuple(refs[2]), 1.0), ""

    return read


READERS = (("NegPy", "negpy"),) + tuple((n, f) for n, f in FINDERS[1:]) + (("Neutrals 50 %", 0.5), ("NC", "nc"))
SCANNERS = (("Coolscan-like", lambda film: film, False), ("Plustek-like", plustek, True))


def ramp_ab(lab: np.ndarray, scene) -> tuple[float, float]:
    """Mean a* and b* of the ramp's midtone steps."""
    v = np.array([region_lab(lab, m) for m in scene.ramp[3:13]])
    return float(v[:, 1].mean()), float(v[:, 2].mean())


def estimate_with(image: np.ndarray, rd, positive: bool):
    from negpy.features.process.fade import estimate_fade_with, render_white_space
    from negpy.features.process.models import ProcessConfig, ProcessMode

    fields = dict(positive_source=positive) if positive else {}
    space = render_white_space(replace(ProcessConfig(), process_mode=ProcessMode.E6, **fields), ProcessMode.E6, None, None)
    return estimate_fade_with(estimate_grid(image, **fields), rd, white_space=space), fields


def neutrals_report() -> None:
    print("# Cold lean of the gray finders\n")
    print("Each cell is the render's mean b* over the true gray ramp's midtone steps after Estimate (Midtone)")
    print("with that gray reader: negative is cold. x is Crossover. Layouts: wall/sun/bright, warmth 0 to 2.\n")
    fades = (("linear 1/.75/.55", linear_fade(REFERENCE, (1.0, 0.75, 0.55))), ("crossover", CROSSOVER))
    summary: dict = {}
    for scanner_name, scanner, positive in SCANNERS:
        for fade_name, film in fades:
            print(f"\n## {scanner_name}, {fade_name}\n")
            head = " | ".join(f"{lay} {w:g}" for lay, w in LAYOUTS)
            print(f"| reader | x | {head} | mean |")
            print("|---|---|" + "---|" * len(LAYOUTS) + "---|")
            rows: dict = {}
            for layout, warmth in LAYOUTS:
                scene = warm_scene(warmth, layout=layout)
                image = _scan(scanner(film), scene)
                if not positive:
                    check_replica(estimate_grid(image))
                for name, f in READERS:
                    est, fields = estimate_with(image, reader(f), positive)
                    for x in (0.0, 1.0):
                        if est.red is None:
                            rows.setdefault((name, x), []).append(np.nan)
                            continue
                        lab = to_lab(render(image, survival_changes(est, x, **fields)))
                        rows.setdefault((name, x), []).append(ramp_ab(lab, scene)[1])
            for (name, x), vals in rows.items():
                summary.setdefault((name, x), []).extend(vals)
                cells = " | ".join("none" if np.isnan(v) else f"{v:+.1f}" for v in vals)
                print(f"| {name} | {x:g} | {cells} | {np.nanmean(vals):+.2f} |")
    print("\n## All fixtures: mean and mean absolute ramp b*\n")
    print("| reader | x | mean b* | mean abs b* | shift from NegPy |")
    print("|---|---|---|---|---|")
    for (name, x), vals in summary.items():
        base = np.nanmean(summary[("NegPy", x)])
        print(f"| {name} | {x:g} | {np.nanmean(vals):+.2f} | {np.nanmean(np.abs(vals)):.2f} | {np.nanmean(vals) - base:+.2f} |")
    band_detail()


def band_detail() -> None:
    """Each band centre of both readers on one fixture, with its b* in the unfaded film."""
    from negpy.features.process.fade import neutral_bands

    s = (1.0, 0.75, 0.55)
    for layout, warmth in (("wall", 1.0), ("bright", 1.0)):
        scene = warm_scene(warmth, layout=layout)
        grid = estimate_grid(_scan(linear_fade(REFERENCE, s), scene))
        print(f"\n## Band centres, Coolscan-like, linear 1/.75/.55, {layout} warmth {warmth:g}\n")
        print("b* in the unfaded film (+ warm). NC's mid is the share-weighted bands 3 and 4.\n")
        print("| reader | band | mean D | share | b* |")
        print("|---|---|---|---|---|")
        refs = find(grid)
        for label, r in zip(("mid", "shadow", "highlight"), refs):
            if r is not None:
                print(f"| NegPy | {label} | {-np.mean(r):.2f} | | {lab_of_log(np.asarray(r) / np.asarray(s))[2]:+.2f} |")
        bands = neutral_bands(grid)
        for c, lv, sh, k in zip(bands.centre, bands.level, bands.share, bands.index):
            print(f"| NC | {k} | {-lv:.2f} | {sh:.3f} | {lab_of_log(c / np.asarray(s))[2]:+.2f} |")


# -------------------------------------------------------------------------------------------------- tone


# --------------------------------------------------------------------------------------------------- main


def main(argv: list[str]) -> None:
    neutrals_report()


if __name__ == "__main__":
    main(sys.argv[1:])
