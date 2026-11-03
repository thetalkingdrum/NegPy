"""Phase 4 numbers on synthetic films: the cold lean of the gray finders, tone strength and Adaptive, and fog
diagnostics.

uv run python experiments/fade_crossover/phase4.py [neutrals] [tone] [fog]
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

from negpy.features.process.fade import EstimateMethod, nc_neutral_refs  # noqa: E402

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

    if finder in ("negpy", "no_sky", "nc"):
        return lambda g: gray_refs(g, finder)
    if isinstance(finder, float):
        return lambda g: gray_refs(g, "negpy", finder)

    def read(g):
        refs = find(g, finder)
        if refs is None:
            return None, "none"
        return (tuple(refs[0]), tuple(refs[1]), None if refs[2] is None else tuple(refs[2]), 1.0), ""

    return read


READERS = (
    (("NegPy", "negpy"),)
    + tuple((n, f) for n, f in FINDERS[1:])
    + (("No Sky (sky sector off)", "no_sky"), ("NegPy, Mix 50 %", 0.5), ("NC", "nc"))
)
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
    return estimate_fade_with(estimate_grid(image, **fields), rd, white_space=space)[EstimateMethod.MIDTONE], fields


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


def render_tone(image: np.ndarray, changes: dict, tone: str, strength: float = 1.0) -> tuple[np.ndarray, dict]:
    """(render, sliders): the render with Tone *tone* at *strength*, read from a first render's
    `fade_tone_points` as the panel does."""
    from negpy.domain.interfaces import PipelineContext
    from negpy.domain.models import WorkspaceConfig
    from negpy.features.process.fade_tone import fade_tone_changes
    from negpy.features.process.models import FadeABTone, ProcessMode, mode_aware_exposure_reset
    from negpy.services.rendering.engine import DarkroomEngine

    base = WorkspaceConfig()
    process = replace(base.process, process_mode=ProcessMode.E6, **changes)
    settings = replace(base, process=process, exposure=mode_aware_exposure_reset(ProcessMode.E6, base.exposure))
    ctx = PipelineContext(original_size=image.shape[:2], scale_factor=1.0, process_mode=ProcessMode.E6)
    out = np.asarray(DarkroomEngine().process(image, settings, "fade-tone", ctx))
    if tone == "off":
        return out, {"density": settings.exposure.density, "grade": settings.exposure.grade, "span": _span(ctx.metrics)}
    if tone == "nc":
        settings = replace(settings, process=replace(settings.process, fade_ab_tone="nc"))
    else:
        settings, _ = fade_tone_changes(settings, FadeABTone(tone), ctx.metrics, strength)
    out = np.asarray(DarkroomEngine().process(image, settings, "fade-tone"))
    return out, {"density": settings.exposure.density, "grade": settings.exposure.grade, "span": _span(ctx.metrics)}


def _span(metrics: dict) -> float:
    p = metrics.get("fade_tone_points")
    return float("nan") if p is None else float(p[-1] - p[0])


def tone_report() -> None:
    from fade_crossover_scene import build_scene, scan_image

    print("\n# Tone: the Grade each choice picks\n")
    print("Fresh Estimate (Midtone) at Crossover 100 %, then Tone at Strength 100 %. span is the frame's 0.5 to 99.5 %")
    print("mean-density span after the fade (D). dE is the 9-patch mean against the unfaded film at tone Off,")
    print("dL* the ramp's mean lightness error.\n")
    scene = build_scene()
    fades = [(f"uniform {s:g}", linear_fade(REFERENCE, (s, s, s))) for s in (1.0, 0.8, 0.6, 0.45, 0.3)]
    fades += [("linear 1/.7/.5", linear_fade(REFERENCE, (1.0, 0.7, 0.5))), ("crossover", CROSSOVER)]
    fades += [
        ("low range .6/.45/.35", linear_fade(REFERENCE, (0.6, 0.45, 0.35))),
        ("low range .4/.3/.25", linear_fade(REFERENCE, (0.4, 0.3, 0.25))),
    ]
    for scanner_name, scanner, positive in SCANNERS:
        ref_lab = to_lab(render_tone(scan_image(scanner(REFERENCE), scene), dict(positive_source=True) if positive else {}, "off")[0])
        print(f"\n## {scanner_name}\n")
        print(
            "| fade | span D | NegPy Density / Grade | Adaptive Density / Grade | dE off / NC / NegPy / Adaptive | dL* off / NC / NegPy / Adaptive |"
        )
        print("|---|---|---|---|---|---|")
        for fade_name, film in fades:
            image = scan_image(scanner(film), scene)
            est, fields = estimate_with(image, reader("negpy"), positive)
            changes = survival_changes(est, 1.0, **fields) if est.red is not None else dict(fields)
            row, de, dl = {}, [], []
            for tone in ("off", "nc", "negpy", "adaptive"):
                out, sliders = render_tone(image, changes, tone)
                row[tone] = sliders
                lab = to_lab(out)
                de.append(float(np.mean([np.linalg.norm(region_lab(lab, m) - region_lab(ref_lab, m)) for m in scene.patches.values()])))
                dl.append(float(np.mean([abs(region_lab(lab, m)[0] - region_lab(ref_lab, m)[0]) for m in scene.ramp])))
            n, a = row["negpy"], row["adaptive"]
            print(
                f"| {fade_name} | {row['off']['span']:.2f} | {n['density']:.2f} / {n['grade']:.0f} | {a['density']:.2f} / {a['grade']:.0f} "
                f"| {' / '.join(f'{v:.1f}' for v in de)} | {' / '.join(f'{v:.1f}' for v in dl)} |"
            )


# --------------------------------------------------------------------------------------------------- fog


def bright_fog_scene(bright: str, fog: str, width: int = 384, height: int = 256, seed: int = 1):
    """(linear scan, true uneven fog (D, zero at its minimum), bright-area mask): a textured picture on
    the unfaded film, Coolscan-like, with a bright picture area and optional real fog.

    bright: "wall", a lit near-white wall over the right third; "sky", a pale sky band over the top
    third; "none". fog: "left", density falling from the left edge (0.12 D across); "strong", the same at
    0.25 D, past the cap; "corner", a bump in
    the top-left; "none", an even 0.03 D only."""
    from fade_crossover_scene import RAMP_RANGE

    y, x = np.mgrid[0:height, 0:width]
    t = ((x * 7 + y * 3) % 40) / 39.0
    log_e = np.repeat((RAMP_RANGE[0] + 0.6 + t * (RAMP_RANGE[1] - RAMP_RANGE[0] - 1.2))[..., None], 3, axis=-1)
    area = np.zeros((height, width), bool)
    if bright == "wall":
        area = x > width * 0.66
        log_e[area] = (0.7, 0.7, 0.68)
    elif bright == "sky":
        area = y < height * 0.33
        log_e[area] = (0.45, 0.55, 0.75)
    density = -np.log10(REFERENCE.scan(log_e))
    if fog == "left":
        uneven = 0.12 * (1.0 - x / (width - 1))
    elif fog == "strong":
        uneven = 0.25 * (1.0 - x / (width - 1))
    elif fog == "corner":
        uneven = 0.14 * np.exp(-((x / (0.3 * width)) ** 2 + (y / (0.3 * width)) ** 2))
    else:
        uneven = np.zeros((height, width))
    grain = np.random.default_rng(seed).normal(0.0, 0.004, density.shape)
    scan = np.clip(np.power(10.0, -(density + 0.03 + uneven[..., None] + grain)), 1e-6, 1.0).astype(np.float32)
    return scan, uneven - uneven.min(), area


FOG_VARIANTS = (
    ("default", {}),
    ("plausible 0.20", dict(plausible=0.20)),
    ("plausible 0.14", dict(plausible=0.14)),
    ("clear band 0.14", dict(clear_band=0.14)),
    ("clear band 0.10", dict(clear_band=0.10)),
    ("min cells 64", dict(min_cells=64)),
    ("min cells 160", dict(min_cells=160)),
    ("clear both sides", dict(both_sides=True)),
)
FOG_SCENES = tuple((b, f) for b in ("none", "wall", "sky") for f in ("none", "left", "corner", "strong"))


def fog_report() -> None:
    from negpy.features.process.fade_fog import FOG_CAP, fit_fog, fog_map

    print("\n# Fog diagnostics\n")
    print("amp is the fitted surface's range before the 0.15 D cap; kept the share of grid cells the fit used.")
    print("err is the RMS (D) of the removed map against the true uneven fog over the frame; bright the mean")
    print("removed over the bright area minus the mean over the rest (true value in brackets); 0 when nothing is removed.\n")
    print("| bright area | fog | variant | amp D | cap hit | kept | removed max D | err D | bright D |")
    print("|---|---|---|---|---|---|---|---|---|")
    summary: dict = {}
    for bright, fog in FOG_SCENES:
        scan, truth, area = bright_fog_scene(bright, fog)
        true_bias = float(truth[area].mean() - truth[~area].mean()) if area.any() else 0.0
        for name, kw in FOG_VARIANTS:
            fit = fit_fog(scan, **kw)
            removed = np.zeros_like(truth) if fit.surface is None else fog_map(scan, fit.surface)
            err = float(np.sqrt(np.mean((removed - np.minimum(truth, FOG_CAP)) ** 2)))
            bias = float(removed[area].mean() - removed[~area].mean()) if area.any() else 0.0
            amp = "" if fit.amplitude is None else f"{fit.amplitude:.3f}"
            summary.setdefault(name, []).append((fog != "none", err, abs(bias - true_bias) if area.any() else 0.0))
            print(
                f"| {bright} | {fog} | {name} | {amp} | {'yes' if fit.cap_hit else ''} | {fit.kept_share:.2f} | {removed.max():.3f} "
                f"| {err:.3f} | {bias:+.3f} ({true_bias:+.3f}) |"
            )
    print("\n| variant | mean err, fog scenes | mean err, no-fog scenes | mean bright-area error |")
    print("|---|---|---|---|")
    for name, rows in summary.items():
        fogged = [e for f, e, _ in rows if f]
        clean = [e for f, e, _ in rows if not f]
        print(f"| {name} | {np.mean(fogged):.3f} | {np.mean(clean):.3f} | {np.mean([b for *_, b in rows]):.3f} |")


# --------------------------------------------------------------------------------------------------- main


def main(argv: list[str]) -> None:
    parts = set(argv) or {"neutrals", "tone", "fog"}
    if "neutrals" in parts:
        neutrals_report()
    if "tone" in parts:
        tone_report()
    if "fog" in parts:
        fog_report()


if __name__ == "__main__":
    main(sys.argv[1:])
