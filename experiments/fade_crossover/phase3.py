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


TONES = ("off", "nc", "negpy")


def render_tone(image: np.ndarray, process_changes: dict, tone: str) -> tuple[np.ndarray, dict]:
    """(render, tone sliders): the render with Tone *tone*. NegPy reads the first render's
    `fade_tone_points` and writes Print Density and ISO-R Grade, as the panel does."""
    from dataclasses import replace

    from negpy.domain.interfaces import PipelineContext
    from negpy.domain.models import WorkspaceConfig
    from negpy.features.process.fade_tone import fade_tone_changes
    from negpy.features.process.models import FadeABTone, ProcessMode, mode_aware_exposure_reset
    from negpy.services.rendering.engine import DarkroomEngine

    base = WorkspaceConfig()
    process = replace(base.process, process_mode=ProcessMode.E6, **process_changes)
    settings = replace(base, process=process, exposure=mode_aware_exposure_reset(ProcessMode.E6, base.exposure))
    ctx = PipelineContext(original_size=image.shape[:2], scale_factor=1.0, process_mode=ProcessMode.E6)
    out = np.asarray(DarkroomEngine().process(image, settings, "fade-tone", ctx))
    if tone == "nc":
        settings = replace(settings, process=replace(settings.process, fade_ab_tone="nc"))
    elif tone == "negpy":
        settings, _ = fade_tone_changes(settings, FadeABTone.NEGPY, ctx.metrics)
    else:
        return out, {}
    out = np.asarray(DarkroomEngine().process(image, settings, "fade-tone"))
    return out, {"density": settings.exposure.density, "grade": settings.exposure.grade}


def tone_report() -> None:
    print("\n# Tone\n")
    scene = build_scene()
    models = (("NegPy, Crossover 0", dict(fade_ab_model="negpy"), 0.0), ("NegPy, Crossover 100 %", dict(fade_ab_model="negpy"), 1.0), ("NC", dict(fade_ab_model="nc"), 1.0))
    low_range = ("low-range linear fade 0.6/0.45/0.35", linear_fade(REFERENCE, (0.6, 0.45, 0.35)))
    for fade_name, film in (*FADES, low_range):
        image = scan_image(film, scene)
        ref_lab = to_lab(render(scan_image(REFERENCE, scene), {}))
        est = estimate(image)
        print(f"\n## {fade_name} (camera RAW slide path)\n")
        print("| color model | tone | Print Density / ISO-R Grade | ramp C* max | ramp C* mean | patch dE mean | patch dE max | ramp dL* mean |")
        print("|---|---|---|---|---|---|---|---|")
        for model_name, fields, x in models:
            for tone in TONES:
                out, sliders = render_tone(image, {**survival_changes(est, x), **fields}, tone)
                lab = to_lab(out)
                ramp_dl = float(np.mean([abs(region_lab(lab, m)[0] - region_lab(ref_lab, m)[0]) for m in scene.ramp]))
                s = scores(lab, ref_lab, scene)
                slid = f"{sliders['density']:.2f} / {sliders['grade']:.0f}" if sliders else ""
                print(f"| {model_name} | {tone} | {slid} | " + " | ".join(f"{v:.1f}" for v in s) + f" | {ramp_dl:.1f} |")


FOGS = ("off", "nc", "negpy")


def render_fog(image: np.ndarray, fog: str, **fields) -> np.ndarray:
    """A scanner slide's render with Uneven fog *fog*: the NegPy bake runs on the source first, as
    `run_pipeline` runs it, and the NC stage inside the engine."""
    from dataclasses import replace

    from negpy.features.process.fade_fog import bake_fog
    from negpy.features.process.models import ProcessConfig, ProcessMode

    changes = dict(positive_source=True, fade_ab_fog=fog, **fields)
    process = replace(ProcessConfig(), process_mode=ProcessMode.E6, **changes)
    return render(bake_fog(image, process), changes)


def _blocks(field: np.ndarray, mask: np.ndarray, size: int = 16) -> np.ndarray:
    """Means of *field* over size x size blocks wholly inside *mask*."""
    h, w = mask.shape
    out = []
    for y in range(0, h - size + 1, size):
        for x in range(0, w - size + 1, size):
            if mask[y : y + size, x : x + size].all():
                out.append(field[y : y + size, x : x + size].reshape(-1, field.shape[-1] if field.ndim == 3 else 1).mean(axis=0))
    return np.asarray(out)


def fog_scores(lab: np.ndarray, clean_lab: np.ndarray, scene) -> dict:
    gray = scene.gray
    h, w = gray.shape
    dl = lab[..., 0] - clean_lab[..., 0]
    top = gray.copy()
    top[int(h * 0.55) :] = False
    bands = [float(dl[:, i0:i1][top[:, i0:i1]].mean()) for i0, i1 in zip(np.linspace(0, w, 9).astype(int)[:-1], np.linspace(0, w, 9).astype(int)[1:])]
    corner = gray.copy()
    corner[int(h * 0.25) :, :] = False
    corner[:, : int(w * 0.75)] = False
    mid = gray.copy()
    mid[: int(h * 0.25)] = False
    chroma = np.hypot(lab[..., 1], lab[..., 2]) - np.hypot(clean_lab[..., 1], clean_lab[..., 2])
    blocks = _blocks(np.concatenate([chroma[..., None], lab[..., 1:]], axis=-1), gray)
    worst = blocks[np.argmax(blocks[:, 0])]
    wall = ~gray
    return {
        "dL* bands left to right": " ".join(f"{b:+.1f}" for b in bands),
        "gradient left (dL* max - min)": max(bands) - min(bands),
        "corner dL* vs middle": float(dl[corner].mean() - dl[mid].mean()),
        "wall dL*": float(dl[wall].mean()),
        "max C* added on gray blocks": float(worst[0]),
        "its hue": float(np.degrees(np.arctan2(worst[2], worst[1]))),
    }


def their_fog_scene():
    """Their expiredRescueOpenCv.test.mjs scene and fog model (a lifted floor in the display-encoded
    positive), decoded through the working TRC into a scanner file's linear source."""
    from fade_crossover_scene import FogScene
    from nc_rescue_fixtures import FOG_H, FOG_W, fog_wall_positive
    from negpy.kernel.image.logic import working_oetf_decode

    aged = fog_wall_positive() / 65535.0
    y, x = np.mgrid[0:FOG_H, 0:FOG_W]
    v = np.round(255.0 * ((x * 7 + y * 3) % 40) / 39.0)
    wall = (x > FOG_W * 0.6) & (y > FOG_H * 0.6)
    v = np.where(wall, 220.0, v)
    # The clean scene keeps the fog's even part (0.10 at the right edge), which no stage removes.
    clean = 0.10 + 0.8 * v / 255.0
    decode = lambda d: np.ascontiguousarray(working_oetf_decode(np.repeat(np.asarray(d, np.float32).reshape(FOG_H, FOG_W, -1)[..., :1], 3, -1)))
    return FogScene(decode(aged[..., :1]), decode(clean), ~wall)


def fog_report() -> None:
    from fade_crossover_scene import fog_scene
    from negpy.features.process.fade_fog import measure_fog

    print("\n# Uneven fog\n")
    for title, scene in (
        ("Uneven fog (left edge and top-right corner) with a light wall, Coolscan-like", fog_scene("fog")),
        ("Clean Coolscan-like scan: a light wall, a dark doorway and a large warm area, no fog", fog_scene("warm")),
        ("Clean Coolscan-like scan: a light wall, a dark doorway and a blue sky, no fog", fog_scene("sky")),
        ("NegativeConverter's own fog: their test scene with its blacks lifted in the positive", their_fog_scene()),
    ):
        print(f"\n## {title}\n")
        surface = measure_fog(scene.fogged)
        print(f"NegPy surface: {'none (under the floor)' if surface is None else f'amplitude {surface.amplitude():.3f} D'}\n")
        clean_lab = to_lab(render_fog(scene.clean, "off"))
        rows = {fog: fog_scores(to_lab(render_fog(scene.fogged, fog)), clean_lab, scene) for fog in FOGS}
        keys = list(next(iter(rows.values())))
        print("| fog | " + " | ".join(keys) + " |\n|---|" + "---|" * len(keys))
        for fog, sc in rows.items():
            print(f"| {fog} | " + " | ".join(v if isinstance(v, str) else f"{v:.2f}" for v in sc.values()) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
