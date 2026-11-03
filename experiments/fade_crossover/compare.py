"""Render real slides through Fade Estimate, the crossover table and the Compare choices, headless.

Phase 4 (the default, --phase 4) writes beside each input:
  <name>.p4-mix.png     Neutrals NegPy and No Sky at Mix 0, 50 and 100 %, each with a fresh Estimate and
                        Crossover 100 %;
  <name>.p4-tone.png    Tone Off, then NegPy and Adaptive at Tone Strength 50 and 100 %, labeled with the
                        Print Density and ISO-R Grade each writes;
  <name>.p4-fog.png     NegPy's fog fit over the scan as loaded: the removed density as a magenta tint and
                        the grid cells it kept (green), took for picture (orange) or found incoherent (red),
                        then the removed map alone and the render with fog NegPy.
It prints each Estimate and the fog fit's amplitude before the cap, whether the cap was hit and the share
of cells kept.

Phase 3 (--phase 3) writes:
  <name>.p3-crossover-0.png, -0.5.png, -1.png  the Estimate's survivals with the table at that strength;
  <name>.p3-model.png   color model NegPy at Crossover 0 and 100 %, and NC;
  <name>.p3-zero.png    scanner zero Full Scale, Clear Film and Gray (scanner files), Crossover 0 and 100 %,
                     each with a fresh Estimate;
  <name>.p3-tone.png    tone Off, NC and NegPy under color models NegPy (Crossover 100 %) and NC;
  <name>.p3-fog.png     uneven fog Off, NC and NegPy under color model NegPy (Crossover 100 %).
Each file opens as a Slide, as the app opens it (a scanner file starts Positive), with the slide defaults.

    uv run python experiments/fade_crossover/compare.py Statue.NEF faded/slide01.tif [--long-edge 2000] [--gpu] [--phase 3|4|all]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from negpy.desktop.workers.chart_fit import ChartSource, fade_estimate_grid
from negpy.domain.models import WorkspaceConfig
from negpy.features.process.fade import (
    EstimateMethod,
    FadeEstimate,
    crossover_amount,
    crossover_bands,
    crossover_description,
    crossover_lean_fit,
    crossover_space,
    crossover_zero,
    estimate_fade,
    fade_density,
    fade_white_space,
    render_white_space,
)
from negpy.features.process.fade_fog import FOG_CAP, FogFit, fit_fog, fog_map, measure_fog
from negpy.features.process.fade_tone import TONE_FITS, fade_tone_changes
from negpy.features.process.models import FadeABTone, ProcessMode, mode_aware_exposure_reset, with_process_mode
from negpy.infrastructure.loaders.source_kind import with_scanner_positive_default
from negpy.services.rendering.image_processor import ImageProcessor, _downsample_to_long_edge

STRENGTHS = (0.0, 0.5, 1.0)
ZEROS = (("Full Scale", "full_scale"), ("Clear Film", "clear_film"), ("Gray", "clear_neutral"))
MODELS = (("NegPy, Crossover 0", "negpy", 0.0), ("NegPy, Crossover 100 %", "negpy", 1.0), ("NC color table", "nc", 1.0))
TONES = (("tone Off", "off"), ("tone NC", "nc"), ("tone NegPy", "negpy"))
FOGS = (("fog Off", "off"), ("fog NC", "nc"), ("fog NegPy", "negpy"))
TILE = 560
LABEL_H = 28


def slide_config(path: str) -> WorkspaceConfig:
    base = WorkspaceConfig()
    config = with_scanner_positive_default(with_process_mode(base, ProcessMode.E6), path)
    return replace(config, exposure=mode_aware_exposure_reset(ProcessMode.E6, config.exposure))


class Frame:
    """One input, loaded once, with what the Estimate needs."""

    def __init__(self, path: Path, long_edge: int, gpu: bool):
        self.path = path
        self.processor = ImageProcessor(use_gpu=gpu)
        self.gpu = gpu
        self.config = slide_config(str(path))
        buffer, _, _ = self.processor._load_source_f32(str(path), self.config)
        self.buffer = _downsample_to_long_edge(np.ascontiguousarray(buffer, dtype=np.float32), long_edge)
        self.cam_xyz, self.camera_wb = self.processor._cam_xyz_by_path.get(str(path), (None, None))
        self.source_hash = f"crossover-{path.name}"
        self.scanner = fade_white_space(self.cam_xyz, self.camera_wb) == ""
        self.grids: dict[str, np.ndarray] = {}

    def grid(self, fog: str = "off") -> np.ndarray:
        """The Estimate grid, read after the fog bake as the panel's Estimate reads it."""
        if fog not in self.grids:
            config = replace(self.config, process=replace(self.config.process, fade_ab_fog=fog))
            self.grids[fog] = fade_estimate_grid(ChartSource(self.buffer, config, self.source_hash, self.cam_xyz, self.camera_wb, (), 1))
        return self.grids[fog]

    def space(self, **process) -> str:
        proc = replace(self.config.process, **process)
        return render_white_space(proc, proc.process_mode, self.cam_xyz, self.camera_wb)

    def estimate(self, method: EstimateMethod, neutrals: str = "negpy", mix: float = 0.0, **process) -> FadeEstimate:
        space = self.space(**process)
        grid = self.grid(process.get("fade_ab_fog", "off"))
        return estimate_fade(grid, white_anchor=False, camera_raw=bool(space), neutrals=neutrals, white_space=space, mix=mix)[method]

    def changes(self, est: FadeEstimate, **process) -> dict:
        """The process fields the Estimate button stores: survivals, the tagged white and table."""
        proc = replace(self.config.process, **process)
        space = self.space(**process)
        out: dict = dict(process, fade_ratio_r=1.0 if est.red is None else est.red, fade_ratio_g=est.green, fade_ratio_b=est.blue)
        if space and est.white is not None:
            out.update(fade_white_r=est.white[0], fade_white_g=est.white[1], fade_white_b=est.white[2], fade_white_space=space)
        if est.crossover is not None:
            tagged = replace(proc, **out)
            out.update(
                fade_crossover_table=est.crossover,
                fade_crossover_space=crossover_space(fade_density(tagged, proc.process_mode, space), space, tagged),
            )
        return out

    def run(self, settings: WorkspaceConfig, cache_stages: bool = False) -> tuple[np.ndarray, dict]:
        result, metrics = self.processor.run_pipeline(
            self.buffer,
            settings,
            self.source_hash,
            render_size_ref=float(max(self.buffer.shape[:2])),
            prefer_gpu=self.gpu,
            wants_uv_grid=False,
            cam_xyz=self.cam_xyz,
            camera_wb=self.camera_wb,
            cache_stages=cache_stages,
        )
        if hasattr(result, "readback"):
            result = result.readback()
        return np.clip(np.asarray(result, dtype=np.float32)[..., :3], 0.0, 1.0), metrics

    def render(self, tone: str = "off", strength: float = 1.0, **process) -> np.ndarray:
        return self.render_toned(tone, strength, **process)[0]

    def render_toned(self, tone: str = "off", strength: float = 1.0, **process) -> tuple[np.ndarray, str]:
        """(render, label): the render with Tone *tone*; a Tone that writes the sliders reads them from a
        first render at *strength*, as the panel does, and the label names what it wrote."""
        settings = replace(self.config, process=replace(self.config.process, **process))
        if FadeABTone(tone) not in TONE_FITS:
            return self.run(replace(settings, process=replace(settings.process, fade_ab_tone=tone)))[0], ""
        _, metrics = self.run(settings, cache_stages=True)
        toned, reason = fade_tone_changes(settings, FadeABTone(tone), metrics, strength)
        if toned is None:
            print(f"  tone {tone}: {reason}")
            return self.run(settings)[0], reason
        label = f"Print Density {toned.exposure.density:.2f}, ISO-R {toned.exposure.grade:.0f}"
        print(f"  tone {tone} at {strength * 100:g} %: {label}")
        return self.run(toned)[0], label


def _png(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8))


def print_estimate(label: str, est: FadeEstimate) -> None:
    print(f"  Estimate, {label}: red {est.red}, green {est.green:.3f}, blue {est.blue:.3f} {est.reason}")
    if est.crossover is None:
        return
    amount = crossover_amount(est.crossover)
    slope, residual = crossover_lean_fit(est.crossover)
    print(f"    table zero {crossover_zero(est.crossover):+.3f}, crossover {amount:.3f} D: {crossover_description(est.crossover)}")
    print("    lean slope r/g/b " + "/".join(f"{v:+.3f}" for v in slope) + f", residual after the line {residual:.3f} D")
    print("    bands (mean log, share, lean r g b):")
    for row in crossover_bands(est.crossover):
        print("     " + "  ".join(f"{v:+.3f}" for v in row))


def sheet(rows: list[list[tuple[str, np.ndarray]]]) -> Image.Image:
    """A labeled contact sheet: one row per list, one tile per (label, render)."""
    font = ImageFont.load_default()
    tiles = []
    for row in rows:
        out = []
        for label, img in row:
            tile = _png(img)
            tile.thumbnail((TILE, TILE))
            canvas = Image.new("RGB", (TILE, tile.height + LABEL_H), (24, 24, 24))
            canvas.paste(tile, ((TILE - tile.width) // 2, LABEL_H))
            ImageDraw.Draw(canvas).text((8, 8), label, fill=(235, 235, 235), font=font)
            out.append(canvas)
        tiles.append(out)
    h = max(t.height for r in tiles for t in r)
    w = max(len(r) for r in tiles)
    out = Image.new("RGB", (TILE * w, h * len(tiles)), (24, 24, 24))
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            out.paste(tile, (x * TILE, y * h))
    return out


def fog_overlay(linear: np.ndarray, fit: FogFit) -> tuple[np.ndarray, np.ndarray]:
    """(overlay, map): the scan as loaded with NegPy's fitted fog over it, and the removed density alone,
    both display-encoded. The removed density tints magenta, full at FOG_CAP; each grid cell is a dot,
    green where the fit kept it, orange where it read as picture, red where its channel floors disagree."""
    shown = np.power(np.clip(np.asarray(linear, dtype=np.float32)[..., :3], 0.0, 1.0), 1.0 / 2.2)
    h, w = shown.shape[:2]
    removed = np.zeros((h, w)) if fit.surface is None else fog_map(linear, fit.surface)
    a = (removed / FOG_CAP)[..., None] * 0.5
    overlay = shown * (1.0 - a) + np.array([1.0, 0.0, 1.0]) * a
    heat = np.repeat((removed / FOG_CAP)[..., None], 3, axis=-1) * np.array([1.0, 0.0, 1.0])
    img = _png(np.clip(overlay, 0.0, 1.0))
    draw = ImageDraw.Draw(img)
    r = max(2, w // 300)
    for u, v, coherent, kept in zip(fit.u, fit.v, fit.coherent, fit.kept):
        colour = (40, 220, 60) if kept else ((255, 150, 0) if coherent else (230, 30, 30))
        x, y = u * w, v * h
        draw.ellipse((x - r, y - r, x + r, y + r), fill=colour)
    return np.asarray(img, dtype=np.float32) / 255.0, np.clip(heat, 0.0, 1.0)


def compare_p4(frame: "Frame", method: EstimateMethod, write) -> None:
    rows = []
    for name, neutrals in (("NegPy", "negpy"), ("No Sky", "no_sky")):
        row = []
        for mix in (0.0, 0.5, 1.0):
            est = frame.estimate(method, neutrals, mix)
            print_estimate(f"{method}, Neutrals {name}, Mix {mix * 100:g} %", est)
            row.append((f"Neutrals {name} | Mix {mix * 100:g} % | Crossover 100 %", frame.render(**frame.changes(est), fade_crossover=1.0)))
        rows.append(row)
    write("mix", sheet(rows))

    base = frame.changes(frame.estimate(method))
    off = ("Tone Off | NegPy, Crossover 100 %", frame.render(**base, fade_crossover=1.0))
    tone_rows = []
    for name, tone in (("NegPy", "negpy"), ("Adaptive", "adaptive")):
        row = [off]
        for strength in (0.5, 1.0):
            img, label = frame.render_toned(tone, strength, **base, fade_crossover=1.0)
            row.append((f"Tone {name} {strength * 100:g} % | {label}", img))
        tone_rows.append(row)
    write("tone", sheet(tone_rows))

    fit = fit_fog(frame.buffer)
    amp = "none" if fit.amplitude is None else f"{fit.amplitude:.3f} D"
    print(
        f"  fog NegPy fit: amplitude before the cap {amp}, cap hit {fit.cap_hit}, cells kept {fit.kept_share:.0%}, {fit.reason or 'removed'}"
    )
    overlay, heat = fog_overlay(frame.buffer, fit)
    fest = frame.estimate(method, fade_ab_fog="negpy")
    after = frame.render(**frame.changes(fest, fade_ab_fog="negpy"), fade_crossover=1.0)
    label = f"amp {amp}, cap {'hit' if fit.cap_hit else 'not hit'}, kept {fit.kept_share:.0%}"
    write(
        "fog",
        sheet([[(f"fog fit | {label}", overlay), (f"removed (magenta = {FOG_CAP:g} D)", heat), ("fog NegPy | Crossover 100 %", after)]]),
    )


def compare(path: Path, long_edge: int, method: EstimateMethod, gpu: bool = False, phase: str = "4") -> None:
    frame = Frame(path, long_edge, gpu)
    proc = frame.config.process
    kind = "scanner file" if frame.scanner else "camera RAW"
    print(f"{path.name}: {kind}, {'Positive' if proc.positive_source else 'Slide'}, {frame.buffer.shape[1]}x{frame.buffer.shape[0]}")
    if frame.grid() is None:
        print("  no Estimate grid")
        return

    def writer(prefix: str):
        def write(suffix: str, image: Image.Image) -> None:
            out = path.with_name(f"{path.stem}.{prefix}-{suffix}.png")
            image.save(out)
            print(f"  wrote {out}")

        return write

    if phase in ("4", "all"):
        compare_p4(frame, method, writer("p4"))
    if phase in ("3", "all"):
        compare_p3(frame, method, writer("p3"))


def compare_p3(frame: "Frame", method: EstimateMethod, write) -> None:
    est = frame.estimate(method)
    print_estimate(f"{method}, NegPy neutrals", est)
    print_estimate(f"{method}, NC neutrals", frame.estimate(method, "nc"))
    base = frame.changes(est)
    for strength in STRENGTHS:
        write(f"crossover-{strength:g}", _png(frame.render(**base, fade_crossover=strength)))

    write("model", sheet([[(label, frame.render(**base, fade_ab_model=model, fade_crossover=x)) for label, model, x in MODELS]]))

    if frame.scanner:
        rows: list[list] = [[], []]
        for label, zero in ZEROS:
            print(f"  scanner zero {label}:")
            zest = frame.estimate(method, fade_ab_scanner_zero=zero)
            print_estimate(label, zest)
            changes = frame.changes(zest, fade_ab_scanner_zero=zero)
            for row, x in zip(rows, (0.0, 1.0)):
                row.append((f"zero {label} | Crossover {x * 100:g} %", frame.render(**changes, fade_crossover=x)))
        write("zero", sheet(rows))

    tone_rows = []
    for label, model, x in (MODELS[1], MODELS[2]):
        print(f"  tone under {label}:")
        tone_rows.append([(f"{label} | {tl}", frame.render(tone, **base, fade_ab_model=model, fade_crossover=x)) for tl, tone in TONES])
    write("tone", sheet(tone_rows))

    surface = measure_fog(frame.buffer)
    print(f"  fog NegPy surface: {'none' if surface is None else f'{surface.amplitude():.3f} D'}")
    fog_row = []
    for label, fog in FOGS:
        fest = frame.estimate(method, fade_ab_fog=fog)
        fog_row.append((f"{label} | NegPy, Crossover 100 %", frame.render(**frame.changes(fest, fade_ab_fog=fog), fade_crossover=1.0)))
    write("fog", sheet([fog_row]))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="camera RAW or scanner TIFF files")
    parser.add_argument("--long-edge", type=int, default=2000, help="downsample to this long edge first (default 2000)")
    parser.add_argument("--method", choices=[m.value for m in EstimateMethod], default=EstimateMethod.MIDTONE.value)
    parser.add_argument("--gpu", action="store_true", help="render on the GPU where a stage allows, as the preview does")
    parser.add_argument("--phase", choices=("3", "4", "all"), default="4", help="which phase's sheets to write (default 4)")
    args = parser.parse_args(argv)
    for path in args.paths:
        compare(path.resolve(), args.long_edge, EstimateMethod(args.method), args.gpu, args.phase)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
