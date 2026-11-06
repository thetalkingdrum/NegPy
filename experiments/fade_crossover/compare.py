"""Render real slides through Fade Estimate, the crossover table and the Compare choices, headless.

Phase 4 (the default, --phase 4) writes beside each input:
  <name>.p4-mix.png     Neutrals 0, 50 and 100 %, each with a fresh Estimate and Crossover 100 %;
  <name>.p4-tone.png    Tone 0, 50 and 100 %, labeled with the Print Density and ISO-R Grade each writes.
It prints each Estimate.

Phase 3 (--phase 3) writes:
  <name>.p3-crossover-0.png, -0.5.png, -1.png  the Estimate's survivals with the table at that strength;
  <name>.p3-model.png   color model NegPy at Crossover 0 and 100 %, and NC;
  <name>.p3-zero.png    scanner zero Full Scale, Clear Film and Gray (scanner files), Crossover 0 and 100 %,
                     each with a fresh Estimate.
Each file opens as a Slide, as the app opens it (a scanner file starts Positive), with the slide defaults.

    uv run python experiments/fade_crossover/compare.py Statue.NEF faded/slide01.tif [--long-edge 2000] [--gpu] [--phase 3|4|all]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from negpy.desktop.workers.chart_fit import ChartSource, fade_estimate_grid
from negpy.domain.models import WorkspaceConfig
from negpy.features.process.fade import (
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
from negpy.features.process.fade_tone import fade_tone_changes
from negpy.features.process.models import ProcessMode, mode_aware_exposure_reset, with_process_mode
from negpy.infrastructure.loaders.source_kind import with_scanner_positive_default
from negpy.services.rendering.image_processor import ImageProcessor, _downsample_to_long_edge

STRENGTHS = (0.0, 0.5, 1.0)
ZEROS = (("Full Scale", "full_scale"), ("Clear Film", "clear_film"), ("Gray", "clear_neutral"))
MODELS = (("NegPy, Crossover 0", "negpy", 0.0), ("NegPy, Crossover 100 %", "negpy", 1.0), ("NC color table", "nc", 1.0))
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
        self._grid: Optional[np.ndarray] = None

    def grid(self) -> Optional[np.ndarray]:
        """The Estimate grid, as the panel's Estimate reads it."""
        if self._grid is None:
            self._grid = fade_estimate_grid(ChartSource(self.buffer, self.config, self.source_hash, self.cam_xyz, self.camera_wb, (), 1))
        return self._grid

    def space(self, **process) -> str:
        proc = replace(self.config.process, **process)
        return render_white_space(proc, proc.process_mode, self.cam_xyz, self.camera_wb)

    def estimate(self, mix: float = 0.0, **process) -> FadeEstimate:
        space = self.space(**process)
        grid = self.grid()
        return estimate_fade(grid, white_anchor=False, camera_raw=bool(space), white_space=space, mix=mix)

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

    def render(self, tone: float = 0.0, **process) -> np.ndarray:
        return self.render_toned(tone, **process)[0]

    def render_toned(self, tone: float = 0.0, **process) -> tuple[np.ndarray, str]:
        """(render, label): the render with Tone at *tone*, read from a first render as the panel does;
        the label names what it wrote."""
        settings = replace(self.config, process=replace(self.config.process, **process))
        if tone <= 0.0:
            return self.run(settings)[0], ""
        _, metrics = self.run(settings, cache_stages=True)
        toned, reason = fade_tone_changes(settings, metrics, tone)
        if toned is None:
            print(f"  tone {tone * 100:g} %: {reason}")
            return self.run(settings)[0], reason
        label = f"Print Density {toned.exposure.density:.2f}, ISO-R {toned.exposure.grade:.0f}"
        print(f"  tone {tone * 100:g} %: {label}")
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


def compare_p4(frame: "Frame", write) -> None:
    row = []
    for mix in (0.0, 0.5, 1.0):
        est = frame.estimate(mix)
        print_estimate(f"Neutrals {mix * 100:g} %", est)
        row.append((f"Neutrals {mix * 100:g} % | Crossover 100 %", frame.render(**frame.changes(est), fade_crossover=1.0)))
    write("mix", sheet([row]))

    base = frame.changes(frame.estimate())
    row = []
    for tone in (0.0, 0.5, 1.0):
        img, label = frame.render_toned(tone, **base, fade_crossover=1.0)
        row.append((f"Tone {tone * 100:g} % | {label or 'your values'}", img))
    write("tone", sheet([row]))


def compare(path: Path, long_edge: int, gpu: bool = False, phase: str = "4") -> None:
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
        compare_p4(frame, writer("p4"))
    if phase in ("3", "all"):
        compare_p3(frame, writer("p3"))


def compare_p3(frame: "Frame", write) -> None:
    est = frame.estimate()
    print_estimate("Neutrals 0 %", est)
    print_estimate("Neutrals 100 %", frame.estimate(1.0))
    base = frame.changes(est)
    for strength in STRENGTHS:
        write(f"crossover-{strength:g}", _png(frame.render(**base, fade_crossover=strength)))

    write("model", sheet([[(label, frame.render(**base, fade_ab_model=model, fade_crossover=x)) for label, model, x in MODELS]]))

    if frame.scanner:
        rows: list[list] = [[], []]
        for label, zero in ZEROS:
            print(f"  scanner zero {label}:")
            zest = frame.estimate(fade_ab_scanner_zero=zero)
            print_estimate(label, zest)
            changes = frame.changes(zest, fade_ab_scanner_zero=zero)
            for row, x in zip(rows, (0.0, 1.0)):
                row.append((f"zero {label} | Crossover {x * 100:g} %", frame.render(**changes, fade_crossover=x)))
        write("zero", sheet(rows))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="camera RAW or scanner TIFF files")
    parser.add_argument("--long-edge", type=int, default=2000, help="downsample to this long edge first (default 2000)")
    parser.add_argument("--gpu", action="store_true", help="render on the GPU where a stage allows, as the preview does")
    parser.add_argument("--phase", choices=("3", "4", "all"), default="4", help="which phase's sheets to write (default 4)")
    args = parser.parse_args(argv)
    for path in args.paths:
        compare(path.resolve(), args.long_edge, args.gpu, args.phase)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
