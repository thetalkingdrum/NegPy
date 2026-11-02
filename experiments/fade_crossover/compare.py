"""Render real slides through Fade Estimate, the crossover table and the NegativeConverter A/B stages,
headless.

For each input, writes beside it:
  <name>.crossover-0.png, -0.5.png, -1.png  the Estimate's survivals with the table (Hybrid) at that strength;
  <name>.compare.png                         a labeled contact sheet: tone Off and NC across, and the color
                                             models NegPy, Hybrid and NC, plus NegPy with the NC neutrals, down.
Each file opens as a Slide, as the app opens it (a scanner file starts Positive), with the slide defaults.

    uv run python experiments/fade_crossover/compare.py Statue.NEF faded/slide01.tif [--long-edge 2000] [--gpu]
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
    crossover_space,
    crossover_zero,
    estimate_fade,
    fade_density,
    render_white_space,
)
from negpy.features.process.models import ProcessMode, mode_aware_exposure_reset, with_process_mode
from negpy.infrastructure.loaders.source_kind import with_scanner_positive_default
from negpy.services.rendering.image_processor import ImageProcessor, _downsample_to_long_edge

STRENGTHS = (0.0, 0.5, 1.0)
#: Contact-sheet rows: (label, color model, Estimate neutrals).
ROWS = (
    ("NegPy (Survival)", "negpy", "negpy"),
    ("Hybrid (Survival + Crossover 100 %)", "hybrid", "negpy"),
    ("NC color table", "nc", "negpy"),
    ("NegPy, NC neutrals", "negpy", "nc"),
)
TONES = (("tone Off", "off"), ("tone NC", "nc"))
TILE = 640
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
        proc = self.config.process
        self.space = render_white_space(proc, proc.process_mode, self.cam_xyz, self.camera_wb)
        self.grid = fade_estimate_grid(ChartSource(self.buffer, self.config, self.source_hash, self.cam_xyz, self.camera_wb, (), 1))

    def estimate(self, method: EstimateMethod, neutrals: str) -> FadeEstimate:
        return estimate_fade(self.grid, white_anchor=False, camera_raw=bool(self.space), neutrals=neutrals)[method]

    def changes(self, est: FadeEstimate) -> dict:
        """The process fields the Estimate button stores: survivals, a RAW's white, the tagged table."""
        proc = self.config.process
        out: dict = dict(fade_ratio_r=1.0 if est.red is None else est.red, fade_ratio_g=est.green, fade_ratio_b=est.blue)
        if self.space and est.white is not None:
            out.update(fade_white_r=est.white[0], fade_white_g=est.white[1], fade_white_b=est.white[2], fade_white_space=self.space)
        if est.crossover is not None:
            tagged = replace(proc, **out)
            out.update(
                fade_crossover_table=est.crossover,
                fade_crossover_space=crossover_space(fade_density(tagged, proc.process_mode, self.space), self.space, tagged),
            )
        return out

    def render(self, **process) -> np.ndarray:
        settings = replace(self.config, process=replace(self.config.process, **process))
        result, _ = self.processor.run_pipeline(
            self.buffer,
            settings,
            self.source_hash,
            render_size_ref=float(max(self.buffer.shape[:2])),
            prefer_gpu=self.gpu,
            wants_uv_grid=False,
            cam_xyz=self.cam_xyz,
            camera_wb=self.camera_wb,
            cache_stages=False,
        )
        if hasattr(result, "readback"):
            result = result.readback()
        return np.clip(np.asarray(result, dtype=np.float32)[..., :3], 0.0, 1.0)


def _png(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8))


def print_estimate(label: str, est: FadeEstimate) -> None:
    print(f"  Estimate, {label}: red {est.red}, green {est.green:.3f}, blue {est.blue:.3f} {est.reason}")
    if est.crossover is None:
        return
    amount = crossover_amount(est.crossover)
    print(f"    table zero {crossover_zero(est.crossover):+.3f}, crossover {amount:.3f} D: {crossover_description(est.crossover)}")
    print("    bands (mean log, share, lean r g b):")
    for row in crossover_bands(est.crossover):
        print("     " + "  ".join(f"{v:+.3f}" for v in row))


def contact_sheet(frame: Frame, estimates: dict[str, FadeEstimate]) -> Image.Image:
    font = ImageFont.load_default()
    tiles = []
    for label, model, neutrals in ROWS:
        row = []
        for tone_label, tone in TONES:
            img = frame.render(**frame.changes(estimates[neutrals]), fade_ab_model=model, fade_ab_tone=tone, fade_crossover=1.0)
            tile = _png(img)
            tile.thumbnail((TILE, TILE))
            canvas = Image.new("RGB", (TILE, tile.height + LABEL_H), (24, 24, 24))
            canvas.paste(tile, ((TILE - tile.width) // 2, LABEL_H))
            ImageDraw.Draw(canvas).text((8, 8), f"{label} | {tone_label}", fill=(235, 235, 235), font=font)
            row.append(canvas)
        tiles.append(row)
    h = max(t.height for r in tiles for t in r)
    sheet = Image.new("RGB", (TILE * len(TONES), h * len(ROWS)), (24, 24, 24))
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            sheet.paste(tile, (x * TILE, y * h))
    return sheet


def compare(path: Path, long_edge: int, method: EstimateMethod, gpu: bool = False) -> None:
    frame = Frame(path, long_edge, gpu)
    proc = frame.config.process
    print(f"{path.name}: {'Positive' if proc.positive_source else 'Slide'}, {frame.buffer.shape[1]}x{frame.buffer.shape[0]}")
    if frame.grid is None:
        print("  no Estimate grid")
        return
    estimates = {neutrals: frame.estimate(method, neutrals) for neutrals in ("negpy", "nc")}
    print_estimate(f"{method}, NegPy neutrals", estimates["negpy"])
    print_estimate(f"{method}, NC neutrals", estimates["nc"])
    changes = frame.changes(estimates["negpy"])
    for strength in STRENGTHS:
        out = path.with_name(f"{path.stem}.crossover-{strength:g}.png")
        _png(frame.render(**changes, fade_ab_model="hybrid", fade_crossover=strength)).save(out)
        print(f"  wrote {out}")
    out = path.with_name(f"{path.stem}.compare.png")
    contact_sheet(frame, estimates).save(out)
    print(f"  wrote {out}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="camera RAW or scanner TIFF files")
    parser.add_argument("--long-edge", type=int, default=2000, help="downsample to this long edge first (default 2000)")
    parser.add_argument("--method", choices=[m.value for m in EstimateMethod], default=EstimateMethod.MIDTONE.value)
    parser.add_argument("--gpu", action="store_true", help="render on the GPU where a stage allows, as the preview does")
    args = parser.parse_args(argv)
    for path in args.paths:
        compare(path.resolve(), args.long_edge, EstimateMethod(args.method), args.gpu)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
