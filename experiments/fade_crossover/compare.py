"""Render real slides through Fade Estimate and the prototype crossover table, headless, on the CPU.

For each input, writes <name>.crossover-0.png, -0.5.png and -1.png beside it: the Estimate's survivals
with the table at that strength. Each file opens as a Slide, as the app opens it (a scanner file
starts Positive), with the slide defaults and no other edit.

    uv run python experiments/fade_crossover/compare.py Statue.NEF faded/slide01.tif [--long-edge 2000]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from negpy.desktop.workers.chart_fit import ChartSource, fade_estimate_grid
from negpy.domain.models import WorkspaceConfig
from negpy.features.process.fade import (
    EstimateMethod,
    crossover_bands,
    crossover_space,
    estimate_fade,
    fade_density,
    render_white_space,
)
from negpy.features.process.models import ProcessMode, mode_aware_exposure_reset, with_process_mode
from negpy.infrastructure.loaders.source_kind import with_scanner_positive_default
from negpy.services.rendering.image_processor import ImageProcessor, _downsample_to_long_edge

STRENGTHS = (0.0, 0.5, 1.0)


def slide_config(path: str) -> WorkspaceConfig:
    base = WorkspaceConfig()
    config = with_scanner_positive_default(with_process_mode(base, ProcessMode.E6), path)
    return replace(config, exposure=mode_aware_exposure_reset(ProcessMode.E6, config.exposure))


def compare(path: Path, long_edge: int, method: EstimateMethod) -> None:
    processor = ImageProcessor(use_gpu=False)
    config = slide_config(str(path))
    buffer, _, _ = processor._load_source_f32(str(path), config)
    buffer = _downsample_to_long_edge(np.ascontiguousarray(buffer, dtype=np.float32), long_edge)
    cam_xyz, camera_wb = processor._cam_xyz_by_path.get(str(path), (None, None))
    source_hash = f"crossover-{path.name}"
    proc = config.process
    print(f"{path.name}: {'Positive' if proc.positive_source else 'Slide'}, {buffer.shape[1]}x{buffer.shape[0]}")

    grid = fade_estimate_grid(ChartSource(buffer, config, source_hash, cam_xyz, camera_wb, (), 1))
    if grid is None:
        print("  no Estimate grid")
        return
    space = render_white_space(proc, proc.process_mode, cam_xyz, camera_wb)
    est = estimate_fade(grid, white_anchor=False, camera_raw=bool(space))[method]
    print(f"  Estimate ({method}): red {est.red}, green {est.green:.3f}, blue {est.blue:.3f} {est.reason}")
    changes: dict = dict(fade_ratio_r=1.0 if est.red is None else est.red, fade_ratio_g=est.green, fade_ratio_b=est.blue)
    if space and est.white is not None:
        changes.update(fade_white_r=est.white[0], fade_white_g=est.white[1], fade_white_b=est.white[2], fade_white_space=space)
    if est.crossover is not None:
        tagged = replace(proc, **changes)
        changes.update(
            fade_crossover_table=est.crossover,
            fade_crossover_space=crossover_space(fade_density(tagged, proc.process_mode, space), space),
        )
        print("  bands (mean log, share, lean r g b):")
        for row in crossover_bands(est.crossover):
            print("   " + "  ".join(f"{v:+.3f}" for v in row))
    else:
        print("  no table from the Estimate; the render measures its own")

    for strength in STRENGTHS:
        settings = replace(config, process=replace(proc, fade_crossover=strength, **changes))
        result, metrics = processor.run_pipeline(
            buffer,
            settings,
            source_hash,
            render_size_ref=float(max(buffer.shape[:2])),
            prefer_gpu=False,
            wants_uv_grid=False,
            cam_xyz=cam_xyz,
            camera_wb=camera_wb,
            cache_stages=False,
        )
        rgb = np.clip(np.asarray(result, dtype=np.float32)[..., :3], 0.0, 1.0)
        out = path.with_name(f"{path.stem}.crossover-{strength:g}.png")
        Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8)).save(out)
        note = " (table measured by the render)" if "fade_crossover" in metrics else ""
        print(f"  wrote {out}{note}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="camera RAW or scanner TIFF files")
    parser.add_argument("--long-edge", type=int, default=2000, help="downsample to this long edge first (default 2000)")
    parser.add_argument("--method", choices=[m.value for m in EstimateMethod], default=EstimateMethod.MIDTONE.value)
    args = parser.parse_args(argv)
    for path in args.paths:
        compare(path.resolve(), args.long_edge, EstimateMethod(args.method))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
