"""Uneven fog (Compare): NegPy's density bake and NegativeConverter's display-space stage."""

from dataclasses import replace

import numpy as np
import pytest

from fade_crossover_scene import fog_scene, render, to_lab
from negpy.features.process.fade_fog import FOG_CAP, bake_fog, fog_token, measure_fog
from negpy.features.process.models import ProcessConfig, ProcessMode


def _process(fog: str = "negpy", **fields) -> ProcessConfig:
    return replace(ProcessConfig(), process_mode=ProcessMode.E6, positive_source=True, fade_ab_fog=fog, **fields)


def _lab(image: np.ndarray, fog: str) -> np.ndarray:
    changes = dict(positive_source=True, fade_ab_fog=fog)
    return to_lab(render(bake_fog(image, _process(fog)), changes))


def _gradient(lab: np.ndarray, clean: np.ndarray, gray: np.ndarray) -> float:
    """The spread of the mean L* error over eight column bands of the gray cells in the upper half."""
    h, w = gray.shape
    d = lab[..., 0] - clean[..., 0]
    edges = np.linspace(0, w, 9).astype(int)
    rows = gray[: h // 2]
    bands = [d[: h // 2, a:b][rows[:, a:b]].mean() for a, b in zip(edges[:-1], edges[1:])]
    return float(max(bands) - min(bands))


def test_the_bake_runs_only_on_a_slide_with_negpy_fog():
    assert fog_token(_process()) and not fog_token(_process("off")) and not fog_token(_process("nc"))
    assert not fog_token(replace(_process(), process_mode=ProcessMode.C41, positive_source=False))
    scene = fog_scene("fog")
    assert bake_fog(scene.fogged, _process("off")) is scene.fogged


def test_negpy_removes_the_uneven_fog_and_leaves_the_wall():
    scene = fog_scene("fog")
    surface = measure_fog(scene.fogged)
    assert surface is not None and 0.05 < surface.amplitude() <= FOG_CAP
    clean = _lab(scene.clean, "off")
    before = _gradient(_lab(scene.fogged, "off"), clean, scene.gray)
    after = _gradient(_lab(scene.fogged, "negpy"), clean, scene.gray)
    assert after < 0.3 * before


@pytest.mark.parametrize("layout", ["warm", "sky"])
def test_a_clean_scan_stays_bit_identical(layout):
    scene = fog_scene(layout)
    assert measure_fog(scene.fogged) is None
    assert bake_fog(scene.fogged, _process()) is scene.fogged


def test_nc_paints_blue_on_a_clean_wall_under_a_sky_and_negpy_does_not():
    """Their per-channel floor reads the sky's thin blue as fog in blue only, so the wall turns blue."""
    scene = fog_scene("sky")
    off = _lab(scene.fogged, "off")
    added = {}
    for fog in ("nc", "negpy"):
        lab = _lab(scene.fogged, fog)
        db = (lab[..., 2] - off[..., 2])[scene.gray]
        added[fog] = float(np.hypot((lab[..., 1] - off[..., 1])[scene.gray], db).max()), float(db.min())
    assert added["nc"][0] > 10.0 and added["nc"][1] < -10.0
    assert added["negpy"][0] == 0.0


def test_run_pipeline_bakes_the_fog_and_keys_it_in_the_source():
    from negpy.domain.models import WorkspaceConfig
    from negpy.features.process.models import mode_aware_exposure_reset
    from negpy.services.rendering.image_processor import ImageProcessor

    scene = fog_scene("fog")
    base = WorkspaceConfig()
    settings = replace(base, exposure=mode_aware_exposure_reset(ProcessMode.E6, base.exposure))

    def run(fog: str) -> np.ndarray:
        cfg = replace(settings, process=_process(fog))
        out, _ = ImageProcessor(use_gpu=False).run_pipeline(scene.fogged, cfg, "fog-scene", render_size_ref=384.0, prefer_gpu=False, wants_uv_grid=False)
        return np.asarray(out)

    off, negpy = run("off"), run("negpy")
    assert np.abs(negpy - off).max() > 0.01
    processor = ImageProcessor(use_gpu=False)
    cfg = replace(settings, process=_process("negpy"))
    first, _ = processor.run_pipeline(scene.fogged, cfg, "fog-scene", render_size_ref=384.0, prefer_gpu=False, wants_uv_grid=False)
    again, _ = processor.run_pipeline(scene.fogged, cfg, "fog-scene", render_size_ref=384.0, prefer_gpu=False, wants_uv_grid=False)
    assert np.array_equal(np.asarray(first), np.asarray(again))
