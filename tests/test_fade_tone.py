"""Tone NegPy (Compare): their automatic tone read in density and written into the Tone sliders."""

from dataclasses import replace

import numpy as np
import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.process.fade_tone import TONE_FIELDS, TONE_PCT, fade_tone_changes, tone_points, tone_values
from negpy.features.process.models import FadeABTone, ProcessMode, mode_aware_exposure_reset


def _slide(**exposure) -> WorkspaceConfig:
    base = WorkspaceConfig()
    cfg = replace(base, process=replace(base.process, process_mode=ProcessMode.E6))
    return replace(cfg, exposure=replace(mode_aware_exposure_reset(ProcessMode.E6, base.exposure), **exposure))


def _ramp(light: float, dense: float) -> tuple[float, ...]:
    return tuple(light + (dense - light) * p / 100.0 for p in TONE_PCT)


def test_the_points_are_the_mean_density_quantiles_light_end_first():
    rng = np.random.default_rng(0)
    grid = np.repeat(rng.uniform(0.2, 1.4, (64, 64, 1)), 3, axis=-1)
    points = tone_points(grid)
    assert len(points) == len(TONE_PCT) and points == tuple(sorted(points))
    assert points[0] == pytest.approx(0.2, abs=0.02) and points[-1] == pytest.approx(1.4, abs=0.02)
    assert tone_points(np.full((16, 16, 3), 0.5)) is None


def test_a_low_range_frame_takes_a_harder_grade_than_a_full_range_one():
    full = tone_values(_ramp(0.2, 2.9), positive_source=True)
    low = tone_values(_ramp(0.3, 1.5), positive_source=True)
    assert low["grade"] < full["grade"]
    assert 0.0 <= low["density"] <= 2.0


@pytest.mark.parametrize("back", [FadeABTone.OFF, FadeABTone.NC])
def test_leaving_negpy_restores_the_sliders_exactly(back):
    before = _slide(density=1.23, grade=101.0, auto_exposure=True, auto_normalize_contrast=True)
    written, reason = fade_tone_changes(before, FadeABTone.NEGPY, {"fade_tone_points": _ramp(0.3, 1.6)})
    assert not reason and written.process.fade_tone_saved == tuple(getattr(before.exposure, f) for f in TONE_FIELDS)
    assert not written.exposure.auto_exposure and not written.exposure.auto_normalize_contrast
    restored, _ = fade_tone_changes(written, back, {})
    assert restored.exposure == before.exposure
    assert (restored.process.fade_ab_tone, restored.process.fade_tone_saved) == (back, None)


def test_negpy_without_a_reading_changes_nothing():
    cfg, reason = fade_tone_changes(_slide(), FadeABTone.NEGPY, {})
    assert cfg is None and reason


def test_the_render_reports_the_points_on_a_slide():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from fade_crossover_scene import CROSSOVER, build_scene, scan_image
    from negpy.domain.interfaces import PipelineContext
    from negpy.services.rendering.engine import DarkroomEngine

    image = scan_image(CROSSOVER, build_scene(192, 144))
    ctx = PipelineContext(original_size=image.shape[:2], scale_factor=1.0, process_mode=ProcessMode.E6)
    DarkroomEngine().process(image, _slide(), "tone-points", ctx)
    points = ctx.metrics["fade_tone_points"]
    assert points[0] < points[len(points) // 2] < points[-1]
