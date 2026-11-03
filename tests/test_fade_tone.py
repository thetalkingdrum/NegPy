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


def test_adaptive_keeps_a_full_range_frame_near_the_default_grade():
    from negpy.features.process.fade_tone import ADAPTIVE_SPAN, adaptive_values

    # _ramp spreads the span over 0 to 100 %; the 0.5 to 99.5 % points span 0.99 of it.
    values = adaptive_values(_ramp(0.25, 0.25 + ADAPTIVE_SPAN / 0.99), positive_source=True)
    assert values["grade"] == 115.0


@pytest.mark.parametrize("span", [0.1, 0.4, 0.8, 1.2])
def test_adaptive_holds_the_grade_at_its_contrast_limit(span):
    from negpy.features.process.fade_tone import ADAPTIVE_CONTRAST, adaptive_values
    from negpy.features.transparency.logic import TRANSFER_CONSTANTS

    floor = round(float(TRANSFER_CONSTANTS["transfer_grade_ref"]) / ADAPTIVE_CONTRAST[1])
    values = adaptive_values(_ramp(0.4, 0.4 + span), positive_source=True)
    assert values["grade"] >= floor
    assert tone_values(_ramp(0.4, 0.4 + span), positive_source=True)["grade"] <= values["grade"]


def test_adaptive_prints_the_median_at_its_midtone():
    from negpy.features.process.fade_tone import ADAPTIVE_MIDTONE, adaptive_values
    from negpy.features.transparency.logic import TRANSFER_CONSTANTS

    c = TRANSFER_CONSTANTS
    pivot = float(c["transfer_contrast_pivot"])
    for points in (_ramp(0.3, 2.0), _ramp(0.6, 1.4)):
        values = adaptive_values(points, positive_source=True)
        offset = (1.0 - values["density"]) * float(c["transfer_density_stops"]) * np.log10(2.0)
        contrast = float(c["transfer_grade_ref"]) / values["grade"]
        assert pivot + (float(np.median(points)) - offset - pivot) * contrast == pytest.approx(ADAPTIVE_MIDTONE, abs=0.01)


def test_adaptive_never_lowers_the_contrast_of_a_wide_frame():
    from negpy.features.process.fade_tone import adaptive_values

    assert adaptive_values(_ramp(0.1, 3.4), positive_source=True)["grade"] == 115.0


@pytest.mark.parametrize("tone", [FadeABTone.NEGPY, FadeABTone.ADAPTIVE])
def test_strength_moves_the_sliders_from_the_saved_values_and_off_still_restores(tone):
    from negpy.features.process.fade_tone import TONE_FITS

    before = _slide(density=1.2, grade=110.0)
    metrics = {"fade_tone_points": _ramp(0.3, 1.2)}
    fit = TONE_FITS[tone](metrics["fade_tone_points"], positive_source=False)
    full, _ = fade_tone_changes(before, tone, metrics, 1.0)
    assert (full.exposure.density, full.exposure.grade) == (fit["density"], fit["grade"])
    none, _ = fade_tone_changes(full, tone, metrics, 0.0)
    assert (none.exposure.density, none.exposure.grade) == (1.2, 110.0)
    half, _ = fade_tone_changes(none, tone, metrics, 0.5)
    assert half.process.fade_tone_strength == 0.5
    assert half.exposure.grade == pytest.approx(round(0.5 * (110.0 + fit["grade"])), abs=1.0)
    assert half.exposure.density == pytest.approx(0.5 * (1.2 + fit["density"]), abs=0.011)
    assert half.process.fade_tone_saved == full.process.fade_tone_saved
    restored, _ = fade_tone_changes(half, FadeABTone.OFF, {})
    assert restored.exposure == before.exposure


def test_strength_starts_from_what_auto_density_prints():
    from negpy.features.transparency.logic import transfer_shown_values

    before = _slide(density=1.0, auto_exposure=True)
    metrics = {"fade_tone_points": _ramp(0.3, 1.2), "metered_anchor": 0.4}
    shown = transfer_shown_values(before.exposure, metrics)["density"]
    none, _ = fade_tone_changes(before, FadeABTone.ADAPTIVE, metrics, 0.0)
    assert none.exposure.density == pytest.approx(shown, abs=0.006)
    assert not none.exposure.auto_exposure
