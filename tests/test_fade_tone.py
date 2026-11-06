"""Tone and Auto Red: an automatic tone for a faded slide, written into its own sliders."""

from dataclasses import replace

import numpy as np
import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.process.fade_tone import TONE_FIELDS, TONE_PCT, adaptive_values, fade_tone_changes, tone_points
from negpy.features.process.models import ProcessMode, mode_aware_exposure_reset


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


def test_tone_at_zero_restores_the_sliders_exactly():
    before = _slide(density=1.23, grade=101.0, auto_exposure=True, auto_normalize_contrast=True)
    written, reason = fade_tone_changes(before, {"fade_tone_points": _ramp(0.3, 1.6)}, 1.0)
    assert not reason and written.process.fade_tone_saved == tuple(getattr(before.exposure, f) for f in TONE_FIELDS)
    assert not written.exposure.auto_exposure and not written.exposure.auto_normalize_contrast
    restored, _ = fade_tone_changes(written, {}, 0.0)
    assert restored.exposure == before.exposure
    assert (restored.process.fade_tone_strength, restored.process.fade_tone_saved) == (0.0, None)


def test_tone_without_a_reading_changes_nothing():
    cfg, reason = fade_tone_changes(_slide(), {}, 0.5)
    assert cfg is None and reason


def test_tone_starts_at_zero():
    assert WorkspaceConfig().process.fade_tone_strength == 0.0


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
    from negpy.features.process.fade_tone import ADAPTIVE_SPAN

    # _ramp spreads the span over 0 to 100 %; the 0.5 to 99.5 % points span 0.99 of it.
    values = adaptive_values(_ramp(0.25, 0.25 + ADAPTIVE_SPAN / 0.99), positive_source=True)
    assert values["grade"] == 115.0


@pytest.mark.parametrize("span", [0.1, 0.4, 0.8, 1.2])
def test_adaptive_holds_the_grade_at_its_contrast_limit(span):
    from negpy.features.process.fade_tone import ADAPTIVE_CONTRAST
    from negpy.features.transparency.logic import TRANSFER_CONSTANTS

    floor = round(float(TRANSFER_CONSTANTS["transfer_grade_ref"]) / ADAPTIVE_CONTRAST[1])
    values = adaptive_values(_ramp(0.4, 0.4 + span), positive_source=True)
    assert values["grade"] >= floor


def test_adaptive_prints_the_median_at_its_midtone():
    from negpy.features.process.fade_tone import ADAPTIVE_MIDTONE
    from negpy.features.transparency.logic import TRANSFER_CONSTANTS

    c = TRANSFER_CONSTANTS
    pivot = float(c["transfer_contrast_pivot"])
    for points in (_ramp(0.3, 2.0), _ramp(0.6, 1.4)):
        values = adaptive_values(points, positive_source=True)
        offset = (1.0 - values["density"]) * float(c["transfer_density_stops"]) * np.log10(2.0)
        contrast = float(c["transfer_grade_ref"]) / values["grade"]
        assert pivot + (float(np.median(points)) - offset - pivot) * contrast == pytest.approx(ADAPTIVE_MIDTONE, abs=0.01)


def test_adaptive_never_lowers_the_contrast_of_a_wide_frame():
    assert adaptive_values(_ramp(0.1, 3.4), positive_source=True)["grade"] == 115.0


def test_strength_moves_the_sliders_from_the_saved_values_and_zero_restores():
    before = _slide(density=1.2, grade=110.0)
    metrics = {"fade_tone_points": _ramp(0.3, 1.2)}
    fit = adaptive_values(metrics["fade_tone_points"], positive_source=False)
    full, _ = fade_tone_changes(before, metrics, 1.0)
    assert (full.exposure.density, full.exposure.grade) == (fit["density"], fit["grade"])
    half, _ = fade_tone_changes(full, metrics, 0.5)
    assert half.process.fade_tone_strength == 0.5
    assert half.exposure.grade == pytest.approx(round(0.5 * (110.0 + fit["grade"])), abs=1.0)
    assert half.exposure.density == pytest.approx(0.5 * (1.2 + fit["density"]), abs=0.011)
    assert half.process.fade_tone_saved == full.process.fade_tone_saved
    restored, _ = fade_tone_changes(half, {}, 0.0)
    assert restored.exposure == before.exposure


def test_strength_starts_from_what_auto_density_prints():
    from negpy.features.transparency.logic import transfer_shown_values

    before = _slide(density=1.0, auto_exposure=True)
    metrics = {"fade_tone_points": _ramp(0.3, 1.2), "metered_anchor": 0.4}
    shown = transfer_shown_values(before.exposure, metrics)["density"]
    low, _ = fade_tone_changes(before, metrics, 0.01)
    fit = adaptive_values(metrics["fade_tone_points"], positive_source=False)["density"]
    assert low.exposure.density == pytest.approx(shown + 0.01 * (fit - shown), abs=0.006)
    assert not low.exposure.auto_exposure


def _scaled(points: tuple[float, ...], red: float, pivot: float = 0.0):
    """The points a frame read at *red* shows at another red: density scales by red / new red about *pivot*."""
    return lambda new: tuple(pivot + (p - pivot) * red / new for p in points)


def test_auto_red_scales_the_span_onto_the_adaptive_span_and_holds_its_bounds() -> None:
    from negpy.features.process.fade_tone import AUTO_RED_MIN, AUTO_RED_SPAN, auto_red

    def points(span: float, light: float = 0.3) -> tuple[float, ...]:
        return tuple(np.linspace(light, light + span, len(TONE_PCT)))

    changes, reason = auto_red(_scaled(points(1.2), 0.9), 0.9)
    assert reason == "" and changes["fade_ratio_r"] == pytest.approx(0.9 * 1.2 / AUTO_RED_SPAN, abs=0.006)
    assert auto_red(_scaled(points(0.3), 1.0), 1.0)[0]["fade_ratio_r"] == AUTO_RED_MIN
    assert auto_red(_scaled(points(2.0), 0.6), 0.6)[0]["fade_ratio_r"] == 0.75
    assert auto_red(_scaled(points(4.0), 1.0), 1.0)[0]["fade_ratio_r"] == 1.0
    assert auto_red(_scaled(points(0.01), 1.0), 1.0)[0] is None
    assert auto_red(lambda _red: None, 1.0)[0] is None


def test_auto_red_shifts_density_by_what_the_median_gains() -> None:
    """A span of 1.2 D sets red 0.75; the shift takes back what the median gains, on top of the shift the
    frame already had."""
    from negpy.features.process.fade_tone import auto_red

    points = tuple(np.linspace(1.0, 2.2, len(TONE_PCT)))
    gain = 1.6 / 0.75 - 1.6
    assert auto_red(_scaled(points, 1.0), 1.0) == ({"fade_ratio_r": 0.75, "fade_density_shift": round(-gain, 4)}, "")
    assert auto_red(_scaled(points, 1.0), 1.0, shift=0.1)[0]["fade_density_shift"] == pytest.approx(0.1 - gain, abs=1e-4)


def test_auto_red_keeps_the_light_end_off_white() -> None:
    """The shift stops where the 0.5 % point reaches AUTO_RED_LIGHT, or where it was when it was lighter."""
    from negpy.features.process.fade_tone import AUTO_RED_LIGHT, auto_red

    points = tuple(np.linspace(0.3, 1.5, len(TONE_PCT)))
    assert auto_red(_scaled(points, 1.0), 1.0)[0]["fade_density_shift"] == pytest.approx(AUTO_RED_LIGHT - 0.3 / 0.75, abs=1e-4)
    light = tuple(np.linspace(0.1, 1.3, len(TONE_PCT)))
    assert auto_red(_scaled(light, 1.0), 1.0)[0]["fade_density_shift"] == pytest.approx(0.1 - 0.1 / 0.75, abs=1e-4)
    centred = tuple(np.linspace(-0.6, 0.6, len(TONE_PCT)))
    assert auto_red(_scaled(centred, 1.0), 1.0)[0]["fade_density_shift"] == pytest.approx(0.6 / 0.75 - 0.6, abs=1e-4)


def test_the_density_shift_adds_density_to_every_channel_after_the_fade() -> None:
    from negpy.features.process.fade import fade_density
    from negpy.features.process.models import ProcessConfig, ProcessMode

    base = replace(ProcessConfig(), process_mode=ProcessMode.E6, fade_ratio_r=0.5)
    plain = fade_density(base, ProcessMode.E6).matrix
    shifted = fade_density(replace(base, fade_density_shift=0.3), ProcessMode.E6).matrix
    assert np.allclose(shifted[:, :3], plain[:, :3]) and np.allclose(shifted[:, 3], -0.3)
    assert fade_density(replace(base, process_mode=ProcessMode.C41, fade_density_shift=0.3), ProcessMode.C41).matrix is None
