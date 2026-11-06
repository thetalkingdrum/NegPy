"""The port of NegativeConverter's expired-film rescue color table (`nc_rescue`) against their own expiredRescue.js,
run under Node by experiments/fade_crossover/port_check and stored in tests/fixtures/nc_rescue_port.json."""

import json
from pathlib import Path

import numpy as np
import pytest

from nc_rescue_fixtures import CASES, PROBES, VARIANTS
from negpy.features.process import nc_rescue
from negpy.features.process.models import ProcessMode

_EXPECTED = json.loads((Path(__file__).parent / "fixtures" / "nc_rescue_port.json").read_text())["cases"]


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request):
    name = request.param
    img = CASES[name]()
    display = img.astype(np.float64) / 65535.0
    return name, img, display, nc_rescue.analyze(display), _EXPECTED[name]


def test_the_analysis_matches(case):
    _, _, _, ours, theirs = case
    t = theirs["analysis"]
    assert ours.samples == t["samples"]
    for field, key in (("lum_low", "lumLow"), ("lum_high", "lumHigh"), ("lum_median", "lumMedian"), ("leveled_median", "leveledMedian")):
        assert getattr(ours, field) == pytest.approx(t[key], abs=1e-4), field
    assert ours.lean == pytest.approx(tuple(t["lean"]), abs=2e-4)
    assert ours.neutral_share == pytest.approx(t["neutralShare"], abs=1e-4)
    for mine, band in zip(ours.bands, t["bands"]):
        assert (mine is None) == (band is None)
        if band is not None:
            assert mine.lum == pytest.approx(band["lum"], abs=2e-4)
            assert mine.mean == pytest.approx(tuple(band["mean"]), abs=2e-4)
            assert mine.share == pytest.approx(band["share"], abs=2e-4)


def test_the_color_table_matches_within_one_level(case):
    _, img, display, ours, theirs = case
    # The port keeps their color table alone: the variant with no tone curve.
    variant = next(v for v, (colour, tone) in VARIANTS.items() if colour and not tone)
    expected = theirs["variants"][variant]
    offsets = nc_rescue.offsets_from_bands(ours)
    assert (offsets is None) == (expected["offsets"] is None)
    if offsets is not None:
        assert np.abs(offsets.ravel() - np.asarray(expected["offsets"])).max() < 0.5
    curve = nc_rescue.tone_curve(ours, nc_rescue.IDENTITY_TONE)
    assert np.abs(curve - np.asarray(expected["tone"])).max() < 0.5
    values = np.array([[img[y, x, c] * 255.0 / 65535.0 for c in range(3)] for y, x in PROBES])
    out = nc_rescue.apply_tone(values, offsets, curve)
    assert np.abs(out - np.asarray(expected["probes"])).max() < 1.0
    whole = nc_rescue.rescue(display)
    probes = np.array([whole[y, x] for y, x in PROBES]) * 255.0
    assert np.abs(probes - np.asarray(expected["probes"])).max() < 1.0


def _crossover_case():
    from fade_crossover_scene import CROSSOVER, build_scene, estimate, scan_image

    image = scan_image(CROSSOVER, build_scene(192, 144))
    return image, estimate(image)


def _render(image, **changes):
    from fade_crossover_scene import render

    return render(image, changes)


def test_the_nc_model_renders_the_scan_without_survivals_or_stain():
    from dataclasses import replace

    from negpy.features.process.fade import fade_matrix, remove_stain
    from negpy.features.process.models import ProcessConfig

    process = replace(ProcessConfig(), process_mode=ProcessMode.E6, fade_ratio_g=0.7, fade_white_anchor=True, fade_ab_model="nc")
    assert fade_matrix(process) is None
    assert not remove_stain(process)
    assert fade_matrix(replace(process, fade_ab_model="negpy")) is not None


def test_the_nc_model_moves_the_render():
    image, est = _crossover_case()
    survivals = dict(fade_ratio_r=est.red, fade_ratio_g=est.green, fade_ratio_b=est.blue)
    assert np.abs(_render(image, **survivals, fade_ab_model="nc") - _render(image, **survivals)).max() > 0.05


def test_the_defaults_render_as_before():
    """NegPy everywhere: a render with the A/B fields at their defaults equals one without them."""
    image, est = _crossover_case()
    survivals = dict(fade_ratio_r=est.red, fade_ratio_g=est.green, fade_ratio_b=est.blue)
    plain = _render(image, **survivals)
    explicit = _render(image, **survivals, fade_neutrals_mix=0.0, fade_ab_model="negpy", fade_crossover=0.0)
    assert np.array_equal(plain, explicit)


def test_the_nc_stages_route_to_the_cpu():
    from dataclasses import replace

    from negpy.domain.models import WorkspaceConfig
    from negpy.services.rendering.image_processor import ImageProcessor

    slide = replace(WorkspaceConfig(), process=replace(WorkspaceConfig().process, process_mode=ProcessMode.E6))
    assert not ImageProcessor._cpu_only(slide)
    assert ImageProcessor._cpu_only(replace(slide, process=replace(slide.process, fade_ab_model="nc")))
    assert not ImageProcessor._cpu_only(replace(slide, process=replace(slide.process, fade_crossover=1.0)))
    negative = replace(slide, process=replace(slide.process, process_mode=ProcessMode.C41, fade_ab_model="nc"))
    assert not ImageProcessor._cpu_only(negative)


def test_nc_neutrals_read_the_survivals_too():
    from fade_crossover_scene import CROSSOVER, build_scene, estimate_grid, scan_image
    from negpy.features.process.fade import estimate_fade

    image = scan_image(CROSSOVER, build_scene())
    grid = estimate_grid(image)
    ours = estimate_fade(grid)
    theirs = estimate_fade(grid, mix=1.0)
    assert theirs.red is not None and theirs.reason == ""
    assert theirs.blue < 0.95
    assert (theirs.green, theirs.blue) != (ours.green, ours.blue)
