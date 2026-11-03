"""Fade Estimate's gray readers (Compare): NegPy's, No Sky (NegPy's without its sky sector), NC's, and the
Mix between NegPy's or No Sky's and NC's."""

import numpy as np
import pytest

from fade_crossover_scene import CROSSOVER, REFERENCE, build_scene, estimate_grid, linear_fade, plustek, scan_image, scan_warm, warm_scene
from negpy.features.process.fade import EstimateMethod, estimate_fade, gray_refs, nc_neutral_refs, neutral_axis_from_grid
from negpy.features.process.models import FadeABNeutrals, ProcessConfig


@pytest.fixture(scope="module")
def grid():
    return estimate_grid(scan_image(CROSSOVER, build_scene(192, 144)))


@pytest.fixture(scope="module")
def sun_grid():
    """Gray objects in warm sun and cool shade on a broadband scan: the sky sector drops the shade."""
    return estimate_grid(scan_warm(plustek(CROSSOVER), warm_scene(2.0, layout="sun")), positive_source=True)


def test_mix_ends_are_the_two_readers_exactly(grid):
    assert gray_refs(grid, FadeABNeutrals.NEGPY, 0.0) == neutral_axis_from_grid(grid)
    assert gray_refs(grid, FadeABNeutrals.NEGPY, 1.0) == nc_neutral_refs(grid)
    assert gray_refs(grid, FadeABNeutrals.NC, 0.0) == nc_neutral_refs(grid)


def test_mix_moves_each_ref_in_log(grid):
    ours, theirs = neutral_axis_from_grid(grid)[0], nc_neutral_refs(grid)[0]
    half, reason = gray_refs(grid, FadeABNeutrals.NEGPY, 0.5)
    assert reason == ""
    for k in (0, 1):
        assert np.allclose(half[k], 0.5 * (np.asarray(ours[k]) + np.asarray(theirs[k])))


def test_mix_moves_the_estimate_between_the_two(grid):
    ends = [estimate_fade(grid, mix=m)[EstimateMethod.MIDTONE] for m in (0.0, 0.5, 1.0)]
    lo, hi = sorted((ends[0].blue, ends[2].blue))
    assert ends[0].blue != ends[2].blue and lo < ends[1].blue < hi


def test_no_sky_keeps_the_shade_the_sky_sector_drops(sun_grid):
    ours, _ = gray_refs(sun_grid, FadeABNeutrals.NEGPY)
    no_sky, _ = gray_refs(sun_grid, FadeABNeutrals.NO_SKY)
    # The shade's cool grays pull the midtone toward blue, where the sky sector left only sunlit grays.
    lean = lambda refs: refs[0][2] - 0.5 * (refs[0][0] + refs[0][1])  # noqa: E731
    assert lean(no_sky) > lean(ours) + 0.01


def test_no_sky_reads_a_frame_without_sky_sector_cells_as_negpy_does():
    g = estimate_grid(scan_image(linear_fade(REFERENCE, (1.0, 0.75, 0.55)), build_scene(192, 144)))
    assert gray_refs(g, FadeABNeutrals.NO_SKY)[0][:2] == gray_refs(g, FadeABNeutrals.NEGPY)[0][:2]


def test_the_mix_defaults_to_negpy_s_reader():
    assert ProcessConfig().fade_neutrals_mix == 0.0
    assert ProcessConfig(fade_ab_neutrals="no_sky").fade_ab_neutrals is FadeABNeutrals.NO_SKY
