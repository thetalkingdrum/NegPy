"""Scanner zero (Compare): the log point a scanner file's fade scales density about."""

from dataclasses import replace

import numpy as np

from fade_crossover_scene import REFERENCE, build_scene, estimate, linear_fade, plustek, render, scan_image, survival_changes
from negpy.features.process.fade import (
    crossover_lean_fit,
    crossover_space,
    density_anchor,
    fade_density,
    render_white_space,
    scanner_zero,
)
from negpy.features.process.models import ProcessConfig, ProcessMode

_CAM = [[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.0, 0.1, 0.9]]
_WB = [2.0, 1.0, 1.5, 1.0]


def _slide(**fields) -> ProcessConfig:
    return replace(ProcessConfig(), process_mode=ProcessMode.E6, positive_source=True, **fields)


def test_full_scale_keeps_the_scanner_file_s_empty_white_space():
    assert render_white_space(_slide(), ProcessMode.E6, None, None) == ""


def test_a_clear_zero_tags_the_white_space_with_the_zero():
    for zero in ("clear_film", "clear_neutral"):
        space = render_white_space(_slide(fade_ab_scanner_zero=zero), ProcessMode.E6, None, None)
        assert space and scanner_zero(space) == zero


def test_a_camera_raw_keeps_its_own_anchor_whatever_the_zero():
    raw = render_white_space(_slide(), ProcessMode.E6, _CAM, _WB)
    assert raw and scanner_zero(raw) is None
    assert render_white_space(_slide(fade_ab_scanner_zero="clear_film"), ProcessMode.E6, _CAM, _WB) == raw


def test_each_zero_names_its_anchor():
    white = (-0.4, -0.5, -0.7)
    film = render_white_space(_slide(fade_ab_scanner_zero="clear_film"), ProcessMode.E6, None, None)
    neutral = render_white_space(_slide(fade_ab_scanner_zero="clear_neutral"), ProcessMode.E6, None, None)
    assert density_anchor(white, False, film) == white
    assert np.allclose(density_anchor(white, False, neutral), (-1.6 / 3.0,) * 3)
    assert density_anchor(white, False, "abc123") == (-0.4, -0.4, -0.4)
    assert density_anchor(white, True, neutral) == white


def test_a_scanner_zero_change_measures_the_table_again():
    process = _slide(fade_ratio_g=0.8, fade_white_r=-0.4, fade_white_g=-0.45, fade_white_b=-0.5)
    tags = set()
    for zero in ("full_scale", "clear_film", "clear_neutral"):
        p = replace(process, fade_ab_scanner_zero=zero)
        space = render_white_space(p, ProcessMode.E6, None, None)
        p = replace(p, fade_white_space=space)
        tags.add(crossover_space(fade_density(p, ProcessMode.E6, space), space, p))
    assert len(tags) == 3


def test_the_scanner_white_stays_per_frame():
    from negpy.services.assets.rolls import frame_roll_defaults

    space = render_white_space(_slide(fade_ab_scanner_zero="clear_film"), ProcessMode.E6, None, None)
    process = _slide(fade_white_r=-0.4, fade_white_g=-0.45, fade_white_b=-0.5, fade_white_space=space)
    defaults = {"fade_ratio_g": 0.8, "fade_white_r": -0.1, "fade_white_g": -0.1, "fade_white_b": -0.1, "fade_white_space": ""}
    assert "fade_white_r" not in frame_roll_defaults(defaults, process)


def test_a_clear_film_zero_removes_the_linear_lean_of_a_dim_white():
    """A broadband scan with clear film 1.3 stops under full scale: scaled about full scale the fade
    leaves a lean that grows with density; about the clear film it leaves almost none."""
    image = scan_image(plustek(linear_fade(REFERENCE, (1.0, 0.7, 0.5))), build_scene())
    full = estimate(image, positive_source=True)
    film = estimate(image, positive_source=True, fade_ab_scanner_zero="clear_film")
    assert np.abs(crossover_lean_fit(full.crossover)[0]).max() > 0.04
    assert np.abs(crossover_lean_fit(film.crossover)[0]).max() < 0.01


def test_a_scanner_file_at_full_scale_renders_as_before():
    image = scan_image(plustek(linear_fade(REFERENCE, (1.0, 0.7, 0.5))), build_scene())
    est = estimate(image, positive_source=True)
    changes = survival_changes(est, 0.0, positive_source=True)
    assert np.array_equal(render(image, changes), render(image, {**changes, "fade_ab_scanner_zero": "full_scale"}))
