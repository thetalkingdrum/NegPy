"""A Transparency crosstalk profile's stage: a separation profile acts before Fade Restoration, a look
profile (`stage = "look"`) after it, and Fade's white and Estimate read the scan before the look."""

import os
from dataclasses import replace

import numpy as np

from negpy.features.exposure.normalization import effective_crosstalk_matrix, look_crosstalk_matrix, separation_crosstalk_matrix
from negpy.features.process.fade import (
    fade_density,
    fade_matrix,
    fade_ratios_from_neutral_axis,
    fade_white_stale,
    look_neutral,
    render_white_space,
)
from negpy.features.process.models import ProcessConfig, ProcessMode
from negpy.kernel.system.config import APP_CONFIG
from negpy.services.assets.crosstalk import CrosstalkProfiles

E6 = str(ProcessMode.E6)
MATRIX = (1.0, 0.89, -0.2, 0.04, 1.0, -0.06, 0.0, -0.02, 1.0)
GAIN = (0.81, 0.92, 1.04)
SURVIVALS = dict(fade_ratio_r=0.9, fade_ratio_g=0.85, fade_ratio_b=0.7)


def _slide(look: bool, **changes) -> ProcessConfig:
    return replace(
        ProcessConfig(
            process_mode=ProcessMode.E6,
            crosstalk_strength=1.0,
            crosstalk_matrix=MATRIX,
            crosstalk_process=E6,
            crosstalk_gain=GAIN,
            crosstalk_look=look,
        ),
        **changes,
    )


def _apply(density, d: np.ndarray) -> np.ndarray:
    m = np.asarray(density.matrix, dtype=np.float64)
    if m.shape[1] == 3:
        return m @ d
    anchor = np.zeros(3) if density.anchor is None else np.asarray(density.anchor)
    return m[:, :3] @ (d - anchor) + m[:, 3] + m[:, :3] @ anchor


def test_look_loads_only_for_transparency_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(tmp_path))
    body = "matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]\n"
    for fname, head in (
        ("look.toml", 'name = "Look"\nprocess = "Transparency"\nstage = "look"\n'),
        ("sep.toml", 'name = "Sep"\nprocess = "Transparency"\n'),
        ("neg.toml", 'name = "Neg"\nstage = "look"\n'),
    ):
        with open(os.path.join(tmp_path, fname), "w") as fh:
            fh.write(head + body)
    assert CrosstalkProfiles.get_look("Look") is True
    assert CrosstalkProfiles.get_look("Sep") is False
    assert CrosstalkProfiles.get_look("Neg") is False


def test_fade_acts_after_a_separation_profile_and_before_a_look_profile():
    m = effective_crosstalk_matrix(_slide(False), E6)
    sep = _slide(False, **SURVIVALS)
    look = _slide(True, **SURVIVALS)
    f = fade_matrix(sep)
    np.testing.assert_allclose(fade_density(sep, E6).matrix, f @ m)
    np.testing.assert_allclose(fade_density(look, E6).matrix, m @ f)
    assert separation_crosstalk_matrix(look, E6) is None
    assert look_crosstalk_matrix(sep, E6) is None


def test_a_look_profile_on_a_negative_path_is_a_plain_profile():
    neg = replace(_slide(True), process_mode=ProcessMode.C41, crosstalk_process=str(ProcessMode.C41))
    assert look_crosstalk_matrix(neg, str(ProcessMode.C41)) is None
    assert separation_crosstalk_matrix(neg, str(ProcessMode.C41)) is not None


def test_a_look_profile_leaves_the_white_space_alone():
    cam_xyz = [[0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [0.05, 0.1, 0.85]]
    wb = [2.0, 1.0, 1.5]
    full = render_white_space(_slide(True), E6, cam_xyz, wb)
    half = render_white_space(_slide(True, crosstalk_strength=0.5), E6, cam_xyz, wb)
    assert full == half == render_white_space(replace(_slide(True), crosstalk_strength=0.0), E6, cam_xyz, wb)
    assert render_white_space(_slide(False), E6, cam_xyz, wb) != render_white_space(_slide(False, crosstalk_strength=0.5), E6, cam_xyz, wb)


def test_a_look_gain_on_a_camera_raw_keeps_the_white_neutral():
    white = dict(fade_white_r=-0.2, fade_white_g=-0.35, fade_white_b=-0.3, fade_white_space="raw")
    for strength in (0.25, 0.5, 1.0):
        process = _slide(True, crosstalk_strength=strength, **white)
        density = fade_density(process, E6, "raw")
        out = _apply(density, np.full(3, -0.2))
        np.testing.assert_allclose(out, out[0], atol=1e-9)
    # A TIFF's own white is full scale: the gain scales density about 0.
    assert fade_density(_slide(True), E6, "").anchor is None


def test_a_look_gain_needs_a_measured_white_and_a_plain_profile_does_not():
    plain = replace(_slide(True), crosstalk_gain=None)
    assert fade_white_stale(_slide(True), "raw")
    assert not fade_white_stale(plain, "raw")
    assert not fade_white_stale(_slide(False, crosstalk_gain=None), "raw")


def test_estimate_reads_an_unfaded_ramp_as_unfaded_through_a_look_profile():
    look = look_crosstalk_matrix(_slide(True), E6)
    u = look_neutral(look)
    assert u is not None
    np.testing.assert_allclose(look @ u, 1.0)

    def refs(d):
        return (tuple(-float(v) for v in d), (0.0, 0.0, 0.0), None, 1.0)

    g, b, why = fade_ratios_from_neutral_axis(refs(1.2 * u), look)
    assert (g, b) == (1.0, 1.0) and why
    f = np.array([1.0, 0.85, 0.7])
    g, b, why = fade_ratios_from_neutral_axis(refs(1.2 * f * u), look)
    assert why == ""
    np.testing.assert_allclose((g, b), (0.85, 0.7))
    # Without the look the same scan reads as faded: the profile's own tilt.
    g, b, _ = fade_ratios_from_neutral_axis(refs(1.2 * u))
    assert abs(g - 1.0) > 0.05


def test_a_faded_ramp_renders_neutral_after_its_estimate():
    process = _slide(True, fade_ratio_r=1.0, fade_ratio_g=0.85, fade_ratio_b=0.7)
    look = look_crosstalk_matrix(process, E6)
    ramp = [s * np.array([1.0, 0.85, 0.7]) * look_neutral(look) for s in (0.3, 1.0, 2.0)]
    density = fade_density(process, E6)
    for d in ramp:
        out = _apply(density, -d)
        np.testing.assert_allclose(out, out[0], rtol=1e-9)


def test_a_look_profile_without_a_neutral_refuses_to_estimate():
    singular = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    refs = ((-1.0, -0.8, -0.7), (0.0, 0.0, 0.0), None, 1.0)
    assert look_neutral(singular) is None
    g, b, why = fade_ratios_from_neutral_axis(refs, singular)
    assert (g, b) == (1.0, 1.0) and why
    assert fade_ratios_from_neutral_axis(refs)[2] == ""


def test_the_noise_floor_reads_the_raw_densities():
    look = np.diag([1.0 / 0.01, 1.0, 1.0])  # u_r = 0.01: a flat raw red reads large after the look
    refs = ((-0.015, -1.0, -1.0), (0.0, 0.0, 0.0), None, 1.0)
    g, b, why = fade_ratios_from_neutral_axis(refs, look)
    assert (g, b) == (1.0, 1.0) and "flat" in why


def test_survivals_go_stale_when_the_look_changes():
    from negpy.features.process.fade import fade_survivals_stale, look_tag

    process = _slide(True, **SURVIVALS)
    process = replace(process, fade_survival_look=look_tag(process, E6))
    assert not fade_survivals_stale(process)
    assert fade_survivals_stale(replace(process, crosstalk_strength=0.5))
    assert fade_survivals_stale(replace(process, crosstalk_look=False))
    assert fade_survivals_stale(replace(process, crosstalk_gain=(0.85, 0.92, 1.04)))
    assert not fade_survivals_stale(replace(process, fade_ratio_g=1.0, fade_ratio_b=1.0))
    # No look profile: an existing edit carries no tag and is never stale.
    plain = _slide(False, **SURVIVALS)
    assert look_tag(plain, E6) == "" and not fade_survivals_stale(plain)


def test_the_editor_save_keeps_gain_and_stage(tmp_path, monkeypatch):
    import tomllib

    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(tmp_path))
    path = CrosstalkProfiles.save("Look", list(MATRIX), process=E6, gain=GAIN, look=True)
    data = tomllib.load(open(path, "rb"))
    assert data["stage"] == "look" and tuple(data["gain"]) == GAIN
    assert CrosstalkProfiles.get_gain("Look") == GAIN and CrosstalkProfiles.get_look("Look")
    neg = tomllib.load(open(CrosstalkProfiles.save("Neg", list(MATRIX), process=str(ProcessMode.C41), gain=GAIN, look=True), "rb"))
    assert "gain" not in neg and "stage" not in neg
    plain = tomllib.load(open(CrosstalkProfiles.save("Plain", list(MATRIX), process=E6, gain=(1.0, 1.0, 1.0)), "rb"))
    assert "gain" not in plain and "stage" not in plain


def test_keeps_neutral_only_while_a_gain_or_fade_acts():
    from negpy.features.process.fade import _chain, _keeps_neutral

    gallery = os.path.join(os.path.dirname(__file__), "..", "crosstalk")
    matrices = [CrosstalkProfiles._parse_file(os.path.join(gallery, f))[1] for f in os.listdir(gallery) if f.endswith(".toml")]
    for flat in matrices + [list(MATRIX)]:
        for step in range(101):
            s = step / 100.0
            base = _slide(True, crosstalk_matrix=tuple(flat), crosstalk_strength=s, crosstalk_gain=None)
            assert _keeps_neutral(look_crosstalk_matrix(base, E6)), (flat, s)
            gained = look_crosstalk_matrix(replace(base, crosstalk_gain=GAIN), E6)
            assert _keeps_neutral(gained) == (s == 0.0), (flat, s)
            faded = _chain(look_crosstalk_matrix(base, E6), fade_matrix(_slide(True, **SURVIVALS)))
            assert not _keeps_neutral(faded)


def test_remove_stain_on_a_tiff_removes_a_scan_stain_under_a_look_gain():
    stain = np.array([-0.02, -0.06, -0.15])
    clean_white = np.zeros(3)
    rng = np.random.default_rng(1)
    pixels = -rng.uniform(0.1, 2.5, (50, 3))

    def render(white, offset):
        w = dict(fade_white_r=white[0], fade_white_g=white[1], fade_white_b=white[2], fade_white_anchor=True)
        density = fade_density(_slide(True, **SURVIVALS, **w), E6, "")
        return np.array([_apply(density, p + offset) for p in pixels])

    stained = render(clean_white + stain, stain)
    clean = render(clean_white - 1e-6, np.zeros(3))
    diff = stained - clean
    np.testing.assert_allclose(diff, diff[0, 0], atol=1e-5)


def test_separation_and_look_together_keep_the_white_neutral():
    from negpy.features.process.fade import _density_of

    unmix = effective_crosstalk_matrix(replace(_slide(False), crosstalk_gain=None), E6)
    look = look_crosstalk_matrix(_slide(True), E6)
    w = np.array([-0.2, -0.35, -0.3])
    process = _slide(True, **SURVIVALS, fade_white_r=w[0], fade_white_g=w[1], fade_white_b=w[2], fade_white_space="raw")
    density = _density_of(process, "raw", unmix, look)
    scan_white = np.linalg.solve(unmix, np.full(3, w.max()))
    out = _apply(density, scan_white)
    np.testing.assert_allclose(out, out[0], atol=1e-9)


def _ramp_grid(direction: np.ndarray, fade: float) -> np.ndarray:
    """A 64×64 log grid of grays along *direction*, faded by *fade*, whose unfaded 0.5 to 99.5 % span
    is AUTO_RED_SPAN."""
    from negpy.features.process.fade_auto_red import AUTO_RED_SPAN

    s = np.linspace(0.2, 0.2 + AUTO_RED_SPAN / 0.99, 64 * 64).reshape(64, 64, 1)
    return (-fade * s * direction).astype(np.float32)


def test_restore_range_reads_a_uniform_fade_alike_through_a_look_profile():
    from negpy.features.process.fade_estimate import auto_red_changes

    look_conf = _slide(True)
    u = look_neutral(look_crosstalk_matrix(look_conf, E6))
    plain_conf = _slide(False, crosstalk_gain=None)
    reds = []
    for conf, direction in ((look_conf, u), (plain_conf, np.ones(3))):
        changes, why = auto_red_changes(conf, None, None, _ramp_grid(direction, 0.8))
        assert why == "", why
        reds.append(changes["fade_ratio_r"])
    np.testing.assert_allclose(reds, 0.8, atol=0.02)
    assert reds[0] == reds[1]
