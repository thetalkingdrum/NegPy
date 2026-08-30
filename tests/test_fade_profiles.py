import os

import pytest

from negpy.kernel.system.config import APP_CONFIG
from negpy.services.assets.fade import FadeProfiles


def _write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


@pytest.fixture(autouse=True)
def _isolate_bundled(tmp_path, monkeypatch):
    monkeypatch.setattr("negpy.services.assets.fade.get_resource_path", lambda _: str(tmp_path / "_no_bundled"))


def test_profile_without_bands_is_rejected(tmp_path, monkeypatch):
    """Delta means nothing without knowing the scanner's measurement wavelengths -- a
    profile missing `bands` must not appear at all, the same way a malformed delta
    doesn't. This is also the fate of a profile saved before `bands` existed."""
    monkeypatch.setattr(APP_CONFIG, "fade_dir", str(tmp_path))
    _write(
        os.path.join(tmp_path, "no_bands.toml"),
        'name = "No Bands"\ndelta = [0.05, 0.01, 0.2, 0.04, 0.08, 0.18]\n',
    )
    assert FadeProfiles.list_profiles() == ["None"]
    assert FadeProfiles.get_delta("No Bands") is None


def test_profile_with_malformed_bands_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "fade_dir", str(tmp_path))
    _write(
        os.path.join(tmp_path, "wrong_length.toml"),
        'name = "Wrong Length"\nbands = [650, 550]\ndelta = [0.05, 0.01, 0.2, 0.04, 0.08, 0.18]\n',
    )
    _write(
        os.path.join(tmp_path, "not_numeric.toml"),
        'name = "Not Numeric"\nbands = ["r", "g", "b"]\ndelta = [0.05, 0.01, 0.2, 0.04, 0.08, 0.18]\n',
    )
    assert FadeProfiles.list_profiles() == ["None"]


def test_valid_profile_round_trips_delta_and_bands(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "fade_dir", str(tmp_path))
    _write(
        os.path.join(tmp_path, "ektachrome.toml"),
        'name = "Ektachrome"\nbands = [650, 550, 450]\ndelta = [0.0564, 0.0055, 0.1946, 0.0190, 0.0531, 0.1453]\n',
    )
    assert FadeProfiles.get_bands("Ektachrome") == (650.0, 550.0, 450.0)
    assert FadeProfiles.get_delta("Ektachrome") == (0.0564, 0.0055, 0.1946, 0.019, 0.0531, 0.1453)


def test_save_round_trips_bands(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "fade_dir", str(tmp_path))
    delta = [0.03, 0.02, 0.01, 0.04, 0.02, 0.03]
    bands = [640.0, 545.0, 460.0]  # a rig with different bands from Gschwind's canonical set

    path = FadeProfiles.save("My Rig", delta, bands)

    assert os.path.isfile(path)
    assert FadeProfiles.get_bands("My Rig") == (640.0, 545.0, 460.0)
    assert FadeProfiles.get_delta("My Rig") == tuple(delta)


def test_none_and_missing_return_no_bands(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "fade_dir", str(tmp_path))
    assert FadeProfiles.get_bands(FadeProfiles.NONE_NAME) is None
    assert FadeProfiles.get_bands("nonexistent") is None
