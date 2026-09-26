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


def test_profile_without_bands_loads(tmp_path, monkeypatch):
    """A profile tuned by eye on a broadband rig has no wavelengths to record."""
    monkeypatch.setattr(APP_CONFIG, "fade_dir", str(tmp_path))
    _write(
        os.path.join(tmp_path, "no_bands.toml"),
        'name = "No Bands"\ntype = "tuned"\ndelta = [0.05, 0.01, 0.2, 0.04, 0.08, 0.18]\n',
    )
    assert FadeProfiles.list_profiles() == ["None", "No Bands"]
    assert FadeProfiles.get_delta("No Bands") == (0.05, 0.01, 0.2, 0.04, 0.08, 0.18)
    assert FadeProfiles.get_bands("No Bands") is None


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


def test_save_without_bands_writes_none(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "fade_dir", str(tmp_path))

    path = FadeProfiles.save("Broadband Rig", [0.1, 0.05, 0.3, 0.08, 0.12, 0.25])

    with open(path, encoding="utf-8") as f:
        assert "bands" not in f.read()
    assert FadeProfiles.get_bands("Broadband Rig") is None
    assert FadeProfiles.get_type("Broadband Rig") == "tuned"


def test_editor_records_bands_only_for_a_measured_or_spec_sheet_profile(tmp_path, monkeypatch, qapp):
    from negpy.desktop.view.widgets.fade_editor_dialog import FadeEditorDialog
    from negpy.services.assets.crosstalk import CrosstalkType

    monkeypatch.setattr(APP_CONFIG, "fade_dir", str(tmp_path))
    FadeProfiles.save("Rig", [0.0] * 6)
    dlg = FadeEditorDialog("Rig", 1.0)

    assert dlg.bands_row.isHidden()
    assert dlg.working_bands() is None

    dlg._set_type(CrosstalkType.MEASURED)
    assert not dlg.bands_row.isHidden()
    assert dlg.working_bands() == [650.0, 550.0, 450.0]

    dlg._on_save()
    assert FadeProfiles.get_bands("Rig") == (650.0, 550.0, 450.0)


def test_editor_shows_a_bundled_profiles_bands_whatever_its_type(tmp_path, monkeypatch, qapp):
    from negpy.desktop.view.widgets.fade_editor_dialog import FadeEditorDialog

    bundled = tmp_path / "_bundled"
    bundled.mkdir()
    monkeypatch.setattr("negpy.services.assets.fade.get_resource_path", lambda _: str(bundled))
    monkeypatch.setattr(APP_CONFIG, "fade_dir", str(tmp_path))
    _write(
        os.path.join(bundled, "generic.toml"),
        'name = "Generic"\ntype = "built-in"\nbands = [650, 550, 450]\ndelta = [0, 0, 0, 0, 0, 0]\n',
    )

    dlg = FadeEditorDialog("Generic", 1.0)

    assert not dlg.bands_row.isHidden()
