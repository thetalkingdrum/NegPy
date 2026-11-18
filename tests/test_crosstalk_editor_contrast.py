"""The crosstalk editor's Channel Contrast fold: a Transparency profile's Stage and per-channel gain."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest
from PyQt6.QtWidgets import QApplication

from negpy.features.process.models import ProcessMode
from negpy.kernel.system.config import APP_CONFIG
from negpy.services.assets.crosstalk import CrosstalkProfiles

_IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
E6, C41 = str(ProcessMode.E6), str(ProcessMode.C41)
GAIN = (0.8749, 0.925, 1.0012)


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(tmp_path))
    monkeypatch.setattr("negpy.services.assets.crosstalk.get_resource_path", lambda _: str(tmp_path / "_none"))


def _dialog(select: str):
    from negpy.desktop.view.widgets.crosstalk_editor_dialog import CrosstalkEditorDialog

    return CrosstalkEditorDialog(select, 0.5, ProcessMode.E6)


def test_a_look_profile_loads_its_stage_and_gain(_app):
    CrosstalkProfiles.save("Look", _IDENTITY, process=E6, gain=GAIN, look=True)
    dlg = _dialog("Look")
    assert [s.value() for s in dlg._gain_sliders] == pytest.approx(list(GAIN))
    assert dlg.selected_look() and dlg.selected_gain() == GAIN
    assert not dlg.contrast_fold.content.isHidden()


def test_an_edited_contrast_and_stage_are_saved(_app):
    CrosstalkProfiles.save("Mine", _IDENTITY, process=E6)
    dlg = _dialog("Mine")
    assert dlg.selected_gain() is None and not dlg.selected_look()
    dlg._gain_sliders[0].setValue(0.9)
    dlg.stage_choice.setCurrentIndex(dlg.stage_choice.findData(True))
    dlg._on_save()
    assert CrosstalkProfiles.get_gain("Mine") == (0.9, 1.0, 1.0)
    assert CrosstalkProfiles.get_look("Mine")


def test_contrast_back_at_one_saves_no_gain(_app):
    CrosstalkProfiles.save("Mine", _IDENTITY, process=E6, gain=GAIN, look=True)
    dlg = _dialog("Mine")
    for s in dlg._gain_sliders:
        s.setValue(1.0)
    dlg._on_save()
    assert CrosstalkProfiles.get_gain("Mine") is None and CrosstalkProfiles.get_look("Mine")


def test_the_fold_is_for_transparency_only(_app):
    CrosstalkProfiles.save("Mine", _IDENTITY, process=E6, gain=GAIN, look=True)
    dlg = _dialog("Mine")
    dlg._set_process(C41)
    assert dlg.contrast_fold.content.isHidden()
    assert dlg.selected_gain() is None and not dlg.selected_look()
    dlg._on_save()
    assert CrosstalkProfiles.get_gain("Mine") is None and not CrosstalkProfiles.get_look("Mine")


def test_a_contrast_change_previews(_app):
    CrosstalkProfiles.save("Mine", _IDENTITY, process=E6)
    dlg = _dialog("Mine")
    seen = []
    dlg.matrix_previewed.connect(lambda *a: seen.append(dlg.selected_gain()))
    sld = dlg._gain_sliders[2]
    sld.setValue(1.1)
    sld.valueChanged.emit(1.1)  # a drag: setValue from code emits nothing
    assert seen and seen[-1] == (1.0, 1.0, 1.1)


def test_a_copy_keeps_the_edited_contrast(_app):
    CrosstalkProfiles.save("Mine", _IDENTITY, process=E6, gain=GAIN, look=True)
    dlg = _dialog("Mine")
    dlg._gain_sliders[1].setValue(0.95)
    dlg._on_copy()
    copy = dlg.selected_name()
    assert copy != "Mine"
    assert CrosstalkProfiles.get_gain(copy) == (GAIN[0], 0.95, GAIN[2]) and CrosstalkProfiles.get_look(copy)
