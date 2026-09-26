"""The Side Absorption sliders tune the fade profile's delta live on the frame.

A drag writes only its own term, since the sliders round and reading all six back would
round the rest. The sliders grey out while delta cannot move the render: F = S D inv(S) is
a scalar whenever green and blue survival match red, for any S.
"""

import sys
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.sidebar.sensor import SensorSidebar
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.normalization import fade_delta_inert_reason, resolve_fade_matrix
from negpy.features.process.models import ProcessMode
from negpy.kernel.system.config import APP_CONFIG
from negpy.services.assets.fade import FadeProfiles

if not QApplication.instance():
    _app = QApplication(sys.argv)

_DELTA = (0.088, 0.0254, 0.2481, 0.0404, 0.1471, 0.2073)
_NAME = "Test Stock"


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "fade_dir", str(tmp_path / "fade"))
    monkeypatch.setattr("negpy.services.assets.fade.get_resource_path", lambda _: str(tmp_path / "_none"))
    FadeProfiles.save(_NAME, list(_DELTA), process=ProcessMode.E6)


def _sidebar(**process):
    cfg = WorkspaceConfig()
    fields = dict(process_mode=ProcessMode.E6, fade_profile=_NAME, fade_delta=_DELTA, fade_ratio_g=0.8)
    fields.update(process)
    ctrl = MagicMock()
    ctrl.state.config = replace(cfg, process=replace(cfg.process, **fields))
    w = SensorSidebar(ctrl)
    w.sync_ui()
    w.fade_terms.header.setChecked(True)
    return w, ctrl


def _set(w, ctrl, **process):
    ctrl.state.config = replace(ctrl.state.config, process=replace(ctrl.state.config.process, **process))
    w.sync_ui()


def test_a_drag_writes_only_its_own_term():
    w, ctrl = _sidebar()
    w.fade_terms.sliders[2].valueCommitted.emit(0.3)
    kwargs = ctrl.set_roll_default.call_args.kwargs
    assert kwargs["fade_delta"] == (_DELTA[0], _DELTA[1], 0.3, *_DELTA[3:])
    assert kwargs["persist"] is True


def test_an_edit_away_from_the_profile_is_shown_and_can_be_reset():
    w, ctrl = _sidebar()
    assert w.fade_terms.note.text() == ""
    assert not w.fade_terms.reset_btn.isEnabled()

    _set(w, ctrl, fade_delta=(0.2, *_DELTA[1:]))
    assert w.fade_terms.note.text() == f"Edited from {_NAME}"
    assert w.fade_terms.reset_btn.isEnabled()

    w.fade_terms.reset_btn.click()
    assert ctrl.set_roll_default.call_args.kwargs["fade_delta"] == _DELTA


def test_the_edited_note_shows_with_the_sliders_hidden():
    w, ctrl = _sidebar(fade_delta=(0.2, *_DELTA[1:]))
    w.show()
    w.fade_terms.header.setChecked(False)
    assert not w.fade_terms.sliders[0].isVisible()
    assert w.fade_terms.note.isVisible()


def test_sliders_grey_out_while_the_layers_faded_equally():
    w, ctrl = _sidebar(fade_ratio_g=1.0, fade_ratio_b=1.0, fade_ratio_r=0.7)
    assert not w.fade_terms.sliders[0].isEnabled()
    assert "Green or Blue Survival" in w.fade_terms.gate_hint.text()

    _set(w, ctrl, fade_ratio_b=1.2)
    assert w.fade_terms.sliders[0].isEnabled()
    assert w.fade_terms.gate_hint.text() == ""


def test_save_as_profile_writes_the_tuned_delta_and_selects_it(monkeypatch):
    tuned = (0.2, *_DELTA[1:])
    w, ctrl = _sidebar(fade_delta=tuned)
    monkeypatch.setattr("negpy.desktop.view.sidebar.profile_terms.QInputDialog.getText", lambda *a, **k: ("Mine", True))
    w.fade_terms.save_btn.click()
    assert FadeProfiles.get_delta("Mine") == pytest.approx(tuned)
    assert ctrl.set_roll_default.call_args.kwargs["fade_profile"] == "Mine"


def test_save_refuses_a_bundled_or_none_name(monkeypatch):
    w, ctrl = _sidebar()
    warned = []
    monkeypatch.setattr("negpy.desktop.view.sidebar.profile_terms.QInputDialog.getText", lambda *a, **k: ("None", True))
    monkeypatch.setattr("negpy.desktop.view.sidebar.profile_terms.QMessageBox.warning", lambda *a, **k: warned.append(a))
    w.fade_terms.save_btn.click()
    assert warned
    ctrl.set_roll_default.assert_not_called()


@pytest.mark.parametrize(("ratio_g", "ratio_b"), [(1.0, 1.0), (0.8, 1.0), (1.0, 1.3), (0.7, 1.3)])
def test_the_inert_reason_matches_the_math(ratio_g, ratio_b):
    process = replace(WorkspaceConfig().process, fade_ratio_r=0.7, fade_ratio_g=ratio_g, fade_ratio_b=ratio_b, fade_delta=_DELTA)
    with_delta = resolve_fade_matrix(1.0, 0.7, ratio_g, ratio_b, _DELTA)
    without = resolve_fade_matrix(1.0, 0.7, ratio_g, ratio_b, None)
    assert bool(fade_delta_inert_reason(process, ProcessMode.E6)) == np.allclose(with_delta, without)


def test_an_active_crosstalk_profile_makes_delta_inert():
    process = replace(
        WorkspaceConfig().process,
        fade_ratio_g=0.8,
        fade_delta=_DELTA,
        crosstalk_strength=1.0,
        crosstalk_matrix=(1.0, -0.1, 0.0, 0.0, 1.0, -0.1, 0.0, 0.0, 1.0),
        crosstalk_process=ProcessMode.E6,
    )
    assert "Crosstalk" in fade_delta_inert_reason(process, ProcessMode.E6)
