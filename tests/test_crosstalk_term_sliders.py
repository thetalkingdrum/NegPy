"""The Crosstalk Matrix Terms sliders tune the matrix's six off-diagonal terms live.

A drag writes only its own term and leaves the diagonal alone. The built-in default
matrix is stored as None, so its terms come from DEFAULT_CROSSTALK_MATRIX.
"""

import sys
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.sidebar.sensor import SensorSidebar
from negpy.domain.models import WorkspaceConfig
from negpy.features.exposure.normalization import crosstalk_terms_inert_reason
from negpy.features.process.models import DEFAULT_CROSSTALK_MATRIX, ProcessMode
from negpy.kernel.system.config import APP_CONFIG
from negpy.services.assets.crosstalk import CrosstalkProfiles

if not QApplication.instance():
    _app = QApplication(sys.argv)

_MATRIX = (1.0, -0.1, -0.02, -0.04, 1.0, -0.08, -0.01, -0.12, 1.0)
_NAME = "Test Stock"


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(tmp_path / "ct"))
    monkeypatch.setattr("negpy.services.assets.crosstalk.get_resource_path", lambda _: str(tmp_path / "_none"))
    CrosstalkProfiles.save(_NAME, list(_MATRIX), process=ProcessMode.C41)


def _sidebar(**process):
    cfg = WorkspaceConfig()
    fields = dict(
        process_mode=ProcessMode.C41,
        crosstalk_profile=_NAME,
        crosstalk_matrix=_MATRIX,
        crosstalk_process=ProcessMode.C41,
        crosstalk_strength=0.8,
    )
    fields.update(process)
    ctrl = MagicMock()
    ctrl.state.config = replace(cfg, process=replace(cfg.process, **fields))
    w = SensorSidebar(ctrl)
    w.sync_ui()
    w.crosstalk_terms.header.setChecked(True)
    return w, ctrl


def _set(w, ctrl, **process):
    ctrl.state.config = replace(ctrl.state.config, process=replace(ctrl.state.config.process, **process))
    w.sync_ui()


def test_sliders_show_the_off_diagonal_terms():
    w, _ = _sidebar()
    assert [s.value() for s in w.crosstalk_terms.sliders] == pytest.approx([-0.1, -0.02, -0.04, -0.08, -0.01, -0.12])


def test_a_drag_writes_only_its_own_term_and_keeps_the_diagonal():
    w, ctrl = _sidebar()
    w.crosstalk_terms.sliders[3].valueCommitted.emit(-0.25)  # Blue in Green: row 1, column 2
    written = ctrl.set_roll_default.call_args.kwargs["crosstalk_matrix"]
    expected = list(_MATRIX)
    expected[5] = -0.25
    assert written == tuple(expected)


def test_editing_the_default_matrix_starts_from_the_built_in_one():
    w, ctrl = _sidebar(crosstalk_profile=CrosstalkProfiles.DEFAULT_NAME, crosstalk_matrix=None)
    w.crosstalk_terms.sliders[0].valueCommitted.emit(-0.2)
    written = ctrl.set_roll_default.call_args.kwargs["crosstalk_matrix"]
    assert written == (DEFAULT_CROSSTALK_MATRIX[0], -0.2, *DEFAULT_CROSSTALK_MATRIX[2:])


def test_an_edit_is_shown_and_reset_restores_the_profile():
    w, ctrl = _sidebar()
    assert w.crosstalk_terms.note.text() == ""
    edited = list(_MATRIX)
    edited[1] = -0.3
    _set(w, ctrl, crosstalk_matrix=tuple(edited))
    assert w.crosstalk_terms.note.text() == f"Edited from {_NAME}"
    w.crosstalk_terms.reset_btn.click()
    assert ctrl.set_roll_default.call_args.kwargs["crosstalk_matrix"] == pytest.approx(_MATRIX)


def test_sliders_grey_out_at_zero_strength():
    w, ctrl = _sidebar(crosstalk_strength=0.0)
    assert not w.crosstalk_terms.sliders[0].isEnabled()
    assert "Strength" in w.crosstalk_terms.gate_hint.text()
    _set(w, ctrl, crosstalk_strength=0.5)
    assert w.crosstalk_terms.sliders[0].isEnabled()


def test_the_group_is_hidden_on_bw():
    w, ctrl = _sidebar()
    w.show()
    assert w.crosstalk_terms.header.isVisible()
    _set(w, ctrl, process_mode=ProcessMode.BW)
    assert not w.crosstalk_terms.header.isVisible()
    assert not w.crosstalk_terms.sliders[0].isVisible()


def test_save_writes_the_tuned_matrix_for_its_process(monkeypatch):
    edited = list(_MATRIX)
    edited[1] = -0.3
    w, ctrl = _sidebar(crosstalk_matrix=tuple(edited))
    monkeypatch.setattr("negpy.desktop.view.sidebar.profile_terms.QInputDialog.getText", lambda *a, **k: ("Mine", True))
    w.crosstalk_terms.save_btn.click()
    assert CrosstalkProfiles.get_matrix("Mine") == pytest.approx(edited)
    assert CrosstalkProfiles.get_process("Mine") == str(ProcessMode.C41)
    assert ctrl.set_roll_default.call_args.kwargs["crosstalk_profile"] == "Mine"


def test_a_matrix_for_another_process_is_inert():
    process = replace(WorkspaceConfig().process, crosstalk_strength=1.0, crosstalk_matrix=_MATRIX, crosstalk_process=ProcessMode.E6)
    assert crosstalk_terms_inert_reason(process, ProcessMode.C41)
    assert not crosstalk_terms_inert_reason(process, ProcessMode.E6)
