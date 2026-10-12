import pathlib
from unittest.mock import MagicMock

import pytest
from conftest import FakeController, FakeRepo

import negpy.desktop.view.sidebar.flatfield as flatfield_module
from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.export import ExportSidebar
from negpy.desktop.view.sidebar.flatfield import FlatFieldSidebar
from negpy.desktop.view.sidebar.geometry import GeometrySidebar
from negpy.services.assets.flatfield import FlatFieldProfiles

NEGPY_ROOT = pathlib.Path(__file__).resolve().parent.parent / "negpy"


def test_no_override_cursor_anywhere():
    # QGuiApplication::setOverrideCursor can crash in CGImageCreate on macOS; busy_button replaces it.
    hits = [str(p.relative_to(NEGPY_ROOT)) for p in NEGPY_ROOT.rglob("*.py") if "setOverrideCursor(" in p.read_text(encoding="utf-8")]
    assert hits == []


def _recorder(button, fail):
    seen = {}

    def action(*_args, **_kwargs):
        seen["enabled"] = button.isEnabled()
        seen["text"] = button.text()
        if fail:
            raise RuntimeError("boom")

    return action, seen


def _geometry(qapp):
    controller = MagicMock()
    controller.state = AppState()
    sidebar = GeometrySidebar(controller)
    return sidebar, controller, sidebar.auto_skew_btn, "auto_skew_frame", sidebar._on_auto_skew


def _sidecars(qapp):
    controller = FakeController(repo=FakeRepo())
    sidebar = ExportSidebar(controller)
    return sidebar, controller, sidebar.export_sidecars_btn, "export_edit_sidecars", sidebar._on_export_sidecars


def _contact_sheet(qapp):
    controller = FakeController(repo=FakeRepo())
    sidebar = ExportSidebar(controller)
    return sidebar, controller, sidebar.contact_sheet_btn, "request_contact_sheet", sidebar._on_contact_sheet


def _flatfield(qapp, monkeypatch):
    monkeypatch.setattr(FlatFieldProfiles, "list_profiles", staticmethod(lambda: []))
    monkeypatch.setattr(flatfield_module.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: ("/flats/ref.arw", "")))
    monkeypatch.setattr(flatfield_module.QInputDialog, "getText", staticmethod(lambda *a, **k: ("rig", True)))
    controller = FakeController(repo=FakeRepo())
    sidebar = FlatFieldSidebar(controller)
    return sidebar, controller, sidebar.add_btn, "save_flatfield_profile", sidebar._on_add


@pytest.mark.parametrize("build", ["geometry", "sidecars", "contact_sheet", "flatfield"])
@pytest.mark.parametrize("fail", [False, True])
def test_action_shows_busy_on_its_button_and_restores_it(qapp, monkeypatch, build, fail):
    builders = {"geometry": _geometry, "sidecars": _sidecars, "contact_sheet": _contact_sheet}
    if build == "flatfield":
        sidebar, controller, button, method, handler = _flatfield(qapp, monkeypatch)
    else:
        sidebar, controller, button, method, handler = builders[build](qapp)
    button.setEnabled(True)
    label = button.text()
    action, seen = _recorder(button, fail)
    setattr(controller, method, MagicMock(side_effect=action))

    if fail:
        with pytest.raises(RuntimeError):
            handler()
    else:
        handler()

    assert seen["enabled"] is False
    assert button.isEnabled()
    assert button.text() == label
