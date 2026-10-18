"""Measure Base: a clear base clicked from the scan Parade in Peek Negative, as a check that is
never saved."""

from unittest.mock import MagicMock

import numpy as np

from negpy.desktop.session import AppState, ToolMode
from negpy.features.exposure.normalization import LogNegativeBounds, log_from_val

_BOUNDS = LogNegativeBounds(floors=(-1.5, -1.4, -1.3), ceils=(-0.3, -0.2, -0.1))


def _controller(patch):
    ctl = MagicMock()
    ctl.state = AppState(negative_peek=True, active_tool=ToolMode.BASE_CHECK)
    ctl.state.last_metrics = {"final_bounds": _BOUNDS}
    ctl._sample_normalized_patch.return_value = patch
    return ctl


def test_a_click_reads_the_base_as_scan_density_and_saves_nothing():
    from negpy.desktop.controller import AppController

    ctl = _controller(np.full((81, 3), 0.9, dtype=np.float32))

    AppController._handle_base_check(ctl, 0.5, 0.5)

    want = -np.asarray(log_from_val((0.9, 0.9, 0.9), _BOUNDS))
    np.testing.assert_allclose(ctl.state.measured_film_base, want, atol=1e-6)
    ctl.measured_film_base_changed.emit.assert_called_once()
    ctl.session.update_config.assert_not_called()
    ctl.set_active_tool.assert_not_called()


def test_a_click_off_flat_film_is_refused():
    from negpy.desktop.controller import AppController

    patch = np.full((81, 3), 0.9, dtype=np.float32)
    patch[::2] = 0.1  # edge print: half the patch far denser
    ctl = _controller(patch)

    AppController._handle_base_check(ctl, 0.5, 0.5)

    assert ctl.state.measured_film_base is None
    assert ctl.set_status.call_args[0][2] == "warning"


def test_closing_the_peek_drops_the_reading_and_the_tool():
    from negpy.desktop.controller import AppController

    ctl = _controller(None)
    ctl.state.measured_film_base = (0.5, 0.5, 0.5)

    AppController._on_negative_peek_changed(ctl, True)
    assert ctl.state.measured_film_base == (0.5, 0.5, 0.5)

    AppController._on_negative_peek_changed(ctl, False)
    assert ctl.state.measured_film_base is None
    ctl.set_active_tool.assert_called_once_with(ToolMode.NONE)
    ctl.measured_film_base_changed.emit.assert_called_once()


def test_the_parade_reads_a_measured_base_before_the_renders():
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    panel = MagicMock()
    panel.controller.state = AppState(negative_peek=True)
    panel._scan_base_density, panel._scan_base_source = np.array([0.2, 0.6, 0.9]), "picked"
    assert RightPanel._scan_base(panel)[1] == "picked"

    panel.controller.state.measured_film_base = (0.3, 0.5, 0.7)
    density, source = RightPanel._scan_base(panel)
    np.testing.assert_allclose(density, (0.3, 0.5, 0.7))
    assert source == "measured"

    panel._scan_base = lambda: RightPanel._scan_base(panel)
    panel._scan_bounds = None
    panel.scan_base_btn.isChecked.return_value = True
    assert RightPanel._scan_marks_and_caption(panel)[1] == "Scan density above measured base"


def test_the_button_shows_with_the_parade_in_peek_negative_only():
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    for view, peek, shown in (("parade", True, True), ("parade", False, False), ("waveform", True, False)):
        panel = MagicMock()
        panel.controller.state = AppState(negative_peek=peek)
        panel.view_btn.currentData.return_value = view
        RightPanel._sync_parade_tools(panel)
        panel.base_check_btn.setVisible.assert_called_once_with(shown)
