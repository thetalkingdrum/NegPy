"""The Analysis crossover plot: the print's neutrals, R and B against G per tone."""

from unittest.mock import MagicMock

import numpy as np

from negpy.desktop.session import AppState
from negpy.features.exposure.analysis import CROSSOVER_BINS, crossover_curve, crossover_point
from negpy.kernel.image.logic import working_oetf_encode


def _ramp(tilt: float, offset: float = 0.0) -> np.ndarray:
    """An encoded gray ramp whose R/B balance shifts by `tilt` from shadow to highlight."""
    t = np.linspace(0.02, 0.95, 600)[None, :]
    k = tilt * (t - 0.4) + offset
    lin = np.stack([t * (1 + k), t, t * (1 - k)], axis=-1).repeat(100, axis=0)
    return np.asarray(working_oetf_encode(np.clip(lin, 0.0, 1.0).astype(np.float32)))


def test_a_neutral_ramp_reads_flat_at_zero() -> None:
    curve = crossover_curve(_ramp(0.0))

    ok = ~np.isnan(curve["rg"])
    assert ok.sum() > CROSSOVER_BINS // 2
    assert np.abs(curve["rg"][ok]).max() < 0.003 and np.abs(curve["bg"][ok]).max() < 0.003


def test_crossover_tilts_the_lines_and_a_cast_only_moves_them() -> None:
    crossed = crossover_curve(_ramp(0.2))
    cast = crossover_curve(_ramp(0.0, offset=0.05))

    ok = np.flatnonzero(~np.isnan(crossed["rg"]))
    assert crossed["rg"][ok[-1]] - crossed["rg"][ok[0]] > 0.03  # warmer toward the highlights
    assert crossed["bg"][ok[-1]] - crossed["bg"][ok[0]] < -0.03
    ok = np.flatnonzero(~np.isnan(cast["rg"]))
    assert np.ptp(cast["rg"][ok]) < 0.005 and cast["rg"][ok].mean() > 0.015


def test_saturated_color_does_not_count_as_a_neutral() -> None:
    frame = _ramp(0.0)
    frame[:, :300] = np.asarray(working_oetf_encode(np.array([0.6, 0.05, 0.05], dtype=np.float32)))

    curve = crossover_curve(frame)

    assert abs(curve["neutral_fraction"] - 0.5) < 0.05
    ok = ~np.isnan(curve["rg"])
    assert np.abs(curve["rg"][ok]).max() < 0.003


def test_the_border_is_cut_away() -> None:
    frame = np.ones((120, 640, 3), dtype=np.float32)  # paper-white border
    frame[10:110, 20:620] = _ramp(0.0)

    curve = crossover_curve(frame, content_rect=(20, 10, 600, 100))

    assert curve["count"].sum() > 0
    assert crossover_curve(np.zeros((4, 4))) is None


def test_crossover_point_of_a_gray_is_neutral() -> None:
    tone, rg, bg = crossover_point((0.5, 0.5, 0.5))
    assert abs(tone - 0.5) < 1e-6 and rg == 0.0 and bg == 0.0


def test_the_plot_paints_both_lines_and_an_empty_state(qapp) -> None:
    from negpy.desktop.view.widgets.charts import CrossoverWidget

    widget = CrossoverWidget()
    widget.resize(300, 150)
    widget.grab()  # empty: "No neutrals to read"
    widget.set_data(crossover_curve(_ramp(0.2)))
    widget.set_marker((0.5, 0.01, -0.01))
    img = widget.grab().toImage()

    from PyQt6.QtGui import QColor

    from negpy.desktop.view.styles.theme import THEME

    def has(hex_: str) -> bool:
        want = QColor(hex_)
        return any(
            abs(QColor(img.pixel(x, y)).red() - want.red()) < 8 and abs(QColor(img.pixel(x, y)).blue() - want.blue()) < 8
            for x in range(0, 300, 2)
            for y in range(0, 150, 2)
        )

    assert has(THEME.channel_red_text) and has(THEME.channel_blue_text)


def test_the_crossover_measures_only_while_it_shows_and_clears_on_a_peek() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    frame = _ramp(0.1)
    panel = MagicMock()
    panel.controller.state = AppState()
    panel.controller.session.state.canvas_value.return_value = False

    panel.view_btn.currentData.return_value = "curve"
    RightPanel._update_crossover(panel, {"base_positive": frame})
    panel.crossover_widget.set_data.assert_not_called()

    panel.view_btn.currentData.return_value = "crossover"
    RightPanel._update_crossover(panel, {"base_positive": frame})
    assert panel.crossover_widget.set_data.call_args[0][0]["count"].sum() > 0

    panel.controller.state = AppState(negative_peek=True)
    RightPanel._update_crossover(panel, {"base_positive": frame})
    panel.crossover_widget.set_data.assert_called_with(None)


def test_a_drag_frame_keeps_the_last_crossover_plot() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    panel = MagicMock()
    panel.controller.state = AppState()
    panel.view_btn.currentData.return_value = "crossover"
    panel.controller.session.state.canvas_value.side_effect = lambda key: key == "interactive"

    RightPanel._update_crossover(panel, {"base_positive": _ramp(0.1)})

    panel.crossover_widget.set_data.assert_not_called()


def test_the_neutral_vectorscope_drops_the_scene_colors_and_keeps_the_cast() -> None:
    from negpy.features.exposure.analysis import SCOPE_HUE_BINS, neutral_vectorscope, vectorscope_histogram

    frame = _ramp(0.0, offset=0.06)  # a warm cast on the grays
    frame[:, :200] = np.asarray(working_oetf_encode(np.array([0.05, 0.5, 0.08], dtype=np.float32)))  # green foliage

    every = vectorscope_histogram(frame)
    neutral = neutral_vectorscope(frame)

    hue_of = lambda counts: int(counts.sum(axis=1).argmax())  # noqa: E731
    green_hue = hue_of(vectorscope_histogram(frame[:, :200]))
    assert every[green_hue].sum() > 0 and neutral[green_hue].sum() == 0
    # The neutrals' cluster sits off center toward the cast, toward red-yellow (+a, +b).
    cast_hue = hue_of(neutral)
    angle = (cast_hue + 0.5) / SCOPE_HUE_BINS * 2 * np.pi - np.pi
    assert -0.5 < angle < 2.0
    assert neutral[:, 0].sum() < neutral.sum()  # not at the very center
    assert neutral_vectorscope(np.zeros((4, 4))) is None


def _scope_panel(neutrals: bool, view: str = "vectorscope", interactive: bool = False) -> MagicMock:
    panel = MagicMock()
    panel.controller.state = AppState()
    panel.scope_neutrals_btn.isChecked.return_value = neutrals
    panel.view_btn.currentData.return_value = view
    panel.controller.session.state.canvas_value.side_effect = lambda key: interactive and key == "interactive"
    return panel


def test_neutrals_only_measures_the_print_and_waits_out_a_drag() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    metrics = {"vectorscope": "engine histogram", "base_positive": _ramp(0.0, offset=0.06)}

    panel = _scope_panel(neutrals=False)
    RightPanel._update_vectorscope(panel, metrics)
    panel.vectorscope_widget.set_data.assert_called_once_with("engine histogram")

    panel = _scope_panel(neutrals=True)
    RightPanel._update_vectorscope(panel, metrics)
    assert panel.vectorscope_widget.set_data.call_args[0][0].sum() > 0

    for idle in (_scope_panel(neutrals=True, interactive=True), _scope_panel(neutrals=True, view="curve")):
        RightPanel._update_vectorscope(idle, metrics)
        idle.vectorscope_widget.set_data.assert_not_called()

    peek = _scope_panel(neutrals=True)
    peek.controller.state = AppState(negative_peek=True)
    RightPanel._update_vectorscope(peek, metrics)
    peek.vectorscope_widget.set_data.assert_called_once_with(None)
