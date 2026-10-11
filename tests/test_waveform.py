"""The Analysis waveform: print luma per image column, binned the same way on both engines,
and the chart view that shows it."""

import unittest
from unittest.mock import MagicMock

import numpy as np

from negpy.desktop.session import AppState
from negpy.features.exposure.analysis import (
    WAVEFORM_BINS,
    WAVEFORM_COLOR_SCALE,
    WAVEFORM_COLS,
    density_parade,
    scan_density_axis,
    scan_density_of_val,
    waveform,
    waveform_color,
    waveform_parade,
)
from negpy.infrastructure.gpu.device import GPUDevice


def test_a_left_to_right_ramp_climbs_column_by_column() -> None:
    ramp = np.linspace(0.0, 0.999, 512, dtype=np.float32)
    img = np.repeat(np.repeat(ramp[None, :, None], 8, axis=0), 3, axis=2)

    counts = waveform(img)

    assert counts.shape == (WAVEFORM_BINS, WAVEFORM_COLS)
    assert counts.sum() == 8 * 512
    peak_bin = counts.argmax(axis=0)
    assert peak_bin[0] == 0 and peak_bin[-1] == WAVEFORM_BINS - 1
    assert np.all(np.diff(peak_bin) >= 0)


def test_every_column_gets_its_share_of_a_narrow_frame() -> None:
    img = np.full((10, 64, 3), 0.5, dtype=np.float32)

    counts = waveform(img)

    # 64 pixels spread over 256 columns: every fourth column holds one pixel column.
    assert counts.sum(axis=0)[::4].tolist() == [10.0] * 64
    assert waveform(np.zeros((4, 4))) is None


def test_each_cell_sums_the_color_of_its_pixels() -> None:
    img = np.zeros((10, 512, 3), dtype=np.float32)
    img[:, :256] = (0.8, 0.3, 0.2)
    img[:, 256:] = (0.2, 0.3, 0.8)

    counts = waveform(img)
    sums = waveform_color(img)

    assert sums.shape == (3, WAVEFORM_BINS, WAVEFORM_COLS)
    mean = sums.sum(axis=1) / counts.sum(axis=0) / WAVEFORM_COLOR_SCALE
    np.testing.assert_allclose(mean[:, 0], (0.8, 0.3, 0.2), atol=1e-3)
    np.testing.assert_allclose(mean[:, -1], (0.2, 0.3, 0.8), atol=1e-3)
    assert waveform_color(np.zeros((4, 4))) is None


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
class TestWaveformParity(unittest.TestCase):
    def test_wgsl_bins_match_the_cpu(self):
        from negpy.domain.models import WorkspaceConfig
        from negpy.services.rendering.image_processor import ImageProcessor

        rng = np.random.default_rng(3)
        grad = np.linspace(0.05, 0.9, 300, dtype=np.float32)
        img = np.repeat(grad[None, :, None], 200, axis=0).repeat(3, axis=2)
        img = np.ascontiguousarray(img + rng.uniform(0, 0.02, img.shape).astype(np.float32))
        proc = ImageProcessor()
        cfg = WorkspaceConfig()

        cpu_out, _ = proc.run_pipeline(img, cfg, "wave|cpu", render_size_ref=300, prefer_gpu=False, readback_metrics=True)
        _gpu_out, gpu_metrics = proc.run_pipeline(img, cfg, "wave|gpu", render_size_ref=300, prefer_gpu=True, readback_metrics=True)

        gpu_wave = gpu_metrics.get("waveform")
        if gpu_wave is None:
            self.skipTest("render fell back to the CPU engine")
        # A CPU render carries no in-shader waveform; the worker bins the float output.
        cpu_wave = waveform(cpu_out)
        self.assertEqual(cpu_wave.sum(), gpu_wave.sum())
        a = cpu_wave / cpu_wave.sum()
        b = gpu_wave / gpu_wave.sum()
        self.assertLess(float(np.abs(a - b).sum()), 0.02)

    def test_wgsl_color_sums_match_the_cpu(self):
        from negpy.domain.models import WorkspaceConfig
        from negpy.services.rendering.image_processor import ImageProcessor

        grad = np.linspace(0.05, 0.9, 300, dtype=np.float32)
        img = np.stack([grad, grad[::-1], np.full_like(grad, 0.4)], axis=-1)
        img = np.ascontiguousarray(np.repeat(img[None], 200, axis=0))
        proc = ImageProcessor()
        cfg = WorkspaceConfig()

        cpu_out, _ = proc.run_pipeline(img, cfg, "wavec|cpu", render_size_ref=300, prefer_gpu=False, readback_metrics=True)
        _gpu_out, gpu_metrics = proc.run_pipeline(img, cfg, "wavec|gpu", render_size_ref=300, prefer_gpu=True, readback_metrics=True)

        gpu_sums = gpu_metrics.get("waveform_color")
        if gpu_sums is None:
            self.skipTest("render fell back to the CPU engine")
        cpu_sums = waveform_color(cpu_out)
        # Per-column channel means: robust to a pixel landing one tone bin apart.
        cpu_mean = cpu_sums.sum(axis=1) / cpu_sums.sum()
        gpu_mean = gpu_sums.sum(axis=1) / gpu_sums.sum()
        self.assertLess(float(np.abs(cpu_mean - gpu_mean).sum()), 0.02)


def test_waveform_widget_paints_and_marks_clipped_columns(qapp) -> None:
    from PyQt6.QtGui import QColor

    from negpy.desktop.view.styles.theme import THEME
    from negpy.desktop.view.widgets.charts import WaveformWidget

    widget = WaveformWidget()
    widget.resize(256, 64)
    counts = np.zeros((WAVEFORM_BINS, WAVEFORM_COLS))
    counts[0, :64] = 5.0  # left quarter at paper black
    counts[64, 64:] = 5.0
    widget.set_data(counts)

    img = widget.grab().toImage()

    assert QColor(img.pixel(20, 61)).name() == QColor(THEME.clip_warning).name()
    assert QColor(img.pixel(200, 61)).name() != QColor(THEME.clip_warning).name()

    widget.set_log_scale(True)
    assert widget._images is None
    widget.set_data(None)
    widget.grab()


def test_the_view_choice_switches_the_chart_and_persists() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    panel = MagicMock()
    panel.view_btn.currentData.return_value = "waveform"

    RightPanel._on_analysis_view(panel, 1)

    panel.chart_stack.setCurrentIndex.assert_called_once_with(1)
    panel.controller.session.repo.save_global_setting.assert_called_once_with("analysis_view", "waveform")


def test_a_peek_clears_the_waveform() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    panel = MagicMock()
    panel.controller.state = AppState(negative_peek=True)

    RightPanel._update_histograms(panel, {"waveform": np.ones((WAVEFORM_BINS, WAVEFORM_COLS))})

    panel.waveform_widget.set_data.assert_called_once_with(None, None)


def test_waveform_widget_draws_each_cell_in_its_color(qapp) -> None:
    from PyQt6.QtGui import QColor

    from negpy.desktop.view.widgets.charts import WaveformWidget

    widget = WaveformWidget()
    widget.resize(WAVEFORM_COLS, WAVEFORM_BINS)
    counts = np.zeros((WAVEFORM_BINS, WAVEFORM_COLS))
    counts[64, :] = 4.0
    color = np.zeros((3, WAVEFORM_BINS, WAVEFORM_COLS))
    color[0, 64, :128] = 4.0 * 0.9 * WAVEFORM_COLOR_SCALE  # left half red
    color[:, 64, 128:] = 4.0 * 0.5 * WAVEFORM_COLOR_SCALE  # right half neutral
    widget.set_data(counts, color)

    img = widget.grab().toImage()
    y = WAVEFORM_BINS - 1 - 64

    red = QColor(img.pixel(60, y))
    gray = QColor(img.pixel(200, y))
    assert red.red() > 200 and red.green() < 60 and red.blue() < 60
    assert gray.red() > 200 and gray.red() == gray.green() == gray.blue()


def test_the_parade_bins_each_channel_on_its_own() -> None:
    img = np.zeros((10, 512, 3), dtype=np.float32)
    img[:, :256] = (0.8, 0.3, 0.2)
    img[:, 256:] = (0.2, 0.3, 0.8)

    parade = waveform_parade(img)

    assert parade.shape == (3, WAVEFORM_BINS, WAVEFORM_COLS)
    assert parade[0].sum() == parade[1].sum() == parade[2].sum() == 10 * 512
    peak = parade.argmax(axis=1)
    assert peak[:, 0].tolist() == [int(0.8 * WAVEFORM_BINS), int(0.3 * WAVEFORM_BINS), int(0.2 * WAVEFORM_BINS)]
    assert peak[:, -1].tolist() == [int(0.2 * WAVEFORM_BINS), int(0.3 * WAVEFORM_BINS), int(0.8 * WAVEFORM_BINS)]
    assert waveform_parade(np.zeros((4, 4))) is None


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
def test_wgsl_parade_matches_the_cpu() -> None:
    from negpy.domain.models import WorkspaceConfig
    from negpy.services.rendering.image_processor import ImageProcessor

    grad = np.linspace(0.05, 0.9, 300, dtype=np.float32)
    img = np.stack([grad, grad[::-1], np.full_like(grad, 0.4)], axis=-1)
    img = np.ascontiguousarray(np.repeat(img[None], 200, axis=0))
    proc = ImageProcessor()
    cfg = WorkspaceConfig()

    cpu_out, _ = proc.run_pipeline(img, cfg, "parade|cpu", render_size_ref=300, prefer_gpu=False, readback_metrics=True)
    _gpu_out, gpu_metrics = proc.run_pipeline(img, cfg, "parade|gpu", render_size_ref=300, prefer_gpu=True, readback_metrics=True)

    gpu = gpu_metrics.get("waveform_parade")
    if gpu is None:
        return
    cpu = waveform_parade(cpu_out)
    assert cpu.sum() == gpu.sum()
    for c in range(3):
        assert float(np.abs(cpu[c] / cpu[c].sum() - gpu[c] / gpu[c].sum()).sum()) < 0.02


def test_the_parade_draws_three_panels_in_channel_colors(qapp) -> None:
    from PyQt6.QtGui import QColor

    from negpy.desktop.view.styles.theme import THEME
    from negpy.desktop.view.widgets.charts import WaveformWidget

    widget = WaveformWidget(parade=True)
    widget.resize(3 * 100, WAVEFORM_BINS)
    counts = np.zeros((3, WAVEFORM_BINS, WAVEFORM_COLS))
    counts[:, 64, :] = 4.0
    counts[0, -1, :32] = 1.0  # red reaches paper white on the left
    widget.set_data(counts)

    img = widget.grab().toImage()
    y = WAVEFORM_BINS - 1 - 64

    for panel, hex_ in enumerate((THEME.channel_red_text, THEME.channel_green_text, THEME.channel_blue_text)):
        got, want = QColor(img.pixel(panel * 101 + 50, y)), QColor(hex_)
        assert max(abs(got.red() - want.red()), abs(got.green() - want.green()), abs(got.blue() - want.blue())) <= 1
    assert QColor(img.pixel(5, 2)).name() == QColor(THEME.clip_warning).name()
    assert QColor(img.pixel(106, 2)).name() != QColor(THEME.clip_warning).name()
    widget.set_data(np.ones((WAVEFORM_BINS, WAVEFORM_COLS)))  # a luma waveform is not a parade
    assert widget._counts is None


def test_a_flat_peek_clears_the_parade() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    panel = MagicMock()
    panel.controller.state = AppState(flat_peek=True)
    parade = np.ones((3, WAVEFORM_BINS, WAVEFORM_COLS))

    RightPanel._update_histograms(panel, {"waveform_parade": parade, "density_parade": parade})

    panel.parade_widget.set_scan_domain.assert_called_once_with(False)
    panel.parade_widget.set_data.assert_called_once_with(None)


def test_peek_negative_shows_the_scan_on_the_parade() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    panel = MagicMock()
    panel.controller.state = AppState(negative_peek=True)
    scan = np.ones((3, WAVEFORM_BINS, WAVEFORM_COLS))

    RightPanel._update_histograms(panel, {"waveform_parade": np.zeros_like(scan), "density_parade": scan})

    panel.parade_widget.set_scan_domain.assert_called_once_with(True)
    panel.parade_widget.set_data.assert_called_once_with(scan)
    panel.waveform_widget.set_data.assert_called_once_with(None, None)


def test_the_print_parade_returns_after_the_peek() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    panel = MagicMock()
    panel.controller.state = AppState()
    printed = np.ones((3, WAVEFORM_BINS, WAVEFORM_COLS))

    RightPanel._update_histograms(panel, {"waveform_parade": printed, "density_parade": np.zeros_like(printed)})

    panel.parade_widget.set_scan_domain.assert_called_once_with(False)
    panel.parade_widget.set_data.assert_called_once_with(printed)


def test_the_scan_parade_bins_each_channel_scan_density_dense_at_the_bottom() -> None:
    log = np.zeros((10, 512, 3), dtype=np.float32)
    log[:, :256] = (-0.3, -1.0, -2.5)  # base-like red, dense blue
    log[:, 256:] = (-2.5, -1.0, -0.3)

    parade = density_parade(log)

    assert parade.shape == (3, WAVEFORM_BINS, WAVEFORM_COLS)
    assert parade[0].sum() == parade[1].sum() == parade[2].sum() == 10 * 512
    peak = parade.argmax(axis=1)
    want = [int(np.floor(scan_density_axis(d) * WAVEFORM_BINS)) for d in (0.3, 1.0, 2.5)]
    assert peak[:, 0].tolist() == want
    assert peak[:, -1].tolist() == want[::-1]
    # Denser is lower: D 2.5 sits below D 0.3.
    assert peak[0, -1] < peak[0, 0]
    assert density_parade(None) is None and density_parade(np.zeros((4, 4))) is None


def test_the_scan_parade_reads_the_crop_and_piles_overflow_at_the_axis_ends() -> None:
    log = np.full((20, 100, 3), -1.0, dtype=np.float32)
    log[:, :50, 0] = 0.2  # past sensor full scale: top edge bin
    log[:, 50:, 0] = -5.0  # denser than the axis reaches: bottom edge bin

    parade = density_parade(log, roi=(5, 15, 0, 100))

    assert parade[0].sum() == 10 * 100
    assert parade[0, -1, : WAVEFORM_COLS // 2].sum() == 10 * 50
    assert parade[0, 0, WAVEFORM_COLS // 2 :].sum() == 10 * 50
    assert parade[1, 0].sum() == parade[1, -1].sum() == 0


def test_a_val_reads_back_to_scan_density_through_its_bounds() -> None:
    from negpy.features.exposure.normalization import LogNegativeBounds

    bounds = LogNegativeBounds(floors=(-2.0, -2.2, -2.6), ceils=(-0.4, -0.7, -1.1))

    np.testing.assert_allclose(scan_density_of_val((0.0, 0.0, 0.0), bounds), (2.0, 2.2, 2.6))
    np.testing.assert_allclose(scan_density_of_val((1.0, 1.0, 1.0), bounds), (0.4, 0.7, 1.1))


@unittest.skipUnless(GPUDevice.get().is_available, "GPU not available")
def test_wgsl_scan_parade_matches_the_cpu() -> None:
    from negpy.domain.models import WorkspaceConfig
    from negpy.services.rendering.image_processor import ImageProcessor

    grad = np.linspace(0.05, 0.9, 300, dtype=np.float32)
    img = np.stack([grad, grad[::-1], np.full_like(grad, 0.4)], axis=-1)
    img = np.ascontiguousarray(np.repeat(img[None], 200, axis=0))
    proc = ImageProcessor()
    cfg = WorkspaceConfig()

    _cpu_out, cpu_metrics = proc.run_pipeline(img, cfg, "dparade|cpu", render_size_ref=300, prefer_gpu=False, readback_metrics=True)
    _gpu_out, gpu_metrics = proc.run_pipeline(img, cfg, "dparade|gpu", render_size_ref=300, prefer_gpu=True, readback_metrics=True)

    gpu = gpu_metrics.get("density_parade")
    if gpu is None:
        return
    cpu = cpu_metrics["density_parade"]
    assert cpu.sum() == gpu.sum()
    for c in range(3):
        assert float(np.abs(cpu[c] / cpu[c].sum() - gpu[c] / gpu[c].sum()).sum()) < 0.02


def _row_lit(img, y: int) -> bool:
    from PyQt6.QtGui import QColor

    from negpy.desktop.view.styles.theme import THEME

    bg = QColor(THEME.canvas_bg_black).name()
    return any(QColor(img.pixel(x, y)).name() != bg for x in range(60, img.width() - 4))


def _row_across(img, y: int) -> bool:
    """A drawn line across a third of the row, over the black trace image and its separators."""
    from PyQt6.QtGui import QColor

    return sum(QColor(img.pixel(x, y)).lightness() > 40 for x in range(img.width())) > img.width() // 3


def test_the_scan_parade_trades_zone_guides_for_density_guides_and_marks(qapp) -> None:
    from negpy.desktop.view.widgets.charts import WaveformWidget
    from negpy.features.exposure.analysis import encoded_of_zone

    widget = WaveformWidget(parade=True)
    widget.resize(300, 200)
    counts = np.zeros((3, WAVEFORM_BINS, WAVEFORM_COLS))
    counts[:, -1, 0] = 1.0  # one cell at the top edge, clear of every guide row
    widget.set_data(counts)
    zone_v = int(200 - encoded_of_zone(5) * 200)
    d2 = int(200 - float(scan_density_axis(2.0)) * 200)

    printed = widget.grab().toImage()
    assert _row_across(printed, zone_v) and not _row_across(printed, d2)

    widget.set_marker((0.5, 0.5, 0.5, 0.5))
    widget.set_scan_domain(True, np.array([[0.2, 0.7]] * 3))
    assert widget._marker is None
    scan = widget.grab().toImage()
    assert _row_across(scan, d2) and not _row_across(scan, zone_v)
    assert _row_across(scan, int(200 - 0.2 * 200)) and _row_across(scan, int(200 - 0.7 * 200))


def test_peek_negative_marks_each_channels_stretch_ends() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel
    from negpy.features.exposure.normalization import LogNegativeBounds

    panel = MagicMock()
    panel.controller.state = AppState(negative_peek=True)
    panel._scan_marks_and_caption = lambda: RightPanel._scan_marks_and_caption(panel)
    bounds = LogNegativeBounds(floors=(-2.0, -2.2, -2.6), ceils=(-0.4, -0.7, -1.1))

    RightPanel._update_histograms(panel, {"density_parade": np.ones((3, WAVEFORM_BINS, WAVEFORM_COLS)), "final_bounds": bounds})

    enabled, marks, caption = panel.parade_widget.set_scan_domain.call_args[0]
    assert enabled and caption == "Scan density"
    np.testing.assert_allclose(marks, scan_density_axis(np.array([[2.0, 0.4], [2.2, 0.7], [2.6, 1.1]])))


def test_peek_negative_rings_the_scan_density_of_the_hovered_pixel() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel
    from negpy.features.exposure.densitometer import DensitometerReading
    from negpy.features.exposure.normalization import LogNegativeBounds

    panel = MagicMock()
    panel.controller.state = AppState(negative_peek=True)
    panel._scan_bounds = LogNegativeBounds(floors=(-2.0, -2.2, -2.6), ceils=(-0.4, -0.7, -1.1))

    RightPanel._on_pixel_readout(panel, (0.25, 0.5, (0.9, 0.9, 0.9)))
    panel.parade_widget.set_marker.assert_not_called()

    reading = DensitometerReading((0.1, 0.2, 0.3), 0.5, 1.0, 5.0, (1.0, 0.5, 0.0))
    RightPanel._on_densitometer(panel, reading)
    x, *tones = panel.parade_widget.set_marker.call_args[0][0]
    assert x == 0.25
    np.testing.assert_allclose(tones, scan_density_axis((0.4, 1.45, 2.6)))

    RightPanel._on_densitometer(panel, None)
    panel.parade_widget.set_marker.assert_called_with(None)


def test_the_parade_hover_draws_greens_level_across_all_panels(qapp) -> None:
    from negpy.desktop.view.widgets.charts import WaveformWidget

    widget = WaveformWidget(parade=True)
    widget.resize(300, 200)
    counts = np.zeros((3, WAVEFORM_BINS, WAVEFORM_COLS))
    counts[:, -1, 0] = 1.0
    widget.set_data(counts)
    widget.set_marker((0.5, 0.8, 0.35, 0.1))

    img = widget.grab().toImage()

    assert _row_across(img, int(200 - 0.35 * 200))


def _scale_panel(view: str, negative_peek: bool, saved: dict) -> MagicMock:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    panel = MagicMock()
    panel.view_btn.currentData.return_value = view
    panel.controller.state = AppState(negative_peek=negative_peek)
    panel.controller.session.repo.get_global_setting.side_effect = lambda key, default=None: saved.get(key, default)
    panel._scope_log = lambda: False
    panel._scan_parade_log = lambda: RightPanel._scan_parade_log(panel)
    return panel


def test_the_scan_parade_keeps_its_own_scale_log_at_first() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    scan = _scale_panel("parade", True, {"histogram_log_scale": False})
    RightPanel._sync_scale_btn(scan)
    scan.parade_widget.set_log_scale.assert_called_once_with(True)
    scan.scale_btn.setCurrentIndex.assert_called_once_with(1)

    printed = _scale_panel("parade", False, {"histogram_log_scale": False, "scan_parade_log": True})
    RightPanel._sync_scale_btn(printed)
    printed.parade_widget.set_log_scale.assert_called_once_with(False)
    printed.scale_btn.setCurrentIndex.assert_called_once_with(0)


def test_lin_log_in_peek_negative_saves_only_the_scan_parades_scale() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    panel = _scale_panel("parade", True, {})

    RightPanel._on_histogram_scale(panel, 0)

    panel.controller.session.repo.save_global_setting.assert_called_once_with("scan_parade_log", False)
    panel.parade_widget.set_log_scale.assert_called_once_with(False)
    panel.curve_widget.set_log_scale.assert_not_called()


def test_the_parade_overlay_adds_the_channels_on_one_plot(qapp) -> None:
    from PyQt6.QtGui import QColor

    from negpy.desktop.view.widgets.charts import WaveformWidget

    widget = WaveformWidget(parade=True)
    widget.resize(WAVEFORM_COLS, WAVEFORM_BINS)
    counts = np.zeros((3, WAVEFORM_BINS, WAVEFORM_COLS))
    counts[:, 64, :128] = 4.0  # left half: R, G and B agree
    counts[0, 80, 128:] = 4.0  # right half: red above green and blue
    counts[1:, 64, 128:] = 4.0
    widget.set_overlay(True)
    widget.set_data(counts)

    img = widget.grab().toImage()
    agree = QColor(img.pixel(60, WAVEFORM_BINS - 1 - 64))
    red = QColor(img.pixel(200, WAVEFORM_BINS - 1 - 80))
    cyan = QColor(img.pixel(200, WAVEFORM_BINS - 1 - 64))

    assert min(agree.red(), agree.green(), agree.blue()) > 200
    assert red.red() > 200 and red.green() < 40 and red.blue() < 40
    assert cyan.red() < 40 and cyan.green() > 200 and cyan.blue() > 200


def test_the_overlay_choice_persists_and_shows_only_with_the_parade() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    panel = MagicMock()
    RightPanel._on_parade_overlay(panel, True)
    panel.parade_widget.set_overlay.assert_called_once_with(True)
    panel.controller.session.repo.save_global_setting.assert_called_once_with("parade_overlay", True)

    panel.view_btn.currentData.return_value = "waveform"
    RightPanel._on_analysis_view(panel, 1)
    panel.parade_overlay_btn.setVisible.assert_called_once_with(False)
