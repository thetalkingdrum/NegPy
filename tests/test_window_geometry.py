import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from negpy.desktop.view.main_window import _DEFAULT_H, _DEFAULT_W, MainWindow, _clamp_geometry


def _inside(geo, avail):
    x, y, w, h = geo
    ax, ay, aw, ah = avail
    return ax <= x and ay <= y and x + w <= ax + aw and y + h <= ay + ah


class TestClampGeometry(unittest.TestCase):
    SMALL = (0, 0, 1366, 728)  # 1368x768 minus a taskbar

    def test_oversized_saved_shrinks_to_fit(self):
        geo = _clamp_geometry((10, 10, _DEFAULT_W, _DEFAULT_H), self.SMALL)
        self.assertEqual(geo, (0, 0, 1366, 728))
        self.assertTrue(_inside(geo, self.SMALL))

    def test_offscreen_position_pulled_inside(self):
        geo = _clamp_geometry((-50, -30, 800, 600), self.SMALL)
        self.assertTrue(_inside(geo, self.SMALL))
        # far-positive position is pulled back so the window stays fully visible
        geo2 = _clamp_geometry((5000, 5000, 800, 600), self.SMALL)
        self.assertTrue(_inside(geo2, self.SMALL))

    def test_default_centered_and_clamped(self):
        geo = _clamp_geometry(None, self.SMALL)
        self.assertEqual(geo, (0, 0, 1366, 728))  # default exceeds work area -> filled
        # on a big screen the default size is centered, not stretched
        big = (0, 0, 2560, 1440)
        x, y, w, h = _clamp_geometry(None, big)
        self.assertEqual((w, h), (_DEFAULT_W, _DEFAULT_H))
        self.assertEqual((x, y), ((2560 - _DEFAULT_W) // 2, (1440 - _DEFAULT_H) // 2))

    def test_screen_offset_respected(self):
        # second monitor whose work area starts at x=1920
        avail = (1920, 0, 1366, 728)
        geo = _clamp_geometry((0, 0, 1000, 700), avail)
        self.assertTrue(_inside(geo, avail))
        self.assertGreaterEqual(geo[0], 1920)


class _FakeRepo:
    def __init__(self, **initial):
        self._values = dict(initial)
        self.saved: dict = {}

    def get_global_setting(self, key, default=None):
        return self._values.get(key, default)

    def save_global_setting(self, key, value):
        self.saved[key] = value


class _Rect:
    def __init__(self, x, y, w, h):
        self._v = (x, y, w, h)

    def x(self):
        return self._v[0]

    def y(self):
        return self._v[1]

    def width(self):
        return self._v[2]

    def height(self):
        return self._v[3]


def _stub_window(state=None, fullscreen=False, maximized=False, geometry=(0, 0, 100, 100), normal_geometry=(1, 2, 300, 400)):
    repo = _FakeRepo(window_state=state) if state is not None else _FakeRepo()
    return SimpleNamespace(
        controller=SimpleNamespace(session=SimpleNamespace(repo=repo)),
        isFullScreen=lambda: fullscreen,
        isMaximized=lambda: maximized,
        geometry=lambda: _Rect(*geometry),
        normalGeometry=lambda: _Rect(*normal_geometry),
        showMaximized=MagicMock(),
        showFullScreen=MagicMock(),
        show=MagicMock(),
    ), repo


def test_show_restored_enters_maximized():
    stub, _ = _stub_window(state="maximized")
    MainWindow.show_restored(stub)
    stub.showMaximized.assert_called_once()
    stub.showFullScreen.assert_not_called()
    stub.show.assert_not_called()


def test_show_restored_enters_fullscreen():
    stub, _ = _stub_window(state="fullscreen")
    MainWindow.show_restored(stub)
    stub.showFullScreen.assert_called_once()
    stub.showMaximized.assert_not_called()


def test_show_restored_defaults_to_normal():
    stub, _ = _stub_window()
    MainWindow.show_restored(stub)
    stub.show.assert_called_once()
    stub.showMaximized.assert_not_called()
    stub.showFullScreen.assert_not_called()


def test_close_persists_maximized_state_and_normal_geometry():
    stub, repo = _stub_window(maximized=True, geometry=(0, 0, 1920, 1080), normal_geometry=(10, 20, 500, 400))
    MainWindow._persist_window_state(stub)
    assert repo.saved["window_state"] == "maximized"
    # the restored-size rect is saved, not the screen-filling maximized one
    assert repo.saved["window_geometry"] == [10, 20, 500, 400]


def test_close_persists_fullscreen_state():
    stub, repo = _stub_window(fullscreen=True, normal_geometry=(10, 20, 500, 400))
    MainWindow._persist_window_state(stub)
    assert repo.saved["window_state"] == "fullscreen"
    assert repo.saved["window_geometry"] == [10, 20, 500, 400]


def test_close_persists_normal_state_and_live_geometry():
    stub, repo = _stub_window(geometry=(10, 20, 500, 400))
    MainWindow._persist_window_state(stub)
    assert repo.saved["window_state"] == "normal"
    assert repo.saved["window_geometry"] == [10, 20, 500, 400]


if __name__ == "__main__":
    unittest.main()
