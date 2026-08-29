"""The filmstrip must not guess a frame's film process when the frame already knows it.

A slide reached the filmstrip inverted while the canvas rendered it as a positive: the
batch path re-detected the mode from an 8-bit preview, and a warm slide reads as an orange
mask. The stored mode is the answer the canvas uses, so the filmstrip uses it too.
"""

import unittest
from unittest.mock import MagicMock, patch

from negpy.desktop.controller import AppController
from negpy.desktop.session import DesktopSessionManager

SLIDE = {"name": "a.nef", "path": "/f/a.nef", "hash": "hash-a"}
FRESH = {"name": "b.nef", "path": "/f/b.nef", "hash": "hash-b"}


def _controller(stored: dict, autodetect_enabled: bool = True, new_file_default: str = "Color Negative"):
    c = MagicMock()
    c.state.uploaded_files = [dict(SLIDE), dict(FRESH)]
    c.state.thumbnails = {}
    c.state.autodetect_enabled = autodetect_enabled
    c._begin_batch.return_value = 1
    c.session.stored_process_mode = lambda asset: stored.get(asset["hash"], "")
    c.session.default_process_mode_for_new_file = lambda: new_file_default
    return c


def _emitted(controller) -> list:
    controller.thumbnail_requested.emit.assert_called_once()
    return controller.thumbnail_requested.emit.call_args.args[0]


class BatchRequest(unittest.TestCase):
    def test_stored_mode_rides_along(self):
        controller = _controller({"hash-a": "Transparency"})

        AppController.generate_missing_thumbnails(controller)

        modes = {f["hash"]: f["process_mode"] for f in _emitted(controller)}
        self.assertEqual(modes, {"hash-a": "Transparency", "hash-b": ""})

    def test_unstored_frame_skips_the_heuristic_when_autodetect_is_off(self):
        """A real open never runs the heuristic either when autodetect is off — it takes
        whatever process mode the next new file would get outright — so the filmstrip must
        not second-guess that choice with its own detection on an unstored frame."""
        controller = _controller({}, autodetect_enabled=False)

        AppController.generate_missing_thumbnails(controller)

        modes = {f["hash"]: f["process_mode"] for f in _emitted(controller)}
        self.assertEqual(modes, {"hash-a": "Color Negative", "hash-b": "Color Negative"})

    def test_stored_mode_still_wins_when_autodetect_is_off(self):
        controller = _controller({"hash-a": "Transparency"}, autodetect_enabled=False)

        AppController.generate_missing_thumbnails(controller)

        modes = {f["hash"]: f["process_mode"] for f in _emitted(controller)}
        self.assertEqual(modes, {"hash-a": "Transparency", "hash-b": "Color Negative"})

    def test_unstored_frame_follows_the_sticky_default_when_autodetect_is_off(self):
        """The fallback is whatever a new file would actually get — sticky B&W or Slide,
        not a hardcoded C41 — or every B&W/Slide roll would render as color negative."""
        controller = _controller({}, autodetect_enabled=False, new_file_default="B&W Negative")

        AppController.generate_missing_thumbnails(controller)

        modes = {f["hash"]: f["process_mode"] for f in _emitted(controller)}
        self.assertEqual(modes, {"hash-a": "B&W Negative", "hash-b": "B&W Negative"})

    def test_the_session_assets_are_not_touched(self):
        """The dicts cross to a worker thread, and a mode written back here would outlive
        the frame's own settings."""
        controller = _controller({"hash-a": "Transparency"})

        AppController.generate_missing_thumbnails(controller)

        self.assertFalse(any("process_mode" in f for f in controller.state.uploaded_files))
        self.assertIsNot(_emitted(controller)[0], controller.state.uploaded_files[0])


class StoredMode(unittest.TestCase):
    def test_a_composite_uses_the_mode_it_inherited(self):
        session = MagicMock()
        asset = {**SLIDE, "process_mode": "Transparency", "hdr_paths": ("/f/x.nef",)}

        with patch("negpy.desktop.session.load_or_promote") as load:
            self.assertEqual(DesktopSessionManager.stored_process_mode(session, asset), "Transparency")
            load.assert_not_called()

    def test_a_saved_edit_answers(self):
        session = MagicMock()
        saved = MagicMock()
        saved.process.process_mode = "Transparency"

        with patch("negpy.desktop.session.load_or_promote", return_value=saved):
            self.assertEqual(DesktopSessionManager.stored_process_mode(session, dict(SLIDE)), "Transparency")

    def test_an_undecided_frame_answers_nothing(self):
        """Not the sticky global: that is a guess about the next file, and the caller wants
        to know whether an answer exists at all."""
        session = MagicMock()

        with patch("negpy.desktop.session.load_or_promote", return_value=None):
            self.assertEqual(DesktopSessionManager.stored_process_mode(session, dict(FRESH)), "")


class DefaultForNewFile(unittest.TestCase):
    def test_answers_whatever_a_new_file_would_actually_get(self):
        """The same call `_hydrate_asset_config` makes for a fresh asset, so this can never
        drift from what opening a real new file actually does."""
        from dataclasses import replace

        from negpy.desktop.session import WorkspaceConfig

        session = MagicMock()
        sticky = replace(WorkspaceConfig(), process=replace(WorkspaceConfig().process, process_mode="B&W Negative"))
        session._apply_sticky_settings.return_value = sticky

        result = DesktopSessionManager.default_process_mode_for_new_file(session)

        self.assertEqual(result, "B&W Negative")
        session._apply_sticky_settings.assert_called_once()
        self.assertEqual(session._apply_sticky_settings.call_args.kwargs.get("only_global"), False)


if __name__ == "__main__":
    unittest.main()
