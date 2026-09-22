from dataclasses import replace
from unittest.mock import patch

import numpy as np
from PIL import Image

from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.storage.local_asset_store import LocalAssetStore
from negpy.services.assets.thumbnails import (
    asset_thumbnail_key,
    get_rendered_thumbnail,
    image_fingerprint,
    thumbnail_fingerprint,
)
from tests import test_thumbnail_refresh_controller as refresh_harness

BASE = WorkspaceConfig()


class TestFingerprint:
    def test_stable(self) -> None:
        assert thumbnail_fingerprint(BASE) == thumbnail_fingerprint(WorkspaceConfig())

    def test_render_field_changes_it(self) -> None:
        edited = replace(BASE, exposure=replace(BASE.exposure, density=BASE.exposure.density + 0.1))
        assert thumbnail_fingerprint(edited) != thumbnail_fingerprint(BASE)

    def test_non_render_state_does_not(self) -> None:
        brush = replace(BASE, retouch=replace(BASE.retouch, manual_dust_size=BASE.retouch.manual_dust_size + 5))
        metadata = replace(BASE, metadata=replace(BASE.metadata, **{_first_str_field(BASE.metadata): "Portra 400"}))
        export = replace(BASE, export=replace(BASE.export, jpeg_quality=BASE.export.jpeg_quality - 10))
        for config in (brush, metadata, export):
            assert thumbnail_fingerprint(config) == thumbnail_fingerprint(BASE)

    def test_diptych_differs_from_one_half(self) -> None:
        assert thumbnail_fingerprint(BASE, BASE) != thumbnail_fingerprint(BASE)


def _first_str_field(section) -> str:
    from dataclasses import fields

    return next(f.name for f in fields(section) if isinstance(getattr(section, f.name), str))


class TestStorage:
    def _store(self, tmp_path) -> LocalAssetStore:
        store = LocalAssetStore(str(tmp_path), str(tmp_path / "icc"))
        store.initialize()
        return store

    def test_rendered_thumbnail_round_trips_its_fingerprint(self, tmp_path) -> None:
        store = self._store(tmp_path)
        fp = thumbnail_fingerprint(BASE)
        buffer = np.full((40, 60, 3), 0.5, dtype=np.float32)
        thumb = get_rendered_thumbnail(buffer, "k", store, fingerprint=fp)
        assert image_fingerprint(thumb) == fp
        assert image_fingerprint(store.get_thumbnail("k")) == fp

    def test_placeholder_has_none(self, tmp_path) -> None:
        store = self._store(tmp_path)
        buffer = np.full((40, 60, 3), 0.5, dtype=np.float32)
        get_rendered_thumbnail(buffer, "k", store)
        assert image_fingerprint(store.get_thumbnail("k")) is None

    def test_foreign_comment_is_not_a_fingerprint(self) -> None:
        img = Image.new("RGB", (4, 4))
        img.info["comment"] = b"Scanned with VueScan"
        assert image_fingerprint(img) is None


class TestReconcile:
    setup_method = refresh_harness.TestThumbnailRefreshController.setup_method
    teardown_method = refresh_harness.TestThumbnailRefreshController.teardown_method

    def _key(self, file_hash: str) -> str:
        return asset_thumbnail_key(next(f for f in self.files if f["hash"] == file_hash))

    def _reconcile(self, edited, hashes=None) -> list[str]:
        for f in self.files:
            self.controller.state.thumbnails.setdefault(asset_thumbnail_key(f), object())
        self.session.repo.load_file_settings_many.return_value = {h: BASE for h in edited}
        with patch.object(self.controller, "diptych_pair", return_value=None):
            self.controller.reconcile_thumbnails(hashes)
        if not self.tasks:
            return []
        dispatched = [f.file_info["hash"] for f in self.tasks[0].frames]
        self.controller._on_thumbnail_render_cancelled()
        return dispatched

    def test_matching_edited_frame_is_left_alone(self) -> None:
        fp = thumbnail_fingerprint(BASE)
        self.controller.state.thumbnail_fingerprints.update({self._key("other"): fp, self._key("third"): fp})
        assert self._reconcile({"other", "third"}) == []
        assert self.controller.state.expected_thumbnail_fingerprints[self._key("other")] == fp

    def test_mismatched_and_placeholder_edited_frames_render(self) -> None:
        self.controller.state.thumbnail_fingerprints[self._key("other")] = "0" * 32
        assert self._reconcile({"other", "third"}) == ["other", "third"]

    def test_unedited_placeholder_is_acceptable(self) -> None:
        assert self._reconcile(set()) == []
        assert self.controller.state.expected_thumbnail_fingerprints[self._key("other")] is None

    def test_unedited_frame_with_an_outdated_render_renders(self) -> None:
        self.controller.state.thumbnail_fingerprints[self._key("other")] = "0" * 32
        assert self._reconcile(set()) == ["other"]

    def test_active_frame_is_skipped(self) -> None:
        assert "active" not in self._reconcile({"active", "other"})
        assert self._key("active") not in self.controller.state.expected_thumbnail_fingerprints

    def test_limited_to_the_given_hashes(self) -> None:
        assert self._reconcile({"other", "third"}, hashes=["third"]) == ["third"]

    def test_offscreen_edits_route_through_reconcile(self) -> None:
        self.session.frames_edited_offscreen.connect.assert_any_call(self.controller.reconcile_thumbnails)

    def test_background_render_carries_the_fingerprint(self) -> None:
        self._reconcile({"other"}, hashes=["other"])
        self.controller.refresh_thumbnails_for(["other"])
        frame = self.tasks[-1].frames[0]
        with patch.object(self.controller, "diptych_pair", return_value=None):
            self.controller._on_thumbnail_rendered(frame, np.zeros((4, 4, 3), dtype=np.float32))
        assert self.thumbnail_updates[-1].fingerprint == thumbnail_fingerprint(BASE)
        self.controller._on_thumbnail_render_cancelled()

    def test_rendered_arrival_records_the_fingerprint(self) -> None:
        img = Image.new("RGB", (4, 4))
        img.info["comment"] = thumbnail_fingerprint(BASE).encode()
        with patch.object(self.controller, "_set_thumbnail", return_value=True):
            self.controller._on_rendered_thumbnail({self._key("other"): img})
        assert self.controller.state.thumbnail_fingerprints[self._key("other")] == thumbnail_fingerprint(BASE)

    def test_render_neutral_write_carries_the_active_fingerprint(self) -> None:
        old = "0" * 32
        self.controller.state.last_metrics["thumbnail_fingerprint"] = old
        with patch.object(self.controller, "active_diptych", return_value=None):
            self.controller._carry_active_thumbnail_fingerprint(before=old)
        assert self.controller.state.last_metrics["thumbnail_fingerprint"] == thumbnail_fingerprint(BASE)

    def test_render_of_an_older_edit_is_not_carried(self) -> None:
        self.controller.state.last_metrics["thumbnail_fingerprint"] = "0" * 32
        with patch.object(self.controller, "active_diptych", return_value=None):
            self.controller._carry_active_thumbnail_fingerprint(before="1" * 32)
        assert self.controller.state.last_metrics["thumbnail_fingerprint"] == "0" * 32

    def test_no_fingerprint_to_carry_stays_empty(self) -> None:
        self.controller.state.last_metrics["thumbnail_fingerprint"] = ""
        self.controller._carry_active_thumbnail_fingerprint(before="")
        assert self.controller.state.last_metrics["thumbnail_fingerprint"] == ""

    def test_unloaded_thumbnail_waits_for_the_queue(self) -> None:
        with patch.object(self.controller, "_set_thumbnail"):
            self.session.repo.load_file_settings_many.return_value = {"other": BASE, "third": BASE}
            self.controller.state.thumbnails[self._key("other")] = object()
            with patch.object(self.controller, "diptych_pair", return_value=None):
                self.controller.reconcile_thumbnails()
        assert [f.file_info["hash"] for f in self.tasks[0].frames] == ["other"]
        self.controller._on_thumbnail_render_cancelled()
