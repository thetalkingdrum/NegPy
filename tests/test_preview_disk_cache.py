"""The disk tier of the preview cache: lossless entries, a strict key and one global cap."""

from __future__ import annotations

import dataclasses
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from negpy.features.lens.models import LensMetadata
from negpy.features.lens.warps import PanasonicWarp, RectilinearWarp
from negpy.features.rgbscan.models import RgbScanConfig
from negpy.infrastructure.loaders.tiff_loader import NonStandardFileWrapper
from negpy.kernel.system.config import APP_CONFIG
from negpy.services.rendering import preview_disk_cache as pdc
from negpy.services.rendering.preview_cache import PreviewBufferCache, PreviewCacheKey
from negpy.services.rendering.preview_manager import PreviewManager

MB = 1024 * 1024


@pytest.fixture
def disk_config(tmp_path, monkeypatch):
    """The process configuration with the disk tier on, in a folder of the test's own."""
    monkeypatch.setattr(APP_CONFIG, "preview_disk_cache", True)
    monkeypatch.setattr(APP_CONFIG, "preview_disk_cache_dir", str(tmp_path / "previews"))
    monkeypatch.setattr(APP_CONFIG, "preview_disk_cache_max_bytes", 4 * 1024**3)
    monkeypatch.setattr(pdc, "_SHARED", {})
    return APP_CONFIG


def _store(config, **kwargs) -> pdc.PreviewDiskCache:
    return pdc.PreviewDiskCache(pdc.cache_root(config), config, min_free_bytes=kwargs.pop("min_free_bytes", 0))


def _noise(shape, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).random(shape, dtype=np.float32)


def _key(name: str = "hash-a", **kwargs) -> tuple:
    return PreviewCacheKey(
        file_hash=name, use_camera_wb=False, workspace_color_space="Adobe RGB", full_resolution=False, **kwargs
    ).as_tuple()


def test_an_entry_reads_back_bit_identical_with_its_metadata(disk_config) -> None:
    store = _store(disk_config)
    buffer = _noise((40, 60, 3))
    lens = LensMetadata(
        source="DNG opcodes",
        warps=(RectilinearWarp(coefficients=((1.0, 0.01, 0.0, 0.0),), center=(0.5, 0.5)), PanasonicWarp(1.0, 0.1, 0.2, 0.3)),
        active_area=(0, 0, 40, 60),
        buffer_area=(1, 2, 39, 58),
    )
    metadata = {
        "orientation": 6,
        "raw_flip": 0,
        "color_space": None,
        "cam_xyz": [[0.5, 0.1, 0.0], [0.2, 0.9, 0.1], [0.0, 0.1, 0.7]],
        "camera_wb": (np.float32(2.1), 1.0, np.float32(1.4)),
        "icc_profile": b"\x00\x01icc",
        "ir": _noise((400, 600), 1),
        "ir_preview": _noise((40, 60), 2),
        "detect_preview": _noise((40, 60, 3), 3),
        "ir_valid_mask": np.ones((40, 60), dtype=bool),
        "lens_correction": lens,
        "nested": {"a": (1, 2), "b": [np.int64(3)]},
    }

    assert store.put(_key(), buffer, (400, 600), metadata)
    out, dims, meta = store.get(_key())

    assert dims == (400, 600)
    assert out.dtype == np.float32 and np.array_equal(out, buffer)
    assert out.flags.writeable
    assert np.array_equal(meta["ir_preview"], metadata["ir_preview"])
    assert np.array_equal(meta["detect_preview"], metadata["detect_preview"])
    assert meta["ir_valid_mask"].dtype == bool
    assert meta["camera_wb"] == metadata["camera_wb"] and isinstance(meta["camera_wb"][0], np.float32)
    assert meta["lens_correction"] == lens
    assert meta["icc_profile"] == metadata["icc_profile"]
    assert meta["nested"] == {"a": (1, 2), "b": [3]}
    assert {k: meta[k] for k in ("orientation", "raw_flip", "color_space", "cam_xyz")} == {
        k: metadata[k] for k in ("orientation", "raw_flip", "color_space", "cam_xyz")
    }
    assert "ir" not in meta


def test_metadata_with_an_unknown_type_stays_out_of_the_disk_tier(disk_config) -> None:
    store = _store(disk_config)

    @dataclasses.dataclass(frozen=True)
    class Foreign:
        value: int = 1

    assert not store.put(_key(), _noise((8, 8, 3)), (8, 8), {"thing": Foreign()})
    assert not store.put(_key("b"), _noise((8, 8, 3)), (8, 8), {"thing": object()})
    assert store.get(_key()) is None
    assert store.usage() == (0, 0)


@pytest.mark.parametrize(
    "change",
    [
        lambda mp: mp.setattr(APP_CONFIG, "preview_render_size", APP_CONFIG.preview_render_size + 128),
        lambda mp: mp.setattr(pdc, "PREVIEW_DISK_CACHE_VERSION", pdc.PREVIEW_DISK_CACHE_VERSION + 1),
        lambda mp: mp.setattr("negpy.__version__", "999.0.0"),
    ],
    ids=["preview size", "cache version", "app version"],
)
def test_a_changed_decoder_identity_is_a_miss(disk_config, monkeypatch, change) -> None:
    store = _store(disk_config)
    store.put(_key(), _noise((8, 8, 3)), (8, 8), {})
    assert store.get(_key()) is not None

    change(monkeypatch)

    assert store.get(_key()) is None


def test_every_key_field_separates_entries(disk_config) -> None:
    store = _store(disk_config)
    store.put(_key(), _noise((8, 8, 3)), (8, 8), {})

    for other in (_key("hash-b"), _key(demosaic="AMaZE"), _key(half=1), _key(highlight_mode=2), _key(lens_token="|lens")):
        assert store.get(other) is None


def test_the_cap_evicts_the_least_recently_used_entries(disk_config, monkeypatch) -> None:
    store = _store(disk_config)
    sizes = []
    for i, name in enumerate("abc"):
        store.put(_key(name), _noise((256, 256, 3), i), (256, 256), {})
        sizes.append(store.usage()[1] - sum(sizes))
    store.get(_key("a"))  # a is now more recent than b

    monkeypatch.setattr(APP_CONFIG, "preview_disk_cache_max_bytes", sizes[0] + sizes[2] + MB // 4)
    store.put(_key("d"), _noise((256, 256, 3), 9), (256, 256), {})

    assert store.get(_key("b")) is None
    assert store.get(_key("c")) is None
    assert store.get(_key("a")) is not None and store.get(_key("d")) is not None
    count, total = store.usage()
    assert count == 2 and total <= APP_CONFIG.preview_disk_cache_max_bytes
    assert len([n for n in os.listdir(store.root) if n.endswith(".npc")]) == 2


def test_an_entry_larger_than_the_cap_is_not_written(disk_config, monkeypatch) -> None:
    monkeypatch.setattr(APP_CONFIG, "preview_disk_cache_max_bytes", 1024)
    store = _store(disk_config)

    assert not store.put(_key(), _noise((64, 64, 3)), (64, 64), {})
    assert store.usage() == (0, 0)


def test_the_free_space_floor_stops_writes(disk_config) -> None:
    store = _store(disk_config, min_free_bytes=1 << 62)

    assert not store.put(_key(), _noise((8, 8, 3)), (8, 8), {})
    assert store.get(_key()) is None


def test_a_corrupt_entry_is_a_miss_and_is_deleted(disk_config) -> None:
    store = _store(disk_config)
    store.put(_key(), _noise((64, 64, 3)), (64, 64), {})
    (path,) = [os.path.join(store.root, n) for n in os.listdir(store.root) if n.endswith(".npc")]
    with open(path, "r+b") as f:
        f.truncate(os.path.getsize(path) // 2)

    assert store.get(_key()) is None
    assert not os.path.exists(path)
    assert store.usage() == (0, 0)


def test_opening_the_folder_drops_orphans_and_stale_temporary_files(disk_config) -> None:
    store = _store(disk_config)
    store.put(_key(), _noise((8, 8, 3)), (8, 8), {})
    orphan = os.path.join(store.root, "0" * 40 + ".npc")
    stale = os.path.join(store.root, "x.npc.1.2.tmp")
    fresh = os.path.join(store.root, "y.npc.1.2.tmp")
    for path in (orphan, stale, fresh):
        open(path, "wb").close()
    os.utime(stale, (0, 0))

    reopened = _store(disk_config)
    reopened.put(_key("b"), _noise((8, 8, 3)), (8, 8), {})

    assert not os.path.exists(orphan) and not os.path.exists(stale)
    assert os.path.exists(fresh)
    assert reopened.get(_key()) is not None
    assert reopened.usage()[0] == 2


def test_two_stores_on_one_folder_share_entries(disk_config) -> None:
    first, second = _store(disk_config), _store(disk_config)
    first.put(_key(), _noise((8, 8, 3)), (8, 8), {})

    assert second.get(_key()) is not None
    second.clear()
    assert first.get(_key()) is None


def test_clear_empties_the_folder(disk_config) -> None:
    store = _store(disk_config)
    store.put(_key(), _noise((8, 8, 3)), (8, 8), {})

    store.clear()

    assert store.usage() == (0, 0)
    assert store.get(_key()) is None
    assert store.put(_key(), _noise((8, 8, 3)), (8, 8), {})


def test_a_disabled_disk_tier_writes_nothing(disk_config, monkeypatch) -> None:
    monkeypatch.setattr(APP_CONFIG, "preview_disk_cache", False)
    cache = PreviewBufferCache(APP_CONFIG, _store(disk_config))
    key = PreviewCacheKey("h", False, "Adobe RGB", False)

    cache.put(key, _noise((8, 8, 3)), (8, 8), {})

    assert not os.path.exists(pdc.cache_root(disk_config))


def test_full_resolution_entries_stay_in_ram(disk_config) -> None:
    store = _store(disk_config)
    cache = PreviewBufferCache(APP_CONFIG, store)
    cache.put(PreviewCacheKey("h", False, "Adobe RGB", True), _noise((8, 8, 3)), (8, 8), {})
    store.flush()

    assert store.usage() == (0, 0)


def _wrapper_loader(data: np.ndarray):
    loader = MagicMock()
    loader.get_loader.side_effect = lambda *a, **k: (NonStandardFileWrapper(data), {"color_space": "Adobe RGB", "orientation": 1})
    return loader


def test_a_new_session_reads_a_plain_frame_from_disk_without_decoding(disk_config) -> None:
    data = _noise((120, 160, 3))
    with patch("negpy.services.rendering.preview_manager.loader_factory", _wrapper_loader(data)) as loader:
        first, dims, _ = PreviewManager().load_linear_preview("/scan.tif", "Adobe RGB", file_hash="h1")
        pdc.shared_disk_cache(APP_CONFIG).flush()
        loader.get_loader.reset_mock()

        again, again_dims, _ = PreviewManager().load_linear_preview("/scan.tif", "Adobe RGB", file_hash="h1")

    loader.get_loader.assert_not_called()
    assert again_dims == dims
    assert np.array_equal(again, first)


def test_a_new_session_reads_a_merged_triplet_from_disk_without_decoding(disk_config, tmp_path) -> None:
    paths = [tmp_path / name for name in ("r.tif", "g.tif", "b.tif")]
    for path in paths:
        path.write_bytes(path.name.encode())
    rgbscan = RgbScanConfig(enabled=True, green_path=str(paths[1]), blue_path=str(paths[2]), align=False)

    def manager() -> PreviewManager:
        pm = PreviewManager()
        pm.load_linear_preview = MagicMock(side_effect=lambda path, *a, **k: (_noise((40, 60, 3), len(path)), (40, 60), {}))
        return pm

    first_pm = manager()
    merged, _, _ = first_pm.load_linear_preview_rgb(str(paths[0]), rgbscan, "Adobe RGB", file_hash="red")
    assert first_pm.load_linear_preview.call_count == 3
    pdc.shared_disk_cache(APP_CONFIG).flush()

    second_pm = manager()
    again, _, _ = second_pm.load_linear_preview_rgb(str(paths[0]), rgbscan, "Adobe RGB", file_hash="red")

    assert second_pm.load_linear_preview.call_count == 0
    assert np.array_equal(again, merged)

    paths[1].write_bytes(b"a new green exposure")
    third_pm = manager()
    third_pm.load_linear_preview_rgb(str(paths[0]), rgbscan, "Adobe RGB", file_hash="red")
    assert third_pm.load_linear_preview.call_count == 3


def test_a_prefetch_takes_a_disk_entry_without_the_decode_memory_check(disk_config) -> None:
    data = _noise((120, 160, 3))
    with patch("negpy.services.rendering.preview_manager.loader_factory", _wrapper_loader(data)) as loader:
        PreviewManager().load_linear_preview("/scan.tif", "Adobe RGB", file_hash="h1")
        pdc.shared_disk_cache(APP_CONFIG).flush()
        loader.reset_mock()

        pm = PreviewManager()
        with patch("negpy.services.rendering.preview_manager.decide_prefetch") as decide:
            assert pm.prefetch_linear_preview("/scan.tif", "Adobe RGB", use_camera_wb=False, file_hash="h1")

    decide.assert_not_called()
    loader.get_loader.assert_not_called()
    assert pm.peek_linear_preview("/scan.tif", "Adobe RGB", use_camera_wb=False, file_hash="h1") is not None
