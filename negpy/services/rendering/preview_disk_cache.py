"""Disk tier of the preview buffer cache: decoded linear previews kept between sessions.

An entry is lossless, because bounds measured on a preview carry into export. It is keyed by
the RAM cache key plus the identity of the decoder that made it, and every entry shares one
global byte cap, evicted least recently used first.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import shutil
import sqlite3
import struct
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Hashable, Optional

import imagecodecs
import numpy as np

from negpy.domain.types import AppConfig
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

# Bump in any change that alters a decoded preview for unchanged settings: a loader, the
# fast decode, orientation, the resize or the IR and detection planes. A release changes
# the identity anyway; this covers the builds between releases.
PREVIEW_DISK_CACHE_VERSION = 1

# The cache never fills the disk below this, whatever the cap.
MIN_FREE_BYTES = 2 * 1024**3

_MAGIC = b"NPXPC\x00\x01\x00"
_SUFFIX = ".npc"
_TEMP_SUFFIX = ".tmp"
_INDEX_NAME = "index.db"
# A temporary file this old belongs to a write that stopped, never to a live one.
_STALE_TEMP_SECONDS = 3600
# Writes wait in RAM; past this many the newest is dropped rather than queued.
_MAX_PENDING_WRITES = 4
# The full-resolution IR plane is a decode input. The preview carries its own `ir_preview`.
_DROPPED_KEYS = frozenset({"ir"})
_ARRAY_KINDS = frozenset("biuf")
_CODEC_THREADS = 2


class _Unencodable(Exception):
    """Metadata holds a value the typed encoding has no form for; the entry stays RAM-only."""


def decoder_identity(app_config: AppConfig) -> str:
    """What else decides a decoded preview, beyond the cache key."""
    import PIL
    import rawpy
    import tifffile

    from negpy import __version__

    return "|".join(
        (
            f"v{PREVIEW_DISK_CACHE_VERSION}",
            f"negpy {__version__}",
            f"rawpy {rawpy.__version__}",
            "libraw " + ".".join(str(part) for part in rawpy.libraw_version),
            f"tifffile {tifffile.__version__}",
            f"imagecodecs {imagecodecs.__version__}",
            f"pillow {PIL.__version__}",
            f"preview {int(app_config.preview_render_size)}",
        )
    )


def _dataclass_types() -> dict[str, type]:
    """The only classes a cache file may rebuild. Each holds plain values and tuples."""
    from negpy.features.lens.models import LensMetadata
    from negpy.features.lens.warps import PanasonicWarp, RectilinearWarp, SonyWarp

    return {cls.__name__: cls for cls in (LensMetadata, PanasonicWarp, RectilinearWarp, SonyWarp)}


def _encode(value: Any, arrays: list[np.ndarray], types: dict[str, type]) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in _ARRAY_KINDS:
            raise _Unencodable(f"array of {value.dtype}")
        arrays.append(np.ascontiguousarray(value))
        return {"$a": len(arrays) - 1}
    if isinstance(value, np.generic):
        if value.dtype.kind not in _ARRAY_KINDS:
            raise _Unencodable(f"scalar of {value.dtype}")
        return {"$s": value.dtype.str, "v": value.item()}
    if isinstance(value, bytes):
        return {"$b": base64.b64encode(value).decode("ascii")}
    if isinstance(value, tuple):
        return {"$t": [_encode(item, arrays, types) for item in value]}
    if isinstance(value, list):
        return [_encode(item, arrays, types) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise _Unencodable("dict with a non-string key")
        return {"$d": {k: _encode(v, arrays, types) for k, v in value.items()}}
    name = type(value).__name__
    if dataclasses.is_dataclass(value) and types.get(name) is type(value):
        return {"$c": name, "f": {f.name: _encode(getattr(value, f.name), arrays, types) for f in dataclasses.fields(value)}}
    raise _Unencodable(name)


def _decode(value: Any, arrays: list[np.ndarray], types: dict[str, type]) -> Any:
    if isinstance(value, list):
        return [_decode(item, arrays, types) for item in value]
    if not isinstance(value, dict):
        return value
    if "$a" in value:
        return arrays[int(value["$a"])]
    if "$s" in value:
        return np.dtype(value["$s"]).type(value["v"])
    if "$b" in value:
        return base64.b64decode(value["$b"])
    if "$t" in value:
        return tuple(_decode(item, arrays, types) for item in value["$t"])
    if "$d" in value:
        return {k: _decode(v, arrays, types) for k, v in value["$d"].items()}
    if "$c" in value:
        cls = types[value["$c"]]
        return cls(**{k: _decode(v, arrays, types) for k, v in value["f"].items()})
    raise ValueError("unknown node in a preview cache file")


def _key_text(identity: str, key: Hashable) -> str:
    return json.dumps([identity, key], separators=(",", ":"), default=repr)


class PreviewDiskCache:
    """Decoded previews in one folder, indexed by a small SQLite file beside them.

    The index and the folder may disagree after a crash or a hand edit: a row with no file is
    a miss, and a file with no row is deleted when the cache first opens in a process.
    """

    def __init__(self, root: str, app_config: AppConfig, *, min_free_bytes: int = MIN_FREE_BYTES) -> None:
        self.root = root
        self._app = app_config
        self._min_free = int(min_free_bytes)
        self._lock = threading.RLock()
        self._db: Optional[sqlite3.Connection] = None
        self._types: Optional[dict[str, type]] = None
        self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="preview-disk-cache")
        self._pending = 0
        self._reconciled = False

    @property
    def enabled(self) -> bool:
        return bool(self._app.preview_disk_cache) and self._app.preview_disk_cache_max_bytes > 0

    # Lookup

    def get(self, key: Hashable) -> Optional[tuple[np.ndarray, tuple, dict]]:
        if not self.enabled:
            return None
        key_text = _key_text(decoder_identity(self._app), key)
        name = hashlib.sha256(key_text.encode("utf-8")).hexdigest()[:40]
        path = os.path.join(self.root, name + _SUFFIX)
        try:
            with open(path, "rb") as f:
                blob = f.read()
        except FileNotFoundError:
            # Most misses are a first open, so the index is left alone; eviction drops a stale row.
            return None
        except OSError as e:
            logger.debug("preview disk cache read failed: %s", e)
            return None
        try:
            entry = self._parse(blob, key_text)
        except Exception as e:
            # A partial or foreign file is a miss, and goes so the next decode can write it again.
            logger.debug("preview disk cache drops unreadable %s: %s", name, e)
            self._remove(name)
            return None
        if entry is None:
            return None
        self._touch(name)
        return entry

    def _parse(self, blob: bytes, key_text: str) -> Optional[tuple[np.ndarray, tuple, dict]]:
        if blob[: len(_MAGIC)] != _MAGIC:
            raise ValueError("not a preview cache file")
        offset = len(_MAGIC)
        (header_len,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        header = json.loads(blob[offset : offset + header_len].decode("utf-8"))
        offset += header_len
        if header["key"] != key_text:
            # A different key that shares the file name: leave it for its owner.
            return None
        arrays: list[np.ndarray] = []
        for dtype, shape, size in header["arrays"]:
            out = np.empty(tuple(shape), dtype=np.dtype(dtype))
            chunk = blob[offset : offset + size]
            if len(chunk) != size:
                raise ValueError("truncated preview cache file")
            if out.size:
                imagecodecs.blosc2_decode(chunk, out=out.reshape(-1).view(np.uint8), numthreads=_CODEC_THREADS)
            arrays.append(out)
            offset += size
        types = self._dataclass_types()
        metadata = _decode(header["meta"], arrays, types)
        dims = tuple(int(v) for v in header["dims"])
        return arrays[0], dims, metadata

    # Store

    def put_async(self, key: Hashable, buffer: np.ndarray, dims: tuple, metadata: dict) -> Optional[Future]:
        """Write in the background; drops the write when too many already wait."""
        if not self.enabled:
            return None
        with self._lock:
            if self._pending >= _MAX_PENDING_WRITES:
                logger.debug("preview disk cache skip: %d writes waiting", self._pending)
                return None
            self._pending += 1
        identity = decoder_identity(self._app)

        def run() -> None:
            try:
                self._put(identity, key, buffer, dims, metadata)
            except Exception:
                logger.debug("preview disk cache write failed", exc_info=True)
            finally:
                with self._lock:
                    self._pending -= 1

        return self._writer.submit(run)

    def put(self, key: Hashable, buffer: np.ndarray, dims: tuple, metadata: dict) -> bool:
        """Write now, on the calling thread; True when the entry is on disk."""
        if not self.enabled:
            return False
        return self._put(decoder_identity(self._app), key, buffer, dims, metadata)

    def _put(self, identity: str, key: Hashable, buffer: np.ndarray, dims: tuple, metadata: dict) -> bool:
        self._reconcile_once()
        key_text = _key_text(identity, key)
        name = hashlib.sha256(key_text.encode("utf-8")).hexdigest()[:40]
        arrays: list[np.ndarray] = []
        try:
            _encode(np.asarray(buffer), arrays, {})
            meta = {k: v for k, v in metadata.items() if k not in _DROPPED_KEYS}
            encoded_meta = _encode(meta, arrays, self._dataclass_types())
        except _Unencodable as e:
            logger.debug("preview disk cache skip: metadata holds %s", e)
            return False

        blobs = [self._compress(arr) for arr in arrays]
        header = json.dumps(
            {
                "key": key_text,
                "dims": [int(v) for v in dims],
                "meta": encoded_meta,
                "arrays": [[arr.dtype.str, list(arr.shape), len(b)] for arr, b in zip(arrays, blobs)],
            },
            separators=(",", ":"),
        ).encode("utf-8")
        size = len(_MAGIC) + 4 + len(header) + sum(len(b) for b in blobs)
        if size > self._app.preview_disk_cache_max_bytes:
            return False
        os.makedirs(self.root, exist_ok=True)
        if shutil.disk_usage(self.root).free - size < self._min_free:
            logger.debug("preview disk cache skip: disk free space is below the floor")
            return False

        path = os.path.join(self.root, name + _SUFFIX)
        temp = f"{path}.{os.getpid()}.{threading.get_ident()}{_TEMP_SUFFIX}"
        try:
            with open(temp, "wb") as f:
                f.write(_MAGIC)
                f.write(struct.pack("<I", len(header)))
                f.write(header)
                for b in blobs:
                    f.write(b)
            os.replace(temp, path)
        except OSError as e:
            logger.debug("preview disk cache write failed: %s", e)
            try:
                os.remove(temp)
            except OSError:
                pass
            return False
        with self._lock:
            db = self._index()
            db.execute(
                "INSERT OR REPLACE INTO entries (name, bytes, used) VALUES (?, ?, ?)",
                (name, size, time.time()),
            )
            db.commit()
            self._evict(keep=name)
        return True

    @staticmethod
    def _compress(arr: np.ndarray) -> bytes:
        return bytes(
            imagecodecs.blosc2_encode(
                arr,
                level=1,
                compressor="zstd",
                shuffle="shuffle",
                typesize=arr.dtype.itemsize,
                numthreads=_CODEC_THREADS,
            )
        )

    # Housekeeping

    def usage(self) -> tuple[int, int]:
        """(entries, bytes) recorded in the index."""
        if not os.path.isdir(self.root):
            return 0, 0
        try:
            with self._lock:
                count, total = self._index().execute("SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM entries").fetchone()
            return int(count), int(total)
        except sqlite3.Error:
            return 0, 0

    def clear(self) -> None:
        """Deletes every entry; the folder stays usable."""
        self.flush()
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None
            shutil.rmtree(self.root, ignore_errors=True)
            self._reconciled = False
        logger.info("Preview disk cache cleared.")

    def flush(self, timeout: float = 30.0) -> None:
        """Waits for the writes already queued."""
        self._writer.submit(lambda: None).result(timeout=timeout)

    def evict_to_cap(self) -> None:
        with self._lock:
            self._evict(keep=None)

    def _index(self) -> sqlite3.Connection:
        if self._db is None:
            os.makedirs(self.root, exist_ok=True)
            db = sqlite3.connect(os.path.join(self.root, _INDEX_NAME), timeout=5.0, check_same_thread=False)
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS entries (name TEXT PRIMARY KEY, bytes INTEGER NOT NULL, used REAL NOT NULL)")
            db.commit()
            self._db = db
        return self._db

    def _touch(self, name: str) -> None:
        try:
            with self._lock:
                db = self._index()
                db.execute("UPDATE entries SET used = ? WHERE name = ?", (time.time(), name))
                db.commit()
        except sqlite3.Error as e:
            logger.debug("preview disk cache index update failed: %s", e)

    def _remove(self, name: str) -> None:
        try:
            os.remove(os.path.join(self.root, name + _SUFFIX))
        except OSError:
            pass
        try:
            with self._lock:
                db = self._index()
                db.execute("DELETE FROM entries WHERE name = ?", (name,))
                db.commit()
        except sqlite3.Error:
            pass

    def _evict(self, keep: Optional[str]) -> None:
        """Oldest first until the total fits the cap. Holds the lock."""
        db = self._index()
        cap = int(self._app.preview_disk_cache_max_bytes)
        (total,) = db.execute("SELECT COALESCE(SUM(bytes), 0) FROM entries").fetchone()
        if total <= cap:
            return
        for name, size in db.execute("SELECT name, bytes FROM entries ORDER BY used ASC").fetchall():
            if total <= cap:
                break
            if name == keep:
                continue
            try:
                os.remove(os.path.join(self.root, name + _SUFFIX))
            except FileNotFoundError:
                pass
            except OSError:
                # Open in another thread or process on Windows: the next eviction retries it.
                continue
            db.execute("DELETE FROM entries WHERE name = ?", (name,))
            total -= size
        db.commit()

    def _reconcile_once(self) -> None:
        """Brings the folder and the index back into agreement, once per process."""
        with self._lock:
            if self._reconciled:
                return
            self._reconciled = True
            try:
                names = os.listdir(self.root)
            except OSError:
                return
            now = time.time()
            on_disk: set[str] = set()
            for entry in names:
                full = os.path.join(self.root, entry)
                if entry.endswith(_SUFFIX):
                    on_disk.add(entry[: -len(_SUFFIX)])
                elif entry.endswith(_TEMP_SUFFIX):
                    try:
                        if now - os.path.getmtime(full) > _STALE_TEMP_SECONDS:
                            os.remove(full)
                    except OSError:
                        pass
            db = self._index()
            indexed = {name for (name,) in db.execute("SELECT name FROM entries")}
            for name in indexed - on_disk:
                db.execute("DELETE FROM entries WHERE name = ?", (name,))
            db.commit()
            for name in on_disk - indexed:
                try:
                    os.remove(os.path.join(self.root, name + _SUFFIX))
                except OSError:
                    pass
            self._evict(keep=None)

    def _dataclass_types(self) -> dict[str, type]:
        if self._types is None:
            self._types = _dataclass_types()
        return self._types


_SHARED: dict[str, PreviewDiskCache] = {}
_SHARED_LOCK = threading.Lock()


def cache_root(app_config: AppConfig) -> str:
    return app_config.preview_disk_cache_dir or os.path.join(app_config.cache_dir, "previews")


def shared_disk_cache(app_config: AppConfig) -> PreviewDiskCache:
    """One store per folder in a process, so every preview manager shares its writer and index."""
    root = cache_root(app_config)
    with _SHARED_LOCK:
        cache = _SHARED.get(root)
        if cache is None:
            cache = PreviewDiskCache(root, app_config)
            _SHARED[root] = cache
        return cache
