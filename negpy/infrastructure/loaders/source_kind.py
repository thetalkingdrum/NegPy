"""A Slide that is not a camera RAW is always Positive: the as-captured transfer reads a camera
RAW's sensor-white decode, which a scan or a finished image does not have."""

import functools
import os
from typing import Any

from negpy.domain.models import WorkspaceConfig
from negpy.features.hdr.models import hdr_active
from negpy.features.process.models import ProcessMode, with_positive_source
from negpy.features.rgbscan.models import is_rgb_triplet
from negpy.features.stitch.models import stitch_active
from negpy.infrastructure.loaders.factory import loader_factory


@functools.lru_cache(maxsize=4096)
def _checked_scanned_positive_source(path: str) -> bool:
    return not loader_factory.is_camera_raw(path)


def is_scanned_positive_source(path: str) -> bool:
    """A TIFF, JPEG, JXL or a film scanner's own raw (Pakon, Noritsu, Coolscan NEF, Flextight
    FFF). Cached per path once the file is readable, because a Coolscan NEF or Flextight FFF is told
    apart by opening it; an unreadable file is not cached, so a volume that comes back is read again."""
    if not path:
        return False
    if not os.access(path, os.R_OK):
        return not loader_factory.is_camera_raw(path)
    return _checked_scanned_positive_source(path)


is_scanned_positive_source.cache_clear = _checked_scanned_positive_source.cache_clear


def _is_composite(config: WorkspaceConfig) -> bool:
    """A merged or assembled frame keeps its saved Positive choice: it renders on the Transfer path."""
    return hdr_active(config.hdr) or stitch_active(config.stitch) or is_rgb_triplet(config.rgbscan)


def with_source_positive(config: WorkspaceConfig, path: str) -> WorkspaceConfig:
    """*config* with Positive on when it is a Slide of a scanned source."""
    if config.process.process_mode != ProcessMode.E6 or config.process.positive_source or _is_composite(config) or not is_scanned_positive_source(path):
        return config
    return with_positive_source(config, True)


def with_source_positive_defaults(defaults: dict[str, Any], path: str) -> dict[str, Any]:
    """A roll's defaults as they apply to the frame at *path*, so its forced Positive never
    reads as a divergence from the roll's Film card."""
    if defaults.get("process_mode") == ProcessMode.E6 and defaults.get("positive_source") is False and is_scanned_positive_source(path):
        return {**defaults, "positive_source": True}
    return defaults
