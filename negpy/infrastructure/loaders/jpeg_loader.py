from collections.abc import Callable
from typing import Any, ContextManager, Optional, Tuple

import imageio.v3 as iio
import numpy as np
from PIL import Image

from negpy.domain.interfaces import IImageLoader
from negpy.domain.models import ColorSpace
from negpy.infrastructure.loaders.helpers import (
    NonStandardFileWrapper,
    decode_via_own_profile,
    fit_bounded_preview,
    identify_color_space_from_icc,
    read_orientation,
    resolve_srgb_to_xyz,
)
from negpy.kernel.image.logic import apply_linear_primaries_transform, srgb_to_linear, uint8_to_float32


class JpegLoader(IImageLoader):
    """
    Loader for JPEG scans.
    """

    def load(self, file_path: str) -> Tuple[ContextManager[Any], dict]:
        img = iio.imread(file_path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        if img.dtype == np.uint8:
            f32 = uint8_to_float32(np.ascontiguousarray(img))
        else:
            f32 = np.clip(img.astype(np.float32) / 255.0, 0, 1)

        icc_bytes: bytes | None = None
        try:
            with Image.open(file_path) as pil_img:
                icc_bytes = pil_img.info.get("icc_profile")
        except Exception:
            icc_bytes = None

        color_space = identify_color_space_from_icc(icc_bytes) or ColorSpace.SRGB.value
        own_profile = decode_via_own_profile(f32, icc_bytes)
        if own_profile is not None:
            f32 = own_profile
        elif color_space == ColorSpace.SRGB.value:
            f32 = srgb_to_linear(f32)
            f32 = apply_linear_primaries_transform(f32, resolve_srgb_to_xyz(icc_bytes))
        metadata = {"orientation": read_orientation(file_path), "color_space": color_space, "icc_profile": icc_bytes, "ir": None}
        return NonStandardFileWrapper(f32), metadata

    def load_bounded_preview(
        self,
        file_path: str,
        max_edge: int,
        *,
        fast_only: bool = False,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> Optional[Image.Image]:
        if should_cancel is not None and should_cancel():
            raise InterruptedError("preview cancelled")
        with Image.open(file_path) as image:
            image.draft("RGB", (max_edge, max_edge))
            if image.width * image.height * 3 > 64 * 1024 * 1024:
                return None
            return fit_bounded_preview(image, max_edge, read_orientation(file_path))
