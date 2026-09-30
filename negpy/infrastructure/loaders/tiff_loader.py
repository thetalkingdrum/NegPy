import os
from collections.abc import Callable
import imageio.v3 as iio
import numpy as np
import tifffile
from PIL import Image
from typing import Any, ContextManager, Optional, Tuple
from negpy.domain.interfaces import IImageLoader
from negpy.domain.models import ColorSpace
from negpy.infrastructure.display.icc_profile import (
    extract_gray_trc_decode_samples,
    extract_trc_decode_samples,
    is_matrix_trc_profile,
)
from negpy.kernel.image.logic import srgb_to_linear, uint8_to_float32, uint16_to_float32
from negpy.infrastructure.loaders.constants import IR_SIDECAR_SUFFIXES, SUPPORTED_TIFF_EXTENSIONS
from negpy.infrastructure.loaders.helpers import (
    NonStandardFileWrapper,
    _tiff_preview_page,
    bounded_tiff_page_preview,
    fit_bounded_preview,
    identify_color_space_from_icc,
    read_orientation,
)
from negpy.infrastructure.loaders.ir_planes import find_ir_plane, normalize_ir_to_float32
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)


def _ir_suffixes(token: str) -> tuple[str, ...]:
    """`_ir` → (`_ir.tif`, `_ir.tiff`). Keeps the read side tied to the token set that
    `is_ir_sidecar_path` hides from asset discovery."""
    assert token in IR_SIDECAR_SUFFIXES
    return tuple(token + e for e in sorted(SUPPORTED_TIFF_EXTENSIONS))


def _find_sidecar(dirname: str, entries: list[str], stem_lower: str, suffixes: tuple[str, ...]) -> Optional[str]:
    """First dir entry whose lowercased name == stem_lower + one of `suffixes` (suffixes already lowercase)."""
    for name in entries:
        if name.lower() in tuple(stem_lower + s for s in suffixes):
            return os.path.join(dirname, name)
    return None


def _read_sidecar_ir(file_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Read an IR sidecar and its optional validity mask fail-closed. Case-insensitive on the
    _ir token and the .tif/.tiff extension so scanner tools that emit lowercase names are picked up."""
    base, _ = os.path.splitext(file_path)
    dirname = os.path.dirname(file_path) or "."
    stem_lower = os.path.basename(base).lower()
    try:
        entries = os.listdir(dirname)
    except OSError:
        return None, None

    candidate = _find_sidecar(dirname, entries, stem_lower, _ir_suffixes("_ir"))
    if candidate is None:
        return None, None
    try:
        arr = tifffile.imread(candidate)
        ir = normalize_ir_to_float32(np.asarray(arr))
    except Exception as e:
        logger.warning(f"Failed to read IR sidecar {candidate}: {e}")
        return None, None

    mask_candidate = _find_sidecar(dirname, entries, stem_lower, _ir_suffixes("_ir_valid"))
    if mask_candidate is None:
        return ir, None
    try:
        valid = np.asarray(tifffile.imread(mask_candidate))
        if valid.shape != ir.shape:
            raise ValueError(f"mask shape {valid.shape} does not match IR shape {ir.shape}")
        if valid.dtype not in (np.dtype(np.bool_), np.dtype(np.uint8)):
            raise ValueError(f"mask dtype {valid.dtype} is not bool or uint8")
        if valid.dtype == np.uint8:
            rows_per_chunk = max(1, (1 << 20) // max(1, valid.shape[1]))
            for start in range(0, valid.shape[0], rows_per_chunk):
                chunk = valid[start : start + rows_per_chunk]
                if not np.all((chunk == 0) | (chunk == 1) | (chunk == 255)):
                    raise ValueError("uint8 mask values must be 0, 1, or 255")
        valid = valid.astype(np.bool_, copy=False)
        ir = ir.copy()
        ir[~valid] = 1.0
        return ir, valid
    except Exception as e:
        logger.warning(f"Failed to read IR validity mask {mask_candidate}; ignoring IR sidecar {candidate}: {e}")
        return None, None


def _read_ir_from_extra_page(file_path: str, main_h: int, main_w: int) -> Optional[np.ndarray]:
    """Finds a grayscale page at full resolution — SilverFast iSRD stores IR as page 2 with
    NewSubfileType=4. Page 0 is the main image here, so it is never a candidate."""
    try:
        with tifffile.TiffFile(file_path) as tif:
            return find_ir_plane(tif.pages[1:], main_h, main_w)
    except Exception as e:
        logger.warning(f"Failed to read extra-page IR from {file_path}: {e}")
    return None


def _extract_ir_from_extrasamples(file_path: str, img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Inspects ExtraSamples; returns (rgb, ir_or_none).

    Convention:
    - ExtraSamples[0] == 0 (UNSPECIFIED) → 4th plane is IR.
    - ExtraSamples[0] in (1, 2) (associated/unassociated alpha) → drop as alpha.
    - ExtraSamples tag missing → treat 4th plane as IR. Many scanner stacks (Nikon
      Coolscan via Nikon Scan, some VueScan profiles) emit 4-sample TIFFs without
      tagging the extra plane. A trailing alpha channel from a scanner is
      vanishingly rare; IR is the overwhelmingly common case.
    """
    if img.ndim != 3 or img.shape[2] != 4:
        return img, None

    extrasamples_kind: Optional[int] = None
    tag_present = False
    try:
        with tifffile.TiffFile(file_path) as tif:
            page = tif.pages[0]
            tags = getattr(page, "tags", None)
            tag = tags.get("ExtraSamples") if tags is not None else None
            if tag is not None and tag.value is not None:
                tag_present = True
                val = tag.value
                if isinstance(val, (tuple, list, np.ndarray)):
                    extrasamples_kind = int(val[0]) if len(val) > 0 else None
                else:
                    extrasamples_kind = int(val)
    except Exception as e:
        logger.warning(f"Failed to read ExtraSamples tag from {file_path}: {e}")

    is_ir = extrasamples_kind == 0 or not tag_present
    if is_ir:
        return np.ascontiguousarray(img[:, :, :3]), normalize_ir_to_float32(img[:, :, 3])
    return np.ascontiguousarray(img[:, :, :3]), None


_FLOAT_TRC_SAMPLES = 4096


def _decode_embedded_trc(icc_bytes: bytes, img: np.ndarray, f32: np.ndarray, file_path: str) -> Optional[np.ndarray]:
    """`f32` linearized through the profile's own TRC curves, or None (logged) when the
    profile has no curve to read: a LUT profile or an unsupported curve type.

    Integer data indexes a table sampled at every code value, so the decode is exact.
    """
    if img.dtype == np.uint8 or img.dtype == np.uint16:
        top = np.iinfo(img.dtype).max
        x = np.arange(top + 1, dtype=np.float64) / top
    else:
        x = np.linspace(0.0, 1.0, _FLOAT_TRC_SAMPLES)

    if is_matrix_trc_profile(icc_bytes):
        dec = extract_trc_decode_samples(icc_bytes, x)
        reason = "an unsupported TRC curve type"
    else:
        gray = extract_gray_trc_decode_samples(icc_bytes, x)
        dec = None if gray is None else np.stack([gray] * 3)
        reason = "no matrix/TRC or gray TRC curves (LUT-based or unreadable)"
    if dec is None:
        logger.warning(f"Embedded ICC profile in {file_path} has {reason}; its encoding is not decoded")
        return None

    dec = np.clip(dec, 0.0, 1.0)
    if np.max(np.abs(dec - x)) <= 0.5 / 65535.0:
        return f32
    lut = dec.astype(np.float32)
    out = np.empty_like(f32)
    for c in range(3):
        if img.dtype == np.uint8 or img.dtype == np.uint16:
            out[..., c] = lut[c][img[..., c]]
        else:
            out[..., c] = np.interp(f32[..., c], x, dec[c])
    return out


class TiffLoader(IImageLoader):
    """
    Loader for TIFF scans. Surfaces an IR channel via `metadata["ir"]` when present
    (either as a 4th sample with ExtraSamples=UNSPECIFIED, or via a `_IR.tif` sidecar).
    """

    def load(self, file_path: str, linear_raw: bool = False, positive_source: bool = False) -> Tuple[ContextManager[Any], dict]:
        img = iio.imread(file_path)
        ir: Optional[np.ndarray] = None
        ir_valid_mask: Optional[np.ndarray] = None

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 4:
            img, ir = _extract_ir_from_extrasamples(file_path, img)
        elif img.ndim == 3 and img.shape[2] > 4:
            img = img[:, :, :3]

        if ir is None:
            ir, ir_valid_mask = _read_sidecar_ir(file_path)

        if ir is None:
            ir = _read_ir_from_extra_page(file_path, img.shape[0], img.shape[1])

        if img.dtype == np.uint8:
            f32 = uint8_to_float32(np.ascontiguousarray(img))
        elif img.dtype == np.uint16:
            f32 = uint16_to_float32(np.ascontiguousarray(img))
        else:
            f32 = np.clip(img.astype(np.float32), 0, 1)

        icc_bytes: bytes | None = None
        try:
            with tifffile.TiffFile(file_path) as tif:
                page = tif.pages[0]
                tags = getattr(page, "tags", None)
                tag = tags.get("InterColorProfile") if tags is not None else None
                if tag is not None and tag.value:
                    icc_bytes = bytes(tag.value)
        except Exception:
            icc_bytes = None

        color_space = None
        if not linear_raw:
            # A label only, for "Same as Source": the decode reads the profile's own curves.
            color_space = identify_color_space_from_icc(icc_bytes)
            if color_space is None and (img.dtype == np.uint8 or positive_source):
                # Untagged 8-bit is display-encoded in practice. Untagged 16-bit is scanner-raw
                # linear, which no ColorSpace names, so it stays None; a positive source has
                # resolved that ambiguity and takes the 8-bit assumption.
                color_space = ColorSpace.SRGB.value
            decoded = _decode_embedded_trc(icc_bytes, img, f32, file_path) if icc_bytes else None
            if decoded is not None:
                f32 = decoded
            elif img.dtype == np.uint8 or positive_source:
                f32 = srgb_to_linear(f32)
        metadata = {
            "orientation": read_orientation(file_path),
            "color_space": color_space,
            "icc_profile": icc_bytes,
            "ir": ir,
            "ir_valid_mask": ir_valid_mask,
        }
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
        orientation = read_orientation(file_path)
        quick = _tiff_preview_page(file_path)
        if quick is not None:
            return fit_bounded_preview(quick, max_edge, orientation)
        if fast_only:
            return None

        with tifffile.TiffFile(file_path) as tif:
            candidates = []
            seen: set[int] = set()

            def collect(page: Any) -> None:
                offset = int(getattr(page, "offset", id(page)))
                if offset in seen:
                    return
                seen.add(offset)
                shape = tuple(int(value) for value in page.shape)
                if len(shape) in (2, 3) and (len(shape) == 2 or shape[2] in (1, 3, 4)):
                    channels = shape[2] if len(shape) == 3 else 1
                    candidates.append((channels >= 3, shape[0] * shape[1], page))
                for child in page.pages or ():
                    collect(child)

            for root in tif.pages:
                collect(root)
            if not candidates:
                return None
            page = max(candidates, key=lambda item: (item[0], item[1]))[2]
            preview = bounded_tiff_page_preview(page, max_edge, should_cancel=should_cancel)
        if preview is None:
            return None
        return fit_bounded_preview(preview, max_edge, orientation)
