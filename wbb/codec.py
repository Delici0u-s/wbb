"""
codec.py — screencast payload -> H x W x 4 RGBA uint8.

Split out of browser.py because the decode path now has to handle two
container formats and the choice is per-bridge, not per-process.

Format is sniffed from the payload's magic bytes rather than taken from
the caller's `format=` request. That is one 8-byte comparison per frame
and it removes a class of bug where a bridge's requested format and the
decoder disagree (e.g. `Page.startScreencast` was restarted with a
different format by the pool, or a caller flipped it via raw CDP).

Alpha
-----
JPEG has no alpha channel; both decoders produce 0xFF there. PNG does,
and the Pillow path preserves it. libjpeg-turbo cannot decode PNG, so
the PNG path is Pillow-only regardless of whether PyTurboJPEG is
installed — see `browser.py`'s `screencast_format=` for the cost.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np
from PIL import Image

try:  # optional fast JPEG path
    from turbojpeg import TJPF_RGBA, TurboJPEG  # noqa: PLC0415

    _turbo: Optional["TurboJPEG"] = TurboJPEG()
except Exception:  # ImportError, or libturbojpeg.so missing at runtime
    _turbo = None
    TJPF_RGBA = None  # type: ignore[assignment]

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def turbojpeg_available() -> bool:
    return _turbo is not None


def _to_rgba_array(img: "Image.Image") -> np.ndarray:
    # `.convert("RGBA")` copies even when the image is already RGBA;
    # skipping that saves one full-frame allocation per frame on the
    # PNG path, which is the path that can least afford it.
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return np.asarray(img, dtype=np.uint8)


def decode_frame(data_b64: str, width: int, height: int) -> np.ndarray:
    """Decode one base64 screencast payload to an H x W x 4 uint8 RGBA array.

    Cost: one base64 decode (O(n) over the payload, one allocation), one
    image decode, and one resize *only* on a size mismatch. No per-frame
    allocation beyond the decoder's own output array on the common path.
    """
    raw = base64.b64decode(data_b64)

    if raw[:8] == _PNG_MAGIC:
        img = Image.open(io.BytesIO(raw))
        if img.size != (width, height):
            img = img.convert("RGBA").resize((width, height), Image.Resampling.LANCZOS)
        return _to_rgba_array(img)

    if _turbo is not None:
        rgba = _turbo.decode(raw, pixel_format=TJPF_RGBA)
        if rgba.shape[:2] == (height, width):
            return rgba
        return np.asarray(
            Image.fromarray(rgba, mode="RGBA").resize(
                (width, height), Image.Resampling.LANCZOS
            ),
            dtype=np.uint8,
        )

    img = Image.open(io.BytesIO(raw))
    if img.size != (width, height):
        img = img.convert("RGBA").resize((width, height), Image.Resampling.LANCZOS)
    return _to_rgba_array(img)


#: Back-compat alias. browser.py used to own this name; anything that
#: imported `wbb.browser._jpeg_to_rgba` keeps working.
_jpeg_to_rgba = decode_frame
