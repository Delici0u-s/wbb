"""
Frame — a snapshot of one rendered frame.

A Frame is a lightweight, read-only view into the FrameBuffer's backing
memory. It is not a copy unless the caller explicitly asks for one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True, slots=True)
class Frame:
    """
    A single rendered frame.

    Attributes
    ----------
    data:
        H×W×4 uint8 RGBA numpy array. Read-only view into shared memory;
        zero-copy. Call ``copy()`` to get a writable array.
    width, height:
        Frame dimensions in pixels.
    frame_id:
        Monotonically increasing counter — unique within a FrameBuffer
        session. Two Frame objects with the same ``frame_id`` represent
        the same rendered frame.
    timestamp:
        Wall-clock time (``time.monotonic()``) at which the frame was
        written into the buffer.
    """

    data: np.ndarray
    width: int
    height: int
    frame_id: int
    timestamp: float = field(default_factory=time.monotonic)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def copy(self) -> np.ndarray:
        """Return a writable copy of the RGBA array."""
        return np.asarray(self.data.copy())

    def crop(self, x: int, y: int, w: int, h: int) -> "Frame":
        """
        Return a new Frame whose data is a zero-copy view of the
        sub-region ``[y:y+h, x:x+w]``.

        The returned Frame shares the same ``frame_id`` and ``timestamp``
        as the source. No pixel data is copied.
        """
        region = self.data[y : y + h, x : x + w]
        return Frame(
            data=region,
            width=w,
            height=h,
            frame_id=self.frame_id,
            timestamp=self.timestamp,
        )

    def save(self, path: str | Path, *, format: Optional[str] = None) -> None:
        """
        Save the frame to *path*.

        The image format is inferred from the file extension unless
        *format* is given (e.g. ``"PNG"``, ``"JPEG"``).

        Requires Pillow (already a hard dependency of wbb).
        """
        from PIL import Image  # noqa: PLC0415

        img = Image.fromarray(self.data, mode="RGBA")
        img.save(str(path), format=format)

    def __repr__(self) -> str:
        return (
            f"Frame(id={self.frame_id}, "
            f"{self.width}×{self.height}, "
            f"t={self.timestamp:.3f})"
        )
