"""
FrameBuffer — shared-memory pixel buffer.

Architecture
------------
* Two POSIX named shared-memory segments (``<name>_a`` and ``<name>_b``)
  act as a double-buffer.  The writer alternates between them; readers
  always attach to the segment currently marked "current" via a tiny
  metadata segment (``<name>_meta``).
* The metadata segment contains a single byte: 0 → current is A, 1 → B.
  Writers flip the byte after finishing a write so readers always see a
  complete frame.
* All read/write coordination uses a ``threading.Event`` so that
  ``await buf.next_frame()`` and async iteration are driven by actual frame
  arrival rather than polling.

Cross-process usage
-------------------
A second process creates a FrameBuffer with ``attach=True`` and the same
*name*.  It maps both segments and the metadata segment read-only and can
read frames without any shared locks — the double-buffer guarantees that a
reader never observes a partial write.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import threading
from contextlib import suppress
from typing import AsyncIterator, Optional

import numpy as np

from wbb.frame import Frame
from wbb._shm import ShmSegment  # thin OS-agnostic wrapper (see _shm.py)

log = logging.getLogger(__name__)


# Layout of the metadata segment (little-endian):
#   [0]      uint8  — active buffer index (0=A, 1=B)
#   [1..8]   uint64 — frame_id of the frame currently in the active buffer
#   [9..16]  double — timestamp (time.monotonic())
_META_FMT = "<BQd"
_META_SIZE = struct.calcsize(_META_FMT)


class FrameBuffer:
    """
    Shared, mutable pixel buffer.

    Parameters
    ----------
    name:
        A short identifier for the shared-memory segments. Two processes
        using the same name attach to the same buffer.
    width, height:
        Frame dimensions. Must match what the writer pushes.
    attach:
        If True, connect to existing segments created by another process.
        If False (default), create the segments and own their lifecycle.
    """

    def __init__(
        self,
        name: str,
        width: int,
        height: int,
        *,
        attach: bool = False,
    ) -> None:
        self.name = name
        self.width = width
        self.height = height
        self._attach = attach

        self._frame_bytes = width * height * 4  # RGBA

        # Shared-memory segments
        self._shm_a = ShmSegment(f"{name}_a", self._frame_bytes, attach=attach)
        self._shm_b = ShmSegment(f"{name}_b", self._frame_bytes, attach=attach)
        self._shm_meta = ShmSegment(f"{name}_meta", _META_SIZE, attach=attach)

        # numpy views (zero-copy) into each buffer
        self._arr_a = np.frombuffer(self._shm_a.buf, dtype=np.uint8).reshape((height, width, 4))
        self._arr_b = np.frombuffer(self._shm_b.buf, dtype=np.uint8).reshape((height, width, 4))

        # writer-side: which buffer is being written next
        self._write_index: int = 0
        self._frame_counter: int = 0

        # reader notification
        self._new_frame_event = threading.Event()
        self._lock = threading.Lock()  # protects _frame_counter on write side

    # ------------------------------------------------------------------
    # Writer interface (called by BrowserBridge)
    # ------------------------------------------------------------------

    def write(self, rgba: np.ndarray) -> int:
        """
        Write *rgba* (H×W×4 uint8) into the inactive buffer, then flip.

        Returns the new frame_id.
        """
        import time

        if rgba.shape != (self.height, self.width, 4):
            raise ValueError(f"Expected shape ({self.height}, {self.width}, 4), got {rgba.shape}")

        with self._lock:
            self._frame_counter += 1
            fid = self._frame_counter
            ts = time.monotonic()
            next_idx = self._write_index ^ 1

            # write into the currently inactive buffer
            target = self._arr_a if next_idx == 0 else self._arr_b
            np.copyto(target, rgba)

            # update metadata atomically (single byte flip last)
            packed = struct.pack(_META_FMT, next_idx, fid, ts)
            self._shm_meta.buf[:_META_SIZE] = packed

            self._write_index = next_idx

        self._new_frame_event.set()
        self._new_frame_event.clear()
        return fid

    # ------------------------------------------------------------------
    # Reader interface
    # ------------------------------------------------------------------

    def read(self) -> Frame:
        """
        Return the latest frame as a zero-copy view.

        The returned :class:`Frame` holds a direct reference into shared
        memory; it becomes stale (silently shows old pixels) once the
        writer next flips, and it must be released — by scope exit,
        reassignment, or ``del`` — before this buffer's ``close()`` can
        fully release the underlying mapping (see ``FrameBuffer.close``).
        Call ``frame.copy()`` to detach a frame you want to keep.
        """
        raw = bytes(self._shm_meta.buf[:_META_SIZE])
        idx, fid, ts = struct.unpack(_META_FMT, raw)
        arr = self._arr_a if idx == 0 else self._arr_b
        view = arr.view()
        view.flags.writeable = False
        return Frame(data=view, width=self.width, height=self.height, frame_id=fid, timestamp=ts)

    # def wait_for_frame(self, timeout: Optional[float] = None) -> Frame:
    #     """Block until the next new frame arrives, then return it."""
    #     self._new_frame_event.wait(timeout=timeout)
    #     return self.read()

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def next_frame(self, timeout: float = 5.0) -> Frame:
        """Async wait for the next frame.

        Returns the latest frame via self.read() if no new frame
        arrives within `timeout` seconds.
        """
        loop = asyncio.get_running_loop()
        got_frame = await loop.run_in_executor(None, self._new_frame_event.wait, timeout)
        # got_frame is False on timeout, True if the event was set
        return self.read()

    async def __aiter__(self) -> AsyncIterator[Frame]:
        while True:
            yield await self.next_frame()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Release this process's memory mappings. Call once when done.

        Lifetime contract
        ------------------
        ``read()`` and async iteration return zero-copy views into shared
        memory — required behaviour per the buffer contract, not an
        optimisation detail. This has a real consequence: CPython will
        not unmap memory while any array still holds a live buffer-
        protocol export on it (the same rule that applies to ``mmap``
        generally — it exists to stop dangling pointers, not to be
        clever). Concretely: **drop every Frame and every array derived
        from one (including via `.crop()`) before calling `close()`**.
        Reassigning the variable, letting it go out of scope, or
        ``del frame`` are all sufficient; simply finishing your last use
        of it inside the same still-live local variable is not.

        If you need a Frame's data to outlive the buffer, call
        ``frame.copy()`` first — that detaches it from shared memory
        entirely.

        ``close()`` itself drops this object's own internal references
        and best-effort releases the underlying segments; it logs
        (rather than raises) if a release could not complete because the
        caller is still holding a view, since that is recoverable once
        the caller drops it and lets normal garbage collection proceed.
        """
        self._arr_a = None  # type: ignore[assignment]
        self._arr_b = None  # type: ignore[assignment]

        for seg in (self._shm_a, self._shm_b, self._shm_meta):
            try:
                seg.close()
            except BufferError:
                log.debug(
                    "Segment '%s' still has an outstanding Frame/array "
                    "reference; it will be released once that reference "
                    "is dropped (see FrameBuffer.close docs).",
                    seg.name,
                )

    def unlink(self) -> None:
        """
        Destroy the underlying OS resources (remove the named segments).

        Only the process that created the segments (``attach=False``)
        should call this. Readers should call ``close()`` only.

        ``unlink()`` removes the *name* so no new process can attach to
        it; it does not require this process's own mapping to be
        released first, so it succeeds even if ``close()`` logged a
        warning about an outstanding Frame reference. The already-mapped
        memory in this process is freed by the OS once this process
        exits, even if ``close()`` never fully completed.
        """
        for seg in (self._shm_a, self._shm_b, self._shm_meta):
            with suppress(Exception):
                seg.unlink()

    def __enter__(self) -> "FrameBuffer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
        if not self._attach:
            self.unlink()
