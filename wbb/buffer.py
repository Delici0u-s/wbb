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
import gc
import logging
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import AsyncIterator, Optional

import numpy as np

from wbb.frame import Frame
from wbb._shm import ShmSegment, unlink_if_exists  # thin OS-agnostic wrapper (see _shm.py)

log = logging.getLogger(__name__)


# Layout of the metadata segment (little-endian):
#   [0]      uint8  — active buffer index (0=A, 1=B)
#   [1..8]   uint64 — frame_id of the frame currently in the active buffer
#   [9..16]  double — timestamp (time.monotonic())
_META_FMT = "<BQd"
_META_SIZE = struct.calcsize(_META_FMT)


#: Retry budget for copy_latest()'s seqlock. Each attempt is one
#: full-frame memcpy plus two 17-byte metadata reads.
_COPY_RETRIES = 32


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
        on_conflict: str = "replace",
    ) -> None:
        """
        `on_conflict` decides what happens when segments with this name
        already exist (`attach=False` only):

        ``"replace"`` (default)
            Unlink the leftovers and create fresh ones, with a warning
            naming each. POSIX shared memory outlives the process that
            created it, so a run killed before `unlink()` leaves
            `/dev/shm/<name>_a` behind and every later run with the same
            buffer name used to die at startup with a bare
            `FileExistsError`.
        ``"attach"``
            Adopt the existing segments if they are large enough.
        ``"error"``
            The old behaviour, with a message that says what to do.

        `"replace"` is wrong if a *live* process is using the same
        buffer name — POSIX offers no way to distinguish that from a
        leftover, so the warning is the only signal. Give concurrent
        buffers distinct names.
        """
        self.name = name
        self.width = width
        self.height = height
        self._attach = attach

        self._frame_bytes = width * height * 4  # RGBA

        # Shared-memory segments
        self._shm_a = ShmSegment(
            f"{name}_a", self._frame_bytes, attach=attach, on_conflict=on_conflict
        )
        self._shm_b = ShmSegment(
            f"{name}_b", self._frame_bytes, attach=attach, on_conflict=on_conflict
        )
        self._shm_meta = ShmSegment(
            f"{name}_meta", _META_SIZE, attach=attach, on_conflict=on_conflict
        )

        # numpy views (zero-copy) into each buffer
        self._arr_a = np.frombuffer(self._shm_a.buf, dtype=np.uint8).reshape((height, width, 4))
        self._arr_b = np.frombuffer(self._shm_b.buf, dtype=np.uint8).reshape((height, width, 4))

        # writer-side: which buffer is being written next
        self._write_index: int = 0
        self._frame_counter: int = 0

        # reader notification
        # self._new_frame_event = threading.Event()
        self._lock = threading.Lock()  # protects _frame_counter on write side
        self._cv = threading.Condition()
        self._generation = 0  # bumped on every write, never reset

        # Dedicated executor for next_frame()'s blocking Condition wait.
        # Created lazily on first await (so a pure writer process never
        # spins one up). The point: a parked reader must NOT consume a
        # worker from asyncio's shared default executor, because that
        # same default pool is where BrowserBridge runs JPEG decode. With
        # the old run_in_executor(None, ...), N blocked readers across N
        # DisplayClients could pin all cpu+4 default workers and starve
        # decode. A private pool parks waiters off to the side instead.
        #
        # Sized small but > 1 so several coroutines in the *same* process
        # (e.g. a DisplayClient + a recorder + a monitor all awaiting the
        # same buffer, as in example 06) can park concurrently without
        # serialising behind one wait thread. Threads are idle-blocked,
        # not burning CPU, so this is cheap. Closed in close().
        self._wait_executor: Optional[ThreadPoolExecutor] = None
        self._wait_executor_lock = threading.Lock()
        self._closed = False

    # ------------------------------------------------------------------
    # Writer interface (called by BrowserBridge)
    # ------------------------------------------------------------------

    def write(self, rgba: np.ndarray) -> int:
        if rgba.shape != (self.height, self.width, 4):
            raise ValueError(f"Expected shape ({self.height}, {self.width}, 4), got {rgba.shape}")

        # next_idx is only ever touched by the writer thread (BrowserBridge
        # drives this from a single executor at a time), so reading/flipping
        # it doesn't need the lock either -- only the metadata write +
        # frame_counter increment need to be atomic w.r.t. concurrent readers.
        next_idx = self._write_index ^ 1
        target = self._arr_a if next_idx == 0 else self._arr_b

        # Memcpy happens OUTSIDE the lock. Readers only ever read the
        # *currently active* buffer (per the metadata), never the inactive
        # one we're writing into here, so there's no race to guard against.
        np.copyto(target, rgba)

        with self._lock:
            self._frame_counter += 1
            fid = self._frame_counter
            ts = time.monotonic()
            packed = struct.pack(_META_FMT, next_idx, fid, ts)
            self._shm_meta.buf[:_META_SIZE] = packed
            self._write_index = next_idx

        with self._cv:
            self._generation += 1
            self._cv.notify_all()
        # self._new_frame_event.set()
        # self._new_frame_event.clear()
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
        if self._arr_a is None or self._arr_b is None:
            raise RuntimeError(
                f"FrameBuffer {self.name!r} is closed; read() after close() would "
                "hand out a view into an unmapped segment."
            )
        raw = bytes(self._shm_meta.buf[:_META_SIZE])
        idx, fid, ts = struct.unpack(_META_FMT, raw)
        arr = self._arr_a if idx == 0 else self._arr_b
        view = arr.view()
        view.flags.writeable = False
        return Frame(data=view, width=self.width, height=self.height, frame_id=fid, timestamp=ts)

    def copy_latest(self, out: Optional[np.ndarray] = None) -> Optional[Frame]:
        """Copy the latest frame out of shared memory, detached and stable.

        Why this exists
        ---------------
        `read()` returns a zero-copy view, and the double buffer only
        protects a reader that consumes it *immediately*. Two writes
        (A -> B -> A) put the writer back into the segment a slow reader
        is still looking at, and `write()` memcpys into it with no
        interlock — so any consumer that holds a Frame across more than
        one write interval can be handed a half-updated frame. A filter
        chain running in a thread-pool executor is exactly that
        consumer.

        This copies under a seqlock-style check: read the metadata, copy,
        re-read the metadata, and retry if the writer moved underneath.
        Returns None if it could not get a clean copy in a few attempts
        (only possible if the writer is producing faster than a memcpy,
        i.e. never in practice).

        Cost: one H*W*4 memcpy (~3.6 MB, sub-millisecond at 720p) plus
        two 17-byte metadata reads. Pass `out=` — a preallocated array of
        the buffer's shape — for zero per-frame allocation.
        """
        if self._arr_a is None or self._arr_b is None:
            raise RuntimeError(f"FrameBuffer {self.name!r} is closed")
        if out is None:
            out = np.empty((self.height, self.width, 4), dtype=np.uint8)
        # A writer running flat out in another thread can invalidate a
        # copy several times in a row, so the retry budget has to be
        # generous and has to yield between attempts — with the GIL, a
        # tight retry loop can starve the very writer it is waiting for.
        for attempt in range(_COPY_RETRIES):
            if attempt:
                time.sleep(0)
            raw = bytes(self._shm_meta.buf[:_META_SIZE])
            idx, fid, ts = struct.unpack(_META_FMT, raw)
            np.copyto(out, self._arr_a if idx == 0 else self._arr_b)
            raw2 = bytes(self._shm_meta.buf[:_META_SIZE])
            if raw2 == raw:
                view = out.view()
                view.flags.writeable = False
                return Frame(
                    data=view, width=self.width, height=self.height,
                    frame_id=fid, timestamp=ts,
                )
        # Never happens against a paced writer; if it does, the caller
        # falls back to the zero-copy view, which is what it would have
        # used anyway before this method existed.
        log.debug("copy_latest: writer outran the reader after %d attempts",
                  _COPY_RETRIES)
        return None

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    def _ensure_wait_executor(self) -> ThreadPoolExecutor:
        # Double-checked lazy init; next_frame can be called from
        # concurrent coroutines.
        ex = self._wait_executor
        if ex is not None:
            return ex
        with self._wait_executor_lock:
            if self._wait_executor is None:
                self._wait_executor = ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix=f"wbb-wait-{self.name}"
                )
            return self._wait_executor

    async def next_frame(self, timeout: float = 5.0) -> Frame:
        loop = asyncio.get_running_loop()

        def _wait_for_new_generation(seen_gen: int) -> bool:
            with self._cv:
                return self._cv.wait_for(lambda: self._generation != seen_gen, timeout=timeout)

        seen_gen = self._generation
        await loop.run_in_executor(
            self._ensure_wait_executor(), _wait_for_new_generation, seen_gen
        )
        return self.read()

    def __del__(self) -> None:
        """Best-effort release for a buffer that was never closed.

        Not a substitute for `close()`/`unlink()` — this cannot unlink,
        because a segment may legitimately outlive this process. What it
        does is stop CPython's own `SharedMemory.__del__` from printing
        `BufferError: cannot close exported pointers exist` as an
        "exception ignored in deallocator" traceback for every abandoned
        buffer, which is noise that says nothing useful and looks like a
        crash.
        """
        try:
            if not getattr(self, "_closed", True):
                self.close(collect=False)
        except Exception:
            pass

    @staticmethod
    def cleanup(name: str) -> int:
        """Remove leftover segments for `name`. Returns how many were removed.

        For cleaning up after a process that died before its own
        `unlink()` ran. Safe to call when nothing is there.
        """
        return sum(
            unlink_if_exists(f"{name}_{suffix}") for suffix in ("a", "b", "meta")
        )

    def wake(self) -> None:
        """Wake every parked next_frame() without committing a new frame.

        The generation counter is bumped, so a waiter returns
        immediately and re-reads the currently committed frame — same
        frame_id as before, which is how a caller tells "woken" from
        "new frame". Used by DisplayClient.request_repaint(); previously
        the only way to do this was to bump `_generation` under `_cv`
        yourself.

        O(1), no allocation, safe from any thread.
        """
        with self._cv:
            self._generation += 1
            self._cv.notify_all()

    async def __aiter__(self) -> AsyncIterator[Frame]:
        while True:
            yield await self.next_frame()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self, *, collect: bool = True) -> None:
        """
        Release this process's memory mappings. Call once when done.

        `collect=True` (default) runs one `gc.collect()` after dropping
        this object's own views and before releasing the segments. That
        reclaims Frames that are only reachable from a reference cycle
        or from the interpreter's last-expression slot — the common
        real-world cause of the BufferError below — without changing the
        contract: a Frame you are still holding in a live local still
        blocks the release, and still only logs.

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
        # Wake parked waiters BEFORE tearing anything down. The old order
        # nulled the arrays first, so a waiter woken by the notify below
        # went straight into read() and hit an AttributeError on a None
        # array — a shutdown-ordering crash in the reader, not the writer.
        # read() now also raises a clear RuntimeError if it loses that
        # race anyway.
        self._closed = True
        with self._cv:
            self._generation += 1
            self._cv.notify_all()

        self._arr_a = None  # type: ignore[assignment]
        self._arr_b = None  # type: ignore[assignment]
        if collect:
            gc.collect()

        if self._wait_executor is not None:
            self._wait_executor.shutdown(wait=False)
            self._wait_executor = None

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
