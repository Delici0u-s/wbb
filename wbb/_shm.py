"""
_shm.py — thin cross-platform wrapper around OS shared memory.

Uses ``multiprocessing.shared_memory.SharedMemory`` on all platforms so
the caller does not need to care about POSIX vs Windows SHM APIs. The
``name`` attribute of ``SharedMemory`` serves as the POSIX segment name
on Linux/macOS and a named file-mapping name on Windows.

``ShmSegment`` exposes a ``buf`` attribute that behaves like a
``memoryview``/``mmap`` into the segment.
"""

from __future__ import annotations

import logging
import os
from contextlib import suppress
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

log = logging.getLogger(__name__)


def unlink_if_exists(name: str) -> bool:
    """Remove a POSIX shared-memory segment if it is there. Never raises.

    Returns True if a segment was removed. Useful for cleaning up after a
    process that died before its own `unlink()` ran — a Ctrl-C at the
    wrong moment, or a hard kill.
    """
    try:
        stale = SharedMemory(name=name, create=False)
    except FileNotFoundError:
        return False
    except Exception:
        return False
    try:
        stale.close()
    except Exception:
        pass
    try:
        stale.unlink()
        return True
    except Exception:
        return False


def _open_segment(
    name: str, size: int, *, attach: bool, on_conflict: str
) -> SharedMemory:
    """Create or attach to a segment, handling a stale leftover by name.

    POSIX shared memory outlives the process that made it. A run killed
    before `FrameBuffer.unlink()` leaves `/dev/shm/<name>_a` behind, and
    every later run with the same buffer name then dies at startup with
    `FileExistsError: /overlay_a` — which says nothing about what to do
    about it.

    `on_conflict` decides:

    ``"replace"`` (default)
        Unlink the leftover and create a fresh segment, logging a
        warning that names it. Right for the overwhelmingly common case
        (the previous run crashed). Wrong if another *live* process is
        genuinely using that name — POSIX gives no way to tell, so the
        warning is the only signal you get.
    ``"attach"``
        Adopt the existing segment instead, if its size matches.
    ``"error"``
        Re-raise, with a message that names the segment and the fix.
    """
    if attach:
        return SharedMemory(name=name, create=False)

    try:
        return SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        pass

    if on_conflict == "error":
        raise FileExistsError(
            f"shared-memory segment {name!r} already exists. A previous run "
            f"probably died before cleaning up. Remove it with "
            f"wbb.buffer.FrameBuffer.cleanup({name.rsplit('_', 1)[0]!r}), or "
            f"pass on_conflict='replace'."
        )

    if on_conflict == "attach":
        existing = SharedMemory(name=name, create=False)
        if existing.size >= size:
            resource_tracker.unregister(existing._name, "shared_memory")
            return existing
        existing.close()
        log.warning(
            "shared-memory segment %r exists but is too small (%d < %d); "
            "replacing it", name, existing.size, size
        )
        unlink_if_exists(name)
        return SharedMemory(name=name, create=True, size=size)

    if on_conflict != "replace":
        raise ValueError(
            f"on_conflict must be 'replace', 'attach' or 'error', got {on_conflict!r}"
        )

    log.warning(
        "shared-memory segment %r already existed (left behind by a process "
        "that did not unlink it); replacing it. If another wbb process is "
        "live and using this buffer name, give one of them a different name.",
        name,
    )
    unlink_if_exists(name)
    return SharedMemory(name=name, create=True, size=size)


class ShmSegment:
    """
    Wrapper around :class:`multiprocessing.shared_memory.SharedMemory`.

    Parameters
    ----------
    name:
        Segment name. Must be unique per system session.
    size:
        Size in bytes. Ignored when *attach* is True.
    attach:
        If True, connect to an existing segment (create=False).
        If False, create a new one (create=True).

    Notes
    -----
    ``buf`` returns the *same* memoryview instance on every access rather
    than re-exporting one from the underlying mmap each time. numpy views
    built with ``np.frombuffer(seg.buf, ...)`` hold an export on whatever
    memoryview they were given; if every ``.buf`` access minted a new one,
    each numpy array would pin a separate export and the mmap could never
    be closed cleanly. Call :meth:`close` (which releases this cached
    view first) rather than reaching into ``_shm`` directly.

    Resource-tracker note
    ----------------------
    ``multiprocessing.shared_memory.SharedMemory`` registers every
    segment it opens — including attached, non-owned ones — with the
    current process's resource tracker for crash-safety cleanup. That
    means an *attaching* process (``attach=True``) would otherwise have
    its own tracker unlink the segment on exit even though it never
    created it and even if it only ever calls ``close()`` — racing with,
    or pre-empting, the owning process's own ``unlink()``. We immediately
    unregister right after attaching to opt this process's tracker out
    of cleanup duty for memory it doesn't own. The owning process
    (``attach=False``) is left registered as normal, since stdlib's own
    ``SharedMemory.unlink()`` already unregisters correctly when *it*
    calls unlink — see CPython bpo-38119 for the underlying upstream
    wart this works around.
    """

    def __init__(
        self,
        name: str,
        size: int,
        *,
        attach: bool = False,
        on_conflict: str = "replace",
    ) -> None:
        self._name = name
        self._shm = _open_segment(name, size, attach=attach, on_conflict=on_conflict)
        self._buf: Optional[memoryview] = self._shm.buf
        self._closed = False

        if attach:
            # Opt this process's tracker out of unlink duty for a segment
            # it merely attached to — see class docstring "Resource-
            # tracker note" above for why this is necessary.
            resource_tracker.unregister(self._shm._name, "shared_memory")

    @property
    def buf(self) -> memoryview:
        if self._buf is None:
            raise RuntimeError(f"Segment '{self._name}' is closed")
        return self._buf

    def close(self) -> None:
        """Release the cached memoryview, then close the mapping.

        Where the BufferError actually comes from
        -----------------------------------------
        Measured, not assumed: `np.frombuffer(mv, ...)` does **not** hold
        an export on `mv` — it holds one on the *mmap underneath* it. So
        `self._buf.release()` succeeds even with live numpy views, and
        the `BufferError: cannot close exported pointers exist` is raised
        one line later by `SharedMemory.close()` -> `mmap.close()`.

        That matters for recovery: the old code set `self._buf = None`
        and only then hit the failing close, leaving the segment in a
        state where `seg.buf` raises "is closed" while the mapping and
        its file descriptor are still very much open, and no retry could
        ever succeed. `_closed` is now only set on a close that actually
        completed, so a second `close()` after the caller drops their
        Frame does the right thing.
        """
        buf, self._buf = self._buf, None
        if buf is not None:
            with suppress(BufferError):
                buf.release()
        try:
            self._shm.close()
        except BufferError:
            # A caller (or numpy view) still exports the mapping, so the
            # mmap cannot be unmapped — and must not be, because those
            # views point into it. What CPython's SharedMemory.close()
            # does on this path is worse than nothing: it raises before
            # closing the file descriptor, so a retry is impossible and
            # the fd leaks for the life of the process.
            #
            # Do the two things that are actually safe: close the fd
            # ourselves, and drop SharedMemory's reference to the mmap so
            # the mapping is freed by refcount when the last exporter
            # goes away. Detaching it also stops SharedMemory.__del__
            # from re-raising the same BufferError as an "exception
            # ignored in deallocator" traceback at interpreter shutdown.
            shm = self._shm
            shm._mmap = None  # type: ignore[attr-defined]
            fd = getattr(shm, "_fd", -1)
            if fd is not None and fd >= 0:
                with suppress(OSError):
                    os.close(fd)
                shm._fd = -1  # type: ignore[attr-defined]
            self._closed = True
            raise
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def unlink(self) -> None:
        self._shm.unlink()

    @property
    def name(self) -> str:
        return self._name
