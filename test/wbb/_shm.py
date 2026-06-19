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

from multiprocessing.shared_memory import SharedMemory
from typing import Optional


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
    """

    def __init__(self, name: str, size: int, *, attach: bool = False) -> None:
        self._name = name
        self._shm = SharedMemory(name=name, create=not attach, size=size)
        self._buf: Optional[memoryview] = self._shm.buf

    @property
    def buf(self) -> memoryview:
        if self._buf is None:
            raise RuntimeError(f"Segment '{self._name}' is closed")
        return self._buf

    def close(self) -> None:
        if self._buf is not None:
            self._buf.release()
            self._buf = None
        self._shm.close()

    def unlink(self) -> None:
        self._shm.unlink()

    @property
    def name(self) -> str:
        return self._name
