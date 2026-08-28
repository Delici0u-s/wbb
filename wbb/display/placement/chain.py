"""
select_backend() — pick the placement backend that fits this window.

Priority is decided by the window's *subsystem*, not by a fixed global
order. That distinction matters on KDE, where both backends are
available and they are not equivalent:

* An **X11 window** (including one on XWayland, which is what SDL2
  produces on a Wayland session whenever it selects the `x11` video
  driver) gets X11/EWMH first. It is a stable, decades-old protocol; it
  can read the window's real geometry back, so a refused move is
  detectable rather than silent; it has several positioning mechanisms
  to fall back through; and it is the only backend that can do
  click-through, via the SHAPE extension. KWin's scripting API has none
  of that.
* A **native Wayland surface** has no X11 window to act on, so KWin's
  D-Bus scripting is the only lever that exists and goes first.

The previous order put KWin first unconditionally. On KDE that meant an
XWayland window — the common case on a Plasma Wayland session — never
reached the X11 backend at all, silently losing click-through and
geometry readback even though both were fully available. KWin still runs
as the fallback for an X11 window whose X11 backend declined (no
python-xlib installed, no reachable display), which is the case that
ordering was presumably meant to serve.

`WBB_PLACEMENT=x11|kwin|none` forces one backend and skips the rest.
For reproducing a report from a machine you do not have, and for
checking whether a placement problem is the backend's fault or the
window's; not a supported configuration surface.

Each backend's activate() is required (by the PlacementBackend protocol
in _base.py) to fail cleanly rather than raise; this function adds one
more layer of defense — an unexpected exception from a backend is logged
and treated as a declined activation, same as a clean `return False`,
rather than taking the whole DisplayClient down.
"""

from __future__ import annotations

import logging
import os

from ._base import NativeHandle, PlacementBackend
from .kwin import KWinPlacement
from .none import NoPlacement
from .x11_ewmh import X11Placement

log = logging.getLogger(__name__)


def _forced() -> str:
    return os.environ.get("WBB_PLACEMENT", "").strip().lower()


def select_backend(handle: NativeHandle, *, wm_class: str) -> PlacementBackend:
    x11 = X11Placement()
    kwin = KWinPlacement(wm_class)

    forced = _forced()
    if forced:
        chosen = {"x11": [x11], "kwin": [kwin], "none": []}.get(forced)
        if chosen is None:
            log.warning(
                "WBB_PLACEMENT=%r is not one of x11/kwin/none; ignoring it.", forced
            )
        else:
            log.info("Placement: WBB_PLACEMENT=%r forces the backend choice", forced)
            chain: list[PlacementBackend] = [*chosen, NoPlacement()]
            return _first_activating(chain, handle)

    # Subsystem decides the order; see the module docstring.
    chain = [x11, kwin] if handle.subsystem == "x11" else [kwin, x11]
    chain.append(NoPlacement())
    return _first_activating(chain, handle)


def _first_activating(
    chain: "list[PlacementBackend]", handle: NativeHandle
) -> PlacementBackend:
    for backend in chain:
        try:
            if backend.activate(handle):
                return backend
        except Exception:
            log.exception(
                "Placement backend %r raised during activate(); treating as "
                "declined and trying the next backend.",
                backend.name,
            )
    # Unreachable in practice — NoPlacement.activate() always returns
    # True — but keeps the return type honest if that ever changes.
    return NoPlacement()
