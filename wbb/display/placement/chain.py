"""
select_backend() — try each PlacementBackend in priority order, keep
the first one that activates.

Order: KWin D-Bus first (works on KDE Plasma, X11 or Wayland session
alike — see kwin.py's module docstring), then X11/EWMH (covers non-KDE
X11 desktops; naturally skips itself under Wayland since there's no
X11 display to open), then the no-op terminal fallback (GNOME-Wayland,
macOS, Windows, or any KDE box with KWin scripting disabled /
python-xlib missing).

Each backend's activate() is required (by the PlacementBackend
protocol in _base.py) to fail cleanly rather than raise; this function
adds one more layer of defense — an unexpected exception from a
backend is logged and treated as a declined activation, same as a
clean `return False`, rather than taking the whole DisplayClient down.
"""

from __future__ import annotations

import logging

from ._base import NativeHandle, PlacementBackend
from .kwin import KWinPlacement
from .none import NoPlacement
from .x11_ewmh import X11Placement

log = logging.getLogger(__name__)


def select_backend(handle: NativeHandle, *, wm_class: str) -> PlacementBackend:
    chain: list[PlacementBackend] = [
        KWinPlacement(wm_class),
        X11Placement(),
        NoPlacement(),
    ]
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
