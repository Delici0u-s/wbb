"""
window_type.py — EWMH `_NET_WM_WINDOW_TYPE` for the SDL window.

Why this is a *pre-map* operation
---------------------------------
EWMH says the window type is read when the WM starts managing the
window, i.e. at MapRequest. Whether a given WM re-evaluates it on a
later PropertyNotify is unspecified and, for KWin's stacking *layer*
specifically, unverified (see the open questions in the review that
introduced this module). So `SDLWindow` creates the window
`SDL_WINDOW_HIDDEN`, writes the property while it is still unmapped,
and only then calls `SDL_ShowWindow`. That is a real map, first time,
with the property already in place — no post-map poke, no re-map.

Degradation
-----------
`apply_window_type` returns False and logs at debug level when the
subsystem is not x11, python-xlib is missing, or the server rejects
the write. It never raises. On a non-KDE WM the KDE-specific atom is
simply an atom the WM does not recognise; EWMH requires a WM to use
the first type in the list it *does* recognise, which is why
CRITICAL_NOTIFICATION ships `_NET_WM_WINDOW_TYPE_NOTIFICATION` as the
second element.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

from .x11_props import set_atom_list_property, set_cardinal_property

log = logging.getLogger(__name__)


class WindowType(str, Enum):
    """What the window claims to be, for WM stacking/decoration policy.

    NORMAL
        Leave whatever SDL set (SDL's X11 backend writes
        `_NET_WM_WINDOW_TYPE_NORMAL` itself). No property write happens.
    UTILITY, DOCK, NOTIFICATION
        Plain EWMH types.
    CRITICAL_NOTIFICATION
        KDE's `_KDE_NET_WM_WINDOW_TYPE_CRITICAL_NOTIFICATION`, with
        `_NET_WM_WINDOW_TYPE_NOTIFICATION` appended as the standard
        fallback. On KWin this places the window in
        CriticalNotificationLayer, above ActiveLayer, so a focused
        fullscreen window does not cover it. On every other WM it
        behaves exactly as NOTIFICATION.

    Caveat that applies to NOTIFICATION and CRITICAL_NOTIFICATION: most
    WMs do not give notification windows keyboard focus and keep them
    out of the taskbar and Alt+Tab. If you rely on
    `DisplayClient(on_key_event=...)`, do not use these types.
    """

    NORMAL = "normal"
    UTILITY = "utility"
    DOCK = "dock"
    NOTIFICATION = "notification"
    CRITICAL_NOTIFICATION = "critical-notification"


_TYPE_ATOMS: dict[WindowType, tuple[str, ...]] = {
    WindowType.UTILITY: ("_NET_WM_WINDOW_TYPE_UTILITY",),
    WindowType.DOCK: ("_NET_WM_WINDOW_TYPE_DOCK",),
    WindowType.NOTIFICATION: ("_NET_WM_WINDOW_TYPE_NOTIFICATION",),
    WindowType.CRITICAL_NOTIFICATION: (
        "_KDE_NET_WM_WINDOW_TYPE_CRITICAL_NOTIFICATION",
        "_NET_WM_WINDOW_TYPE_NOTIFICATION",
    ),
}

#: gamescope's nested X server reads this CARDINAL off a client window to
#: composite it as an overlay on top of the game. Only meaningful when
#: `DISPLAY` (or `display_name=`) points at gamescope's nested server.
GAMESCOPE_OVERLAY_PROPERTY = "GAMESCOPE_EXTERNAL_OVERLAY"


def needs_pre_map(window_type: WindowType) -> bool:
    """True if this type requires the window to be created hidden."""
    return window_type in _TYPE_ATOMS


def apply_window_type(
    handle: object,
    window_type: WindowType,
    *,
    display_name: Optional[str] = None,
) -> bool:
    """Write `_NET_WM_WINDOW_TYPE` for `handle`. Never raises.

    `handle` is a `placement.NativeHandle`. Returns True only if the
    property was actually written.
    """
    atoms = _TYPE_ATOMS.get(window_type)
    if atoms is None:
        return False

    subsystem = getattr(handle, "subsystem", "unknown")
    window_id = getattr(handle, "x11_window", None)
    if subsystem != "x11" or not window_id:
        log.debug(
            "window_type=%s ignored: needs an X11 window (subsystem=%r). "
            "Under a native Wayland session there is no client-side window-type "
            "protocol; this is a documented no-op.",
            window_type.value,
            subsystem,
        )
        return False

    ok = set_atom_list_property(
        int(window_id), "_NET_WM_WINDOW_TYPE", atoms, display_name=display_name
    )
    if ok:
        log.info("window_type=%s applied (%s)", window_type.value, ", ".join(atoms))
    else:
        log.debug("window_type=%s could not be applied", window_type.value)
    return ok


def apply_gamescope_overlay(
    handle: object,
    enabled: bool = True,
    *,
    display_name: Optional[str] = None,
) -> bool:
    """Set `GAMESCOPE_EXTERNAL_OVERLAY` on the window. Never raises.

    Unverified against a running gamescope — see the open questions. It
    is here because it is the same one-line property write as
    `apply_window_type` and costs nothing to expose; treat a False
    return, or no visible effect, as "gamescope is not the compositor
    on this display".
    """
    window_id = getattr(handle, "x11_window", None)
    if getattr(handle, "subsystem", "") != "x11" or not window_id:
        return False
    return set_cardinal_property(
        int(window_id),
        GAMESCOPE_OVERLAY_PROPERTY,
        [1 if enabled else 0],
        display_name=display_name,
    )
