"""
NoPlacement — terminal fallback. Always "activates" (there's nothing
to fail), every call is a no-op. Logs exactly once, at activation
time, so callers who asked for always_on_top/position get a single
clear explanation instead of silent nothing or a per-call spam of
warnings.
"""

from __future__ import annotations

import logging

from ._base import NativeHandle

log = logging.getLogger(__name__)


class NoPlacement:
    name = "none"

    def activate(self, handle: NativeHandle) -> bool:
        if handle.subsystem == "wayland":
            hint = (
                "This is a Wayland session, where there is NO protocol for a "
                "client to set its own absolute position or stay-on-top — the "
                "only mechanism is KDE's KWin D-Bus scripting backend, which "
                "needs pydbus: `pip install pydbus` (PyGObject must also be "
                "present, which it usually is on KDE). Installing python-xlib "
                "will NOT help here: there is no X11 window for it to act on. "
                "If you are on KDE Plasma, also ensure KWin scripting is "
                "enabled. (click_through specifically is unsupported even with "
                "KWin — see placement/kwin.py.) If you genuinely need X11-style "
                "placement, start the app under XWayland / an X11 session so "
                "SDL2 selects its x11 subsystem and the EWMH backend applies."
            )
        elif handle.subsystem == "x11":
            hint = (
                "This is an X11 session but the EWMH backend did not activate — "
                "install python-xlib (`pip install python-xlib`). On KDE you can "
                "alternatively install pydbus for the KWin backend."
            )
        else:
            hint = (
                "Unknown windowing subsystem. On KDE Plasma install pydbus "
                "(`pip install pydbus`); on other X11 desktops install "
                "python-xlib (`pip install python-xlib`)."
            )
        log.warning(
            "Placement: no backend available for this session (subsystem=%r). "
            "always_on_top / set_position() will have no effect. %s",
            handle.subsystem,
            hint,
        )
        return True

    def set_above(self, above: bool) -> None:
        pass

    def set_position(self, x: int, y: int, width: int, height: int) -> None:
        pass

    def supports_position(self) -> bool:
        return False

    def set_click_through(self, enabled: bool) -> bool:
        log.warning(
            "click_through requested but no placement backend is active "
            "(see the earlier 'no backend available' warning); ignoring."
        )
        return False
