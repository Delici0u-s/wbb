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
        log.warning(
            "Placement: no backend available for this session (subsystem=%r). "
            "always_on_top / set_position() will have no effect. On KDE "
            "Plasma, install pydbus (pip install pydbus) and make sure KWin "
            "scripting is enabled. On other X11 desktops, install "
            "python-xlib (pip install python-xlib).",
            handle.subsystem,
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
