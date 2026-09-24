"""
Win32Placement — always-on-top, absolute position and click-through for
a native Windows window, via plain user32 calls (no extra dependency).

Unlike X11 and Wayland, Win32 lets a client place its own top-level
window directly, so there is exactly one mechanism per capability and
no fallback chain inside this backend:

* always-on-top  -> ``SetWindowPos(HWND_TOPMOST / HWND_NOTOPMOST)``
* position       -> ``SetWindowPos`` with the client-area origin
                    translated to the window-rect origin
* readback       -> ``ClientToScreen(hwnd, (0, 0))``
* click-through  -> ``WS_EX_LAYERED | WS_EX_TRANSPARENT``

Coordinates
-----------
``set_position``/``actual_position`` use the *client-area* origin, the
same point SDL reports as the window position and the same point the
X11 backend moves. ``SetWindowPos`` itself takes the outer window rect,
so for a window with a frame the frame offset is measured and subtracted
on every call (it can change with DPI, theme, or style changes).

All values are physical pixels. That only holds because
``_window._ensure_sdl_init`` makes the process per-monitor-DPI-aware
before SDL starts; without it Windows virtualises coordinates on scaled
monitors and every position would be off by the scale factor.
"""

from __future__ import annotations

import ctypes
import logging
from ctypes import wintypes
from typing import Optional

from .. import _win32
from ._base import NativeHandle

log = logging.getLogger(__name__)


class Win32Placement:
    name = "win32"

    def __init__(self) -> None:
        self._hwnd: Optional[int] = None
        self._active = False
        #: True when set_click_through added WS_EX_LAYERED itself, so
        #: disabling it removes only what this backend added.
        self._added_layered = False

    # ------------------------------------------------------------------
    def activate(self, handle: NativeHandle) -> bool:
        if handle.subsystem != "windows" or not handle.win32_hwnd:
            return False
        if not _win32.available():
            log.debug("win32 backend: user32 not available; skipping")
            return False
        if not _win32.api().IsWindow(handle.win32_hwnd):
            log.debug("win32 backend: handle is not a live window; skipping")
            return False
        self._hwnd = handle.win32_hwnd
        self._active = True
        log.info("Placement: using the win32 backend")
        return True

    # ------------------------------------------------------------------
    def set_above(self, above: bool) -> None:
        if not self._active:
            return
        insert_after = _win32.HWND_TOPMOST if above else _win32.HWND_NOTOPMOST
        ok = _win32.api().SetWindowPos(
            self._hwnd,
            insert_after,
            0,
            0,
            0,
            0,
            _win32.SWP_NOMOVE | _win32.SWP_NOSIZE | _win32.SWP_NOACTIVATE,
        )
        if not ok:
            log.debug("win32: SetWindowPos(topmost=%s) failed (error %d)", above, _win32.last_error())

    def _frame_offset(self) -> tuple[int, int]:
        """Client-area origin minus window-rect origin, in pixels."""
        a = _win32.api()
        rect = wintypes.RECT()
        origin = wintypes.POINT(0, 0)
        if not a.GetWindowRect(self._hwnd, ctypes.byref(rect)):
            return (0, 0)
        if not a.ClientToScreen(self._hwnd, ctypes.byref(origin)):
            return (0, 0)
        return (origin.x - rect.left, origin.y - rect.top)

    def set_position(self, x: int, y: int, width: int, height: int) -> None:
        if not self._active:
            return
        dx, dy = self._frame_offset()
        ok = _win32.api().SetWindowPos(
            self._hwnd,
            None,
            int(x) - dx,
            int(y) - dy,
            0,
            0,
            _win32.SWP_NOSIZE | _win32.SWP_NOZORDER | _win32.SWP_NOACTIVATE,
        )
        if not ok:
            log.debug("win32: SetWindowPos(move) failed (error %d)", _win32.last_error())

    def supports_position(self) -> bool:
        return self._active

    def actual_position(self) -> Optional[tuple[int, int]]:
        if not self._active:
            return None
        origin = wintypes.POINT(0, 0)
        if not _win32.api().ClientToScreen(self._hwnd, ctypes.byref(origin)):
            return None
        return (origin.x, origin.y)

    def position_method(self) -> str:
        return "setwindowpos"

    def next_position_method(self) -> bool:
        return False

    # ------------------------------------------------------------------
    def set_click_through(self, enabled: bool) -> bool:
        """
        ``WS_EX_TRANSPARENT`` alone does nothing useful for hit-testing;
        it only passes input through when combined with
        ``WS_EX_LAYERED``. Adding ``WS_EX_LAYERED`` to a window that is
        *not* already driven by UpdateLayeredWindow makes it invisible
        until its layering is defined, so in that case it gets
        ``SetLayeredWindowAttributes(alpha=255)`` — fully opaque, content
        unchanged.

        That call must NOT be made on an alpha window: it would switch
        the window to the SetLayeredWindowAttributes model and make
        every later UpdateLayeredWindow fail. An alpha window already
        has WS_EX_LAYERED (set by LayeredPresenter), which is what the
        check below keys on.

        On disable, WS_EX_LAYERED is removed again only if this method
        added it.
        """
        if not self._active or self._hwnd is None:
            return False
        a = _win32.api()
        try:
            style = _win32.get_exstyle(self._hwnd)
            if enabled:
                was_layered = bool(style & _win32.WS_EX_LAYERED)
                _win32.set_exstyle(
                    self._hwnd, style | _win32.WS_EX_LAYERED | _win32.WS_EX_TRANSPARENT
                )
                if not was_layered:
                    self._added_layered = True
                    if not a.SetLayeredWindowAttributes(self._hwnd, 0, 255, _win32.LWA_ALPHA):
                        log.warning(
                            "click_through: SetLayeredWindowAttributes failed "
                            "(error %d)",
                            _win32.last_error(),
                        )
                        return False
            else:
                style &= ~_win32.WS_EX_TRANSPARENT
                if self._added_layered:
                    style &= ~_win32.WS_EX_LAYERED
                    self._added_layered = False
                _win32.set_exstyle(self._hwnd, style)
            return True
        except Exception:
            log.exception("click_through: win32 style change failed")
            return False
