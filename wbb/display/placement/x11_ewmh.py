"""
X11Placement — always-on-top via the EWMH ``_NET_WM_STATE_ABOVE``
client message, position via plain ``XMoveWindow``.

Unlike the KWin backend, this is a real, stable, decades-old protocol
— every EWMH-compliant window manager (i3, Openbox, Mutter's X11 mode,
KWin's X11 mode, etc.) honors it the same way. It is offered *after*
the KWin backend in the default chain (see ``chain.py``) purely
because KWin's own mechanism also works under KWin's X11 mode and is
already active by the time this would be tried — not because this is
less correct. On non-KDE X11 desktops, this is the one that actually
fires.

Does nothing under Wayland: there is no X11 display to connect to,
and ``activate()`` reports that cleanly rather than raising — see
``XOpenDisplay`` returning ``None``.

Requires ``python-xlib``.
"""

from __future__ import annotations

import logging
from typing import Optional

from ._base import NativeHandle

log = logging.getLogger(__name__)


class X11Placement:
    name = "x11-ewmh"

    def __init__(self) -> None:
        self._disp = None
        self._window = None
        self._active = False

    # ------------------------------------------------------------------
    def activate(self, handle: NativeHandle) -> bool:
        if handle.subsystem != "x11" or handle.x11_window is None:
            return False

        try:
            from Xlib import display as xdisplay  # noqa: PLC0415
        except ImportError:
            log.debug("x11-ewmh backend: python-xlib not installed; skipping")
            return False

        try:
            self._disp = xdisplay.Display()
            self._window = self._disp.create_resource_object("window", handle.x11_window)
        except Exception:
            log.debug("x11-ewmh backend: could not open X display; skipping")
            return False

        self._active = True
        log.info("Placement: using X11/EWMH (_NET_WM_STATE_ABOVE + XMoveWindow)")
        return True

    # ------------------------------------------------------------------
    def set_above(self, above: bool) -> None:
        if not self._active or self._disp is None or self._window is None:
            return
        from Xlib import X  # noqa: PLC0415
        from Xlib.protocol import event as xevent  # noqa: PLC0415

        root = self._disp.screen().root
        net_wm_state = self._disp.intern_atom("_NET_WM_STATE")
        net_wm_state_above = self._disp.intern_atom("_NET_WM_STATE_ABOVE")

        # _NET_WM_STATE client message, per the EWMH spec:
        # data.l[0]: 0=remove, 1=add, 2=toggle
        ev = xevent.ClientMessage(
            window=self._window,
            client_type=net_wm_state,
            data=(32, [1 if above else 0, net_wm_state_above, 0, 0, 0]),
        )
        mask = X.SubstructureNotifyMask | X.SubstructureRedirectMask
        root.send_event(ev, event_mask=mask)
        self._disp.flush()

    def set_position(self, x: int, y: int, width: int, height: int) -> None:
        if not self._active or self._window is None:
            return
        self._window.configure(x=x, y=y)
        self._disp.flush()

    def supports_position(self) -> bool:
        return self._active

    def set_click_through(self, enabled: bool) -> bool:
        """
        Real implementation, via the X11 Shape extension's *input*
        shape (``SK.Input``) — the same primitive
        ``Gdk.Surface.set_input_region()`` used under the hood in the
        old GTK4 client. Setting the input shape to an empty rectangle
        list means no rectangle of this window ever receives pointer
        events, so they fall through to whatever is stacked below —
        this is the actual mechanism, not a GTK-specific trick, which
        is why it transfers directly to a raw Xlib call here.

        ``enabled=False`` restores normal input handling by setting
        the input shape back to "the whole window" (a single rectangle
        covering width×height) rather than literally clearing the
        shape extension state — XShape has no single "remove shape and
        go back to default" call that's simpler than just re-asserting
        the full-window rectangle, so that's what this does.

        python-xlib auto-discovers and loads the SHAPE extension's
        methods onto ``Window`` objects at ``Display()`` connection
        time, *only if the connected X server actually advertises it*
        (see ``Display.__init__``'s extension-discovery loop) — so
        ``shape_rectangles`` simply won't exist as an attribute on
        ``self._window`` if the server lacks SHAPE, which is why this
        checks with ``hasattr`` rather than importing
        ``Xlib.ext.shape`` and assuming it applies (importing the
        module doesn't register anything by itself; only a live,
        SHAPE-advertising connection does).
        """
        if not self._active or self._window is None or self._disp is None:
            return False
        if not hasattr(self._window, "shape_rectangles"):
            log.warning(
                "click_through requested but the connected X server does "
                "not advertise the SHAPE extension"
            )
            return False

        from Xlib.ext import shape  # noqa: PLC0415 — only for the SO/SK enums, not for registration

        try:
            if enabled:
                # Empty list -> the input region is empty -> every
                # pointer event passes through. This is the X11
                # equivalent of Gdk.Surface.set_input_region(empty).
                self._window.shape_rectangles(shape.SO.Set, shape.SK.Input, 0, 0, 0, [])
            else:
                geom = self._window.get_geometry()
                self._window.shape_rectangles(
                    shape.SO.Set,
                    shape.SK.Input,
                    0,
                    0,
                    0,
                    [(0, 0, geom.width, geom.height)],
                )
            self._disp.flush()
            return True
        except Exception:
            log.exception("click_through: XShape call failed")
            return False
