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


def _u32(value: int) -> int:
    """Two's-complement `value` into the unsigned slot python-xlib packs.

    A format-32 ClientMessage payload is packed with `array('I', ...)`
    (see `Xlib.protocol.rq.PropertyData.pack_value`), which raises
    OverflowError on a negative number rather than reinterpreting it. So
    a perfectly ordinary global coordinate — a monitor to the left of
    the primary has a negative x, one mounted higher gives its neighbour
    a negative y — used to take the whole render loop down with it.
    Masking here produces the same 32 bits the X server reads back as a
    signed INT32, which is what the protocol expects on the wire.
    """
    return int(value) & 0xFFFFFFFF


class X11Placement:
    name = "x11-ewmh"

    #: Positioning mechanisms, in the order they are tried. Ordered by
    #: how well specified they are rather than by how likely they are to
    #: work: the EWMH client message is the one window managers are
    #: asked to honour exactly, the move-only ConfigureRequest is the
    #: conservative fallback, and the full-geometry ConfigureRequest is
    #: last because a WM enforcing SDL's size hints can reject it whole.
    _METHODS = ("ewmh", "move", "configure")

    def __init__(self) -> None:
        self._disp = None
        self._window = None
        self._active = False
        self._method_index = 0

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
        """Move the window using the currently selected mechanism.

        There is no single mechanism every window manager honours, so
        this owns several and the caller advances between them based on
        a readback (see `next_position_method` and
        DisplayClient._settle_placement). Trying them blindly in one
        call is not an option: if two mechanisms are both honoured, the
        second overwrites the first, and if one is reinterpreted by the
        WM's placement policy it can undo a move that already worked.
        One at a time, measured, is the only version that converges.
        """
        if not self._active or self._window is None or self._disp is None:
            return
        method = self._METHODS[self._method_index]
        try:
            if method == "ewmh":
                self._send_moveresize(x, y, width, height)
            elif method == "move":
                self._send_configure(x, y, None, None)
            else:
                self._send_configure(x, y, width, height)
        except Exception:
            log.debug(
                "x11-ewmh: position mechanism %r raised; it will be "
                "advanced past on the next settle check",
                method,
                exc_info=True,
            )

    def position_method(self) -> str:
        return self._METHODS[self._method_index]

    def next_position_method(self) -> bool:
        """Advance to the next mechanism, or False when exhausted.

        Monotonic, so repeated calls terminate. Not reset by a later
        move: once the readback has shown which mechanism this window
        manager actually honours, every subsequent `set_position()`
        uses it directly and pays nothing for the search.
        """
        if self._method_index + 1 >= len(self._METHODS):
            return False
        self._method_index += 1
        log.info(
            "x11-ewmh: switching position mechanism to %r",
            self._METHODS[self._method_index],
        )
        return True

    def _send_moveresize(self, x: int, y: int, width: int, height: int) -> None:
        """EWMH `_NET_MOVERESIZE_WINDOW` client message to the root window.

        The EWMH-blessed way for a client to position a window it owns
        (freedesktop wm-spec, "Other Root Window Messages"). Window
        managers treat it like a ConfigureRequest but are asked to
        honour the exact geometry, and with StaticGravity the (x, y) is
        the client window's top-left in root coordinates regardless of
        decorations — the same reference `actual_position()` measures.
        This mirrors `set_above()`, which is also an EWMH client
        message rather than a direct property poke.

        data.l[0] = gravity (low byte) + presence/source flags:
          gravity          = StaticGravity (10)   -> bits 0..7
          x,y,w,h present  = 0xF00                -> bits 8..11
          source           = pager/taskbar (0b10) -> bits 12..15
        """
        from Xlib import X  # noqa: PLC0415
        from Xlib.protocol import event as xevent  # noqa: PLC0415

        STATIC_GRAVITY = 10
        flags = STATIC_GRAVITY | (0x1 << 8) | (0x1 << 9) | (0x1 << 10) | (0x1 << 11)
        flags |= 0x2 << 12  # source indication: pager/taskbar

        root = self._disp.screen().root
        atom = self._disp.intern_atom("_NET_MOVERESIZE_WINDOW")
        ev = xevent.ClientMessage(
            window=self._window,
            client_type=atom,
            data=(32, [flags, _u32(x), _u32(y), _u32(width), _u32(height)]),
        )
        mask = X.SubstructureNotifyMask | X.SubstructureRedirectMask
        root.send_event(ev, event_mask=mask)
        self._disp.flush()

    def _send_configure(
        self, x: int, y: int, width: "int | None", height: "int | None"
    ) -> None:
        """ConfigureRequest via `XConfigureWindow` on the client window.

        The window manager selects SubstructureRedirect on the frame, so
        this arrives as a ConfigureRequest it is free to reinterpret —
        apply a placement policy, snap to a work area, or ignore it.
        That is exactly why it is not the first mechanism tried. It is
        here because some window managers act on this and not on the
        client message above, and there is no way to know which from
        inside the process without measuring.

        `width`/`height` of None sends a move with no size change. SDL
        pins PMinSize == PMaxSize on a non-resizable window, so a
        ConfigureRequest carrying a size can be rejected outright by a
        WM enforcing those hints, taking the position with it. The
        move-only form avoids that, which is why it is ordered ahead of
        the full geometry form.
        """
        attrs: dict[str, int] = {"x": int(x), "y": int(y)}
        if width is not None and height is not None:
            attrs["width"] = int(width)
            attrs["height"] = int(height)
        self._window.configure(**attrs)
        self._disp.flush()

    def supports_position(self) -> bool:
        return self._active

    def actual_position(self) -> Optional[tuple[int, int]]:
        """The client window's top-left in root coordinates.

        Walks the window up its parent chain summing each level's
        `get_geometry()` x/y, which are relative to that level's parent.
        Stopping at (and not including) the root yields root-relative
        coordinates, correctly accounting for the reparenting the window
        manager does when it wraps the window in a frame.

        This is the *client* window's origin, not the frame's — which is
        the same reference `set_position()` uses, since it sends
        `_NET_MOVERESIZE_WINDOW` with StaticGravity (see there). So the
        two numbers are comparable without a decoration correction, and
        stay comparable if the window ever stops being borderless.

        Cost: one `get_geometry` + one `query_tree` round-trip per level,
        typically two levels under a reparenting WM. Off the hot path —
        DisplayClient only calls this during the placement-settle window
        and on demand.
        """
        if not self._active or self._window is None or self._disp is None:
            return None
        try:
            root_id = self._disp.screen().root.id
            win = self._window
            x = y = 0
            # Bounded so a cycle or an unexpectedly deep tree cannot
            # spin here; real chains are 1-3 levels.
            for _ in range(32):
                geom = win.get_geometry()
                x += int(geom.x)
                y += int(geom.y)
                parent = win.query_tree().parent
                if parent is None or int(parent.id) == int(root_id):
                    return (x, y)
                win = parent
            return None
        except Exception:
            log.debug("x11-ewmh: could not read the window geometry back", exc_info=True)
            return None

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
