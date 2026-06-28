"""
Placement backend protocol.

Three independent capabilities — always-on-top, absolute position,
borderless — have no single portable mechanism across X11/Wayland/KWin
(see the design discussion this module implements: GTK4's own removal
of set_keep_above, Wayland's deliberate omission of a positioning
protocol, KWin's Wayland D-Bus gap). So instead of one mechanism with
a binary "did it work" branch, this is a *chain* of backends, each one
either fully activating or cleanly declining, tried in priority order
by `select_backend()` in `chain.py`.

Borderless is NOT part of this protocol on purpose: SDL2's
SDL_WINDOW_BORDERLESS flag (set at window-creation time, in
_window.py) works on every backend SDL2 itself supports, with zero
window-manager privilege needed — there is nothing to chain-select for
it, unlike the old GTK4 DisplayClient which routed it through
layer-shell purely because it was bundled with the other two flags.
Decoupling it removes 2/3 of the original portability problem for
free; see the design conversation for why.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Protocol


@dataclasses.dataclass(frozen=True, slots=True)
class NativeHandle:
    """
    Whatever SDL2 reports about the real window via
    ``SDL_GetWindowWMInfo`` — see ``_window.py``'s ``native_handle()``.

    Exactly one of the wayland/x11 fields is populated, matching
    ``subsystem``. ``window_title``/``wm_class`` are passed through
    separately because the KWin backend matches by those, not by a
    raw surface pointer (see ``placement/kwin.py`` for why — KWin's
    scripting API has no concept of a wl_surface pointer from another
    process; it works in terms of "the window whose resourceClass is
    X", the same names ``Window Rules`` in System Settings use).
    """

    subsystem: str  # "wayland" | "x11" | "unknown"
    window_title: str
    wm_class: str
    wayland_display: Optional[int] = None
    wayland_surface: Optional[int] = None
    x11_display: Optional[int] = None
    x11_window: Optional[int] = None


class PlacementBackend(Protocol):
    """
    One way of asking the desktop environment for always-on-top /
    absolute position. Implementations must never raise out of
    ``activate()`` — a backend that isn't applicable (wrong session
    type, missing library, compositor refuses the request) returns
    False and the chain moves on; see ``chain.py``.
    """

    #: Short identifier used in log messages, e.g. "kwin", "x11-ewmh".
    name: str

    def activate(self, handle: NativeHandle) -> bool:
        """
        Attempt to take control of placement for this window. Returns
        True if this backend is live and ``set_above``/``set_position``
        will have real effect; False if this backend doesn't apply
        here (caller should try the next one in the chain).

        Must be cheap to call speculatively — chain.py calls this on
        every registered backend in order until one returns True, so
        an expensive failed probe here is paid on every single
        DisplayClient that ends up on a platform where this backend
        doesn't apply.
        """
        ...

    def set_above(self, above: bool) -> None:
        """No-op if activate() returned False or was never called."""
        ...

    def set_position(self, x: int, y: int, width: int, height: int) -> None:
        """
        Move the window so its top-left corner is at (x, y), local to
        the monitor the window currently lives on. width/height are
        the window's current pixel size — some backends (none of the
        current ones, but kept in the signature for whichever comes
        next) need it to compute complementary margins the way the
        old layer-shell code did.

        No-op if activate() returned False or was never called.
        """
        ...

    def supports_position(self) -> bool:
        """True if set_position() will have a real effect right now."""
        ...

    def set_click_through(self, enabled: bool) -> bool:
        """
        Attempt to make the window pass all mouse/touch input through
        to whatever is beneath it. Returns whether it actually took
        effect — unlike set_above/set_position (which are silent
        no-ops on failure, matching the old GTK4 client's degrade-
        quietly behavior for always_on_top), this one returns a bool
        because click-through has no real fallback: a caller asking
        for it specifically needs to know whether their window will
        actually eat clicks or not, since that changes whether the
        on_mouse_event callback will ever fire for clicks that should
        have passed through.

        Backends without a click-through mechanism (KWin's scripting
        API has none — see placement/kwin.py's module docstring) must
        return False rather than raising.
        """
        return False
