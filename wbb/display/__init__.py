"""
wbb_display — SDL2-backed replacement for wbb's GTK4 DisplayClient.

Public surface is intentionally identical in *shape* to the old GTK4
DisplayClient (wbb/display.py) so call sites like main.py's
`DisplayClient(buffer, title=..., filters=..., position=..., ...)` do
not need to change. What changed underneath:

* Windowing/rendering: SDL2 (via PySDL2) instead of GTK4. No second
  main loop to bridge against asyncio — see _window.py's module
  docstring for why that deletes most of the old file's thread-safety
  machinery outright rather than just hiding it better.
* Placement (always-on-top / absolute position / borderless): a
  capability-detected backend chain (see placement/_base.py) instead
  of a single hardcoded wlr-layer-shell path. KWin's D-Bus scripting
  backend is tried first because it is the only mechanism that works
  on KDE Plasma regardless of X11 vs. Wayland session — see
  placement/kwin.py's module docstring for the full rationale and its
  real limitations (this is privileged, KWin-version-coupled surface
  area, not a stable public protocol like EWMH or wlr-layer-shell).
"""

from __future__ import annotations

from .client import DisplayClient, WindowPosition
from ._window import DisplayBounds, list_displays

__all__ = ["DisplayClient", "WindowPosition", "DisplayBounds", "list_displays"]
