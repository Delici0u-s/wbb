"""
_window.py — SDL2 window, renderer, and streaming texture for one
DisplayClient.

Why no second thread / no second main loop
--------------------------------------------
The GTK4 DisplayClient this replaces spent a large fraction of its
code (``_GtkThread``-equivalent setup, ``_gtk_ready``/``_gtk_stopped``
handshakes, ``GLib.idle_add`` for every frame and every input
callback, ``run_coroutine_threadsafe`` for the reverse direction)
managing the fact that GTK *owns* its own GLib main loop and that loop
must run on the thread that created the window. SDL2 has no such
ownership claim — ``SDL_PollEvent``/``SDL_RenderPresent`` are just
function calls, not a competing event loop. So the entire frame
pump + input loop here runs as a single coroutine on the caller's own
asyncio loop, alternating between "drain pending SDL events"
(non-blocking, ``SDL_PollEvent`` returns 0 immediately once the queue
is empty) and "wait for the next frame from the FrameBuffer" — no
thread handoff, no cross-thread Event/Future plumbing anywhere in this
file. This is the main structural win promised in the design
discussion, not just a smaller diff.

Pixel format
------------
``SDL_PIXELFORMAT_ABGR8888`` is SDL2's name for what numpy gives you
when you read an H×W×4 uint8 array as one little-endian uint32 per
pixel: byte order R,G,B,A in memory becomes, read as a single
little-endian uint32, 0xAABBGGRR — which is exactly the channel
ordering SDL calls ABGR8888. No channel-swizzling step is needed
between wbb's RGBA numpy convention and this texture format, unlike
the old code's BGR->RGBA swap in browser.py's turbojpeg path (that one
is for a different reason — turbojpeg decodes to BGR — and is
unrelated to this).

Multi-monitor coordinates — read this before debugging a position bug
------------------------------------------------------------------------
Both placement backends (KWin's ``frameGeometry``/``geometry``, X11's
``XMoveWindow``/EWMH) operate on a **single global virtual-desktop
coordinate space** that spans every monitor — this is how X11/Xinerama
and KWin's own internal model both work; there is no per-monitor-local
origin at that layer. A monitor placed to the left of and slightly
above your primary monitor sits at, e.g., x<0 and a *negative* y
offset (its top-left corner is above the primary's top-left corner) —
not at (0, 0) the way you'd expect if each monitor had its own
separate coordinate origin.

``list_displays()``/``display_bounds()`` below expose
``SDL_GetDisplayBounds``, which reports each display's bounding
rectangle already converted into that same global space SDL itself
uses — so a position computed as ``(primary_bounds.x + local_x,
primary_bounds.y + local_y)`` lands correctly regardless of which
monitor is "first" in enumeration order or how the monitors are
physically arranged relative to each other. ``DisplayClient`` exposes
this as ``resolve_position(monitor_index, local_x, local_y)`` — see
client.py — rather than asking callers to call SDL functions directly.
"""

from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sdl2
import sdl2.syswm as syswm

from .placement import NativeHandle

log = logging.getLogger(__name__)

_SDL_INITIALIZED = False


def _ensure_sdl_init() -> None:
    global _SDL_INITIALIZED
    if _SDL_INITIALIZED:
        return
    if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
        raise RuntimeError(f"SDL_Init failed: {sdl2.SDL_GetError().decode(errors='replace')}")
    _SDL_INITIALIZED = True


@dataclass
class MouseState:
    """
    Tiny local replacement for GTK4's per-gesture button tracking. SDL2
    reports mouse buttons as a bitmask on motion events but as a
    discrete button number on press/release — this just normalizes
    "which button" into the same int convention the old GTK4
    DisplayClient used (1=left, 2=middle, 3=right), matching
    Gdk.BUTTON_PRIMARY/MIDDLE/SECONDARY numbering so existing
    on_mouse_event callbacks (like main.py's mC()) don't need to
    change.
    """

    last_x: float = 0.0
    last_y: float = 0.0


_SDL_BUTTON_MAP = {
    sdl2.SDL_BUTTON_LEFT: 1,
    sdl2.SDL_BUTTON_MIDDLE: 2,
    sdl2.SDL_BUTTON_RIGHT: 3,
}


@dataclass(frozen=True)
class DisplayBounds:
    """One monitor's bounding rectangle in SDL's global virtual-desktop
    coordinate space — see this module's "Multi-monitor coordinates"
    docstring section for why that space, not a per-monitor-local
    origin, is what every placement backend actually expects."""

    index: int
    x: int
    y: int
    width: int
    height: int


def list_displays() -> list[DisplayBounds]:
    """
    Enumerate every monitor's global bounding rectangle via
    ``SDL_GetNumVideoDisplays``/``SDL_GetDisplayBounds``.

    Safe to call before any window exists — display enumeration in
    SDL2 does not require a window, only ``SDL_Init(SDL_INIT_VIDEO)``
    to have run (``_ensure_sdl_init()`` below).

    Enumeration order matches whatever the OS/compositor reports
    (commonly, but not guaranteed to be, left-to-right physical
    order) — same caveat the old GTK4 client's ``monitor=`` parameter
    docstring already called out for ``Gdk.Display.get_monitors()``.
    Match by inspecting ``.x``/``.y`` against what you know about your
    own physical layout, not by assuming index 0 is "the left one".
    """
    _ensure_sdl_init()
    out: list[DisplayBounds] = []
    n = sdl2.SDL_GetNumVideoDisplays()
    rect = sdl2.SDL_Rect()
    for i in range(n):
        if sdl2.SDL_GetDisplayBounds(i, ctypes.byref(rect)) == 0:
            out.append(DisplayBounds(index=i, x=rect.x, y=rect.y, width=rect.w, height=rect.h))
        else:
            log.warning(
                "SDL_GetDisplayBounds failed for display %d: %s",
                i,
                sdl2.SDL_GetError().decode(errors="replace"),
            )
    return out


class SDLWindow:
    """
    Owns one SDL2 window + accelerated renderer + one streaming
    texture that gets re-uploaded every frame.

    Must be constructed and used entirely from the thread that will
    also call ``poll_events``/``push_frame`` — in this codebase that's
    always the single asyncio task DisplayClient.run_async() drives
    (see client.py). SDL2 itself has no thread-affinity requirement as
    strict as GTK4's, but mixing SDL calls across threads without your
    own locking is still unsupported by SDL2's own docs, so this class
    doesn't attempt it.
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        title: str,
        wm_class: str,
        borderless: bool,
        resizable: bool = False,
    ) -> None:
        _ensure_sdl_init()

        # SDL_HINT_APP_NAME / the X11 WM_CLASS hint must be set BEFORE
        # SDL_CreateWindow — this is what the KWin/X11 placement
        # backends match against (see placement/kwin.py,
        # placement/x11_ewmh.py), so it has to be stable and set early,
        # not relabeled after the fact.
        sdl2.SDL_SetHint(b"SDL_VIDEO_X11_WMCLASS", wm_class.encode())

        flags = sdl2.SDL_WINDOW_SHOWN
        if borderless:
            flags |= sdl2.SDL_WINDOW_BORDERLESS
        if resizable:
            flags |= sdl2.SDL_WINDOW_RESIZABLE

        self._wm_class = wm_class
        self.window = sdl2.SDL_CreateWindow(
            title.encode(),
            sdl2.SDL_WINDOWPOS_UNDEFINED,
            sdl2.SDL_WINDOWPOS_UNDEFINED,
            width,
            height,
            flags,
        )
        if not self.window:
            raise RuntimeError(f"SDL_CreateWindow failed: {sdl2.SDL_GetError().decode(errors='replace')}")

        self.renderer = sdl2.SDL_CreateRenderer(
            self.window, -1, sdl2.SDL_RENDERER_ACCELERATED | sdl2.SDL_RENDERER_PRESENTVSYNC
        )
        if not self.renderer:
            # Accelerated+vsync isn't available everywhere (e.g. some
            # software/VM GL drivers) — fall back to whatever SDL can
            # give us rather than hard-failing the whole window.
            log.warning(
                "Accelerated+vsync renderer unavailable (%s); falling back "
                "to SDL's default renderer flags.",
                sdl2.SDL_GetError().decode(errors="replace"),
            )
            self.renderer = sdl2.SDL_CreateRenderer(self.window, -1, 0)
        if not self.renderer:
            raise RuntimeError(f"SDL_CreateRenderer failed: {sdl2.SDL_GetError().decode(errors='replace')}")

        self._texture: Optional[ctypes.c_void_p] = None
        self._tex_size: tuple[int, int] = (0, 0)
        self._mouse = MouseState()

    # ------------------------------------------------------------------
    # Native handle (for placement backends)
    # ------------------------------------------------------------------
    def native_handle(self) -> NativeHandle:
        info = syswm.SDL_SysWMinfo()
        sdl2.SDL_VERSION(info.version)
        if sdl2.SDL_GetWindowWMInfo(self.window, ctypes.byref(info)) != sdl2.SDL_TRUE:
            return NativeHandle(subsystem="unknown", window_title="", wm_class=self._wm_class)

        title = (sdl2.SDL_GetWindowTitle(self.window) or b"").decode(errors="replace")

        if info.subsystem == sdl2.SDL_SYSWM_WAYLAND:
            return NativeHandle(
                subsystem="wayland",
                window_title=title,
                wm_class=self._wm_class,
                wayland_display=int(info.info.wl.display or 0) or None,
                wayland_surface=int(info.info.wl.surface or 0) or None,
            )
        elif info.subsystem == sdl2.SDL_SYSWM_X11:
            return NativeHandle(
                subsystem="x11",
                window_title=title,
                wm_class=self._wm_class,
                x11_display=int(info.info.x11.display or 0) or None,
                x11_window=int(info.info.x11.window or 0) or None,
            )
        return NativeHandle(subsystem="unknown", window_title=title, wm_class=self._wm_class)

    def current_display_index(self) -> int:
        """Which monitor (index into list_displays()) this window is
        currently on, per SDL_GetWindowDisplayIndex. Returns 0 (assume
        primary) if SDL can't determine it — same defensive default
        list_displays()'s caller (client.py) already falls back to."""
        idx = sdl2.SDL_GetWindowDisplayIndex(self.window)
        return idx if idx >= 0 else 0

    # ------------------------------------------------------------------
    # Frame upload
    # ------------------------------------------------------------------
    def push_frame(self, arr: np.ndarray) -> None:
        """
        Upload one H×W×4 uint8 RGBA frame and present it. Re-creates
        the streaming texture if the frame's dimensions changed since
        the last call (mirrors the old GTK4 client's
        ``_FrameTexturePaintable.set_texture`` resize handling — same
        "filters can change output size between frames" contract from
        ``DisplayClient``'s docstring).
        """
        arr = np.ascontiguousarray(arr, dtype=np.uint8)
        h, w = arr.shape[0], arr.shape[1]

        if (w, h) != self._tex_size:
            if self._texture:
                sdl2.SDL_DestroyTexture(self._texture)
            self._texture = sdl2.SDL_CreateTexture(
                self.renderer,
                sdl2.SDL_PIXELFORMAT_ABGR8888,
                sdl2.SDL_TEXTUREACCESS_STREAMING,
                w,
                h,
            )
            if not self._texture:
                raise RuntimeError(
                    f"SDL_CreateTexture failed: {sdl2.SDL_GetError().decode(errors='replace')}"
                )
            self._tex_size = (w, h)
            sdl2.SDL_SetWindowSize(self.window, w, h)

        pitch = w * 4
        ptr = arr.ctypes.data_as(ctypes.c_void_p)
        sdl2.SDL_UpdateTexture(self._texture, None, ptr, pitch)
        sdl2.SDL_RenderClear(self.renderer)
        sdl2.SDL_RenderCopy(self.renderer, self._texture, None, None)
        sdl2.SDL_RenderPresent(self.renderer)

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def poll_events(self) -> list[dict]:
        """
        Drain all currently queued SDL events into a list of plain
        dicts. Non-blocking — returns immediately once the queue is
        empty, same contract as ``SDL_PollEvent``'s own return value,
        so the caller's asyncio loop (client.py) controls the actual
        poll cadence rather than this function blocking on it.
        """
        out: list[dict] = []
        event = sdl2.SDL_Event()
        while sdl2.SDL_PollEvent(ctypes.byref(event)) != 0:
            decoded = self._decode_event(event)
            if decoded is not None:
                out.append(decoded)
        return out

    def _decode_event(self, event: "sdl2.SDL_Event") -> Optional[dict]:
        t = event.type
        if t == sdl2.SDL_QUIT:
            return {"kind": "quit"}

        if t == sdl2.SDL_MOUSEBUTTONDOWN or t == sdl2.SDL_MOUSEBUTTONUP:
            b = event.button
            self._mouse.last_x, self._mouse.last_y = float(b.x), float(b.y)
            return {
                "kind": "mouse",
                "event_type": "down" if t == sdl2.SDL_MOUSEBUTTONDOWN else "up",
                "x": float(b.x),
                "y": float(b.y),
                "button": _SDL_BUTTON_MAP.get(b.button, b.button),
            }

        if t == sdl2.SDL_MOUSEMOTION:
            m = event.motion
            self._mouse.last_x, self._mouse.last_y = float(m.x), float(m.y)
            return {"kind": "mouse", "event_type": "move", "x": float(m.x), "y": float(m.y), "button": 0}

        if t == sdl2.SDL_MOUSEWHEEL:
            w = event.wheel
            # SDL2's wheel.y is "lines/clicks", positive = away from
            # the user (scroll up) — matches the sign convention the
            # old GTK4 EventControllerScroll callback used, so
            # main.py's existing sC() callback needs no changes.
            return {"kind": "scroll", "dx": float(w.x), "dy": float(w.y)}

        if t == sdl2.SDL_KEYDOWN or t == sdl2.SDL_KEYUP:
            k = event.key
            name = sdl2.SDL_GetKeyName(k.keysym.sym).decode(errors="replace")
            return {
                "kind": "key",
                "event_type": "down" if t == sdl2.SDL_KEYDOWN else "up",
                "key": name,
            }

        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._texture:
            sdl2.SDL_DestroyTexture(self._texture)
            self._texture = None
        if self.renderer:
            sdl2.SDL_DestroyRenderer(self.renderer)
            self.renderer = None  # type: ignore[assignment]
        if self.window:
            sdl2.SDL_DestroyWindow(self.window)
            self.window = None  # type: ignore[assignment]
