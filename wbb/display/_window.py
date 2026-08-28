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

from .. import _compat
from . import alpha as _alpha
from .placement import NativeHandle
from .window_type import WindowType, apply_gamescope_overlay, apply_window_type, needs_pre_map

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


#: Windows created by this process and not yet closed. Only used to
#: decide whether SDL's video subsystem can safely be bounced to pick up
#: a forced ARGB visual (see alpha.prepare).
_LIVE_WINDOWS: set[int] = set()


def _live_windows() -> int:
    return len(_LIVE_WINDOWS)


def preinit_alpha(display_name: Optional[str] = None) -> bool:
    """Force an ARGB visual for windows created later in this process.

    Call this before anything else touches SDL if you intend to create an
    `alpha=True` window *after* something that initialises SDL video —
    `list_displays()` and `centered_position()` both do. Returns True if
    a 32-bit visual was found and the hint was set.

    Not needed when the alpha window is the first SDL window in the
    process; `SDLWindow` handles that case itself.
    """
    prep = _alpha.prepare(True, display_name=display_name, live_windows=_live_windows())
    return prep.x11_visual_id is not None


def centered_position(
    width: int,
    height: int,
    monitor: int = 0,
) -> tuple[int, int]:
    """Monitor-relative top-left that centres a `width` x `height` window.

    Positions in wbb are relative to a monitor, so the return value is
    meant to be passed straight to `DisplayClient(position=...,
    monitor=...)`. Hard-coding an absolute position is what breaks on a
    multi-monitor setup with mixed resolutions: (40, 40) is fine on the
    monitor you developed on and off-screen on the next one.

    Falls back to (0, 0) if the monitor cannot be measured.
    """
    try:
        displays = list_displays()
    except Exception:
        return (0, 0)
    if not displays:
        return (0, 0)
    d = displays[monitor] if 0 <= monitor < len(displays) else displays[0]
    return (max(0, (d.width - width) // 2), max(0, (d.height - height) // 2))


def video_driver() -> str:
    """Which video driver SDL2 selected — "x11", "wayland", "windows"...

    Worth checking before debugging anything placement-related: on a
    Plasma Wayland session SDL2 may pick either `x11` (an XWayland
    window, where EWMH and the SHAPE extension are available) or
    `wayland` (a native surface, where neither is and the compositor
    owns positioning entirely). The two behave nothing alike and there
    is no way to tell them apart from the session type alone.

    Initialises SDL video if it is not up yet. Returns "" if SDL cannot
    report one.
    """
    _ensure_sdl_init()
    name = sdl2.SDL_GetCurrentVideoDriver()
    return name.decode(errors="replace") if name else ""


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
        window_type: WindowType = WindowType.NORMAL,
        alpha: bool = False,
        gamescope_overlay: bool = False,
        x11_display_name: Optional[str] = None,
        position: Optional[tuple[int, int]] = None,
    ) -> None:
        # MUST run before _ensure_sdl_init(). SDL2's X11 backend resolves
        # the screen's visual once, in X11_VideoInit — i.e. inside
        # SDL_Init(SDL_INIT_VIDEO) — not per window. A
        # SDL_VIDEO_X11_VISUALID set after SDL_Init is simply never read,
        # which is why alpha=True previously reported "found visual 0x95
        # (depth 32)" and then produced a 24-bit window anyway.
        self._alpha_prep = _alpha.prepare(
            alpha, display_name=x11_display_name, live_windows=_live_windows()
        )
        _ensure_sdl_init()

        # SDL_HINT_APP_NAME / the X11 WM_CLASS hint must be set BEFORE
        # SDL_CreateWindow — this is what the KWin/X11 placement
        # backends match against (see placement/kwin.py,
        # placement/x11_ewmh.py), so it has to be stable and set early,
        # not relabeled after the fact.
        #
        # X11 and Wayland use DIFFERENT SDL hints for the class/app-id,
        # and which one matters depends on the subsystem SDL2 selects at
        # runtime (x11 vs native wayland). Set BOTH so the window's
        # resourceClass / app-id is `wm_class` either way — otherwise, on
        # a native Wayland session, KWin's `resourceClass == wmClass`
        # match in the scripting backend finds nothing and positioning
        # silently no-ops even with pydbus installed.
        wmc = wm_class.encode()
        sdl2.SDL_SetHint(b"SDL_VIDEO_X11_WMCLASS", wmc)
        # SDL_VIDEO_WAYLAND_WMCLASS exists on newer SDL2; SDL_APP_ID is the
        # broader name. Set whichever the installed SDL exposes (both are
        # harmless no-ops if unrecognised by this SDL build).
        sdl2.SDL_SetHint(b"SDL_VIDEO_WAYLAND_WMCLASS", wmc)
        sdl2.SDL_SetHint(b"SDL_APP_ID", wmc)

        # Window type is read by the WM at map time, so a window that
        # needs a non-default type is created HIDDEN, gets the property,
        # and is only then shown — one real map, with the property
        # already in place. See window_type.py.
        self._pre_map = needs_pre_map(window_type) or gamescope_overlay
        flags = sdl2.SDL_WINDOW_HIDDEN if self._pre_map else sdl2.SDL_WINDOW_SHOWN
        if borderless:
            flags |= sdl2.SDL_WINDOW_BORDERLESS
        if resizable:
            flags |= sdl2.SDL_WINDOW_RESIZABLE

        self._wm_class = wm_class
        self._window_type = window_type
        self._x11_display_name = x11_display_name
        # Stored, not passed through, because _create_window_and_renderer
        # runs a second time on the alpha software-renderer retry below
        # and must recreate the window at the same place.
        self._position = position
        self._create_window_and_renderer(title, width, height, flags, alpha)

        handle = self.native_handle()
        self.alpha_active: bool = False
        if alpha:
            verified = _alpha.verify(handle, display_name=x11_display_name)
            # None => not determinable (native Wayland, or no python-xlib).
            # On Wayland the surface is ARGB anyway, so None is optimistic
            # rather than pessimistic; False is a hard "the window is 24-bit".
            self.alpha_active = verified is not False

            if not self.alpha_active:
                # The accelerated renderer forces SDL_WINDOW_OPENGL, and
                # X11_GL_GetVisual then picks its own GLX visual, which
                # can override the one we forced. The software renderer
                # composites through the window's own visual instead, so
                # retry once with it before giving up. One extra window
                # creation at startup; nothing per frame.
                log.info(
                    "alpha: window is not 32-bit with the accelerated "
                    "renderer; retrying with the software renderer"
                )
                sdl2.SDL_DestroyRenderer(self.renderer)
                sdl2.SDL_DestroyWindow(self.window)
                self._create_window_and_renderer(
                    title, width, height, flags, alpha, force_software=True
                )
                handle = self.native_handle()
                self.alpha_active = (
                    _alpha.verify(handle, display_name=x11_display_name) is not False
                )

            if not self.alpha_active:
                log.warning(
                    "alpha=True requested but the window is not 32-bit (%s), "
                    "with either renderer. Frames will render opaque. Check "
                    "that a compositor is running and that python-xlib can "
                    "reach the display.",
                    self._alpha_prep.describe(),
                )
            elif self._software_renderer:
                log.info("alpha: active (software renderer)")

        self.window_type_active: bool = False
        if self._pre_map:
            self.window_type_active = apply_window_type(
                handle, window_type, display_name=x11_display_name
            )
            if gamescope_overlay:
                apply_gamescope_overlay(handle, True, display_name=x11_display_name)
            sdl2.SDL_ShowWindow(self.window)

        self._texture: Optional[ctypes.c_void_p] = None
        self._tex_size: tuple[int, int] = (0, 0)
        self._tex_capacity: tuple[int, int] = (0, 0)
        #: Sub-rectangle of the texture actually in use, or None when the
        #: frame fills it exactly.
        self._src_rect: Optional["sdl2.SDL_Rect"] = None
        self._mouse = MouseState()
        _LIVE_WINDOWS.add(id(self))

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
    def _create_window_and_renderer(
        self,
        title: str,
        width: int,
        height: int,
        flags: int,
        alpha: bool,
        *,
        force_software: bool = False,
    ) -> None:
        """Create the SDL window and its renderer. Idempotent per call."""
        self._software_renderer = force_software
        if force_software:
            sdl2.SDL_SetHint(b"SDL_RENDER_DRIVER", b"software")

        # A real position here, rather than SDL_WINDOWPOS_UNDEFINED, is
        # what stops the window manager applying its own placement
        # policy at map time. SDL2's X11 backend turns a concrete x/y
        # into a USPosition entry in WM_NORMAL_HINTS, the standard "the
        # user asked for exactly this spot" signal, and WMs skip
        # placement for windows carrying it. Without this the window is
        # born wherever the WM decides (KWin's placement policy is
        # configurable and "Centered" is one of the choices) and the
        # requested position only ever arrives afterwards, as a move
        # the WM is free to refuse.
        #
        # NOT independently verified that KWin honours USPosition for
        # every window type — notably notification/OSD types, which KWin
        # groups as "special windows". DisplayClient's placement-settle
        # readback reports what actually happened; see
        # client.py's _settle_placement().
        pos_x, pos_y = (
            self._position if self._position is not None
            else (sdl2.SDL_WINDOWPOS_UNDEFINED, sdl2.SDL_WINDOWPOS_UNDEFINED)
        )
        self.window = sdl2.SDL_CreateWindow(
            title.encode(),
            pos_x,
            pos_y,
            width,
            height,
            flags,
        )
        if not self.window:
            raise RuntimeError(
                f"SDL_CreateWindow failed: {sdl2.SDL_GetError().decode(errors='replace')}"
            )

        renderer_flags = (
            sdl2.SDL_RENDERER_SOFTWARE
            if force_software
            else sdl2.SDL_RENDERER_ACCELERATED | sdl2.SDL_RENDERER_PRESENTVSYNC
        )
        self.renderer = sdl2.SDL_CreateRenderer(self.window, -1, renderer_flags)
        if not self.renderer:
            # Accelerated+vsync isn't available everywhere (e.g. some
            # software/VM GL drivers) — fall back to whatever SDL can
            # give us rather than hard-failing the whole window.
            log.warning(
                "Renderer unavailable (%s); falling back to SDL's default "
                "renderer flags.",
                sdl2.SDL_GetError().decode(errors="replace"),
            )
            self.renderer = sdl2.SDL_CreateRenderer(self.window, -1, 0)
        if not self.renderer:
            raise RuntimeError(
                f"SDL_CreateRenderer failed: {sdl2.SDL_GetError().decode(errors='replace')}"
            )

        # Clear colour + its blend mode. With alpha on this writes alpha 0
        # instead of blending; with alpha off it clears to opaque black
        # rather than SDL's default draw colour.
        _alpha.configure_renderer(self.renderer, alpha)

    def push_frame(self, arr: np.ndarray) -> bool:
        """
        Upload one H×W×4 uint8 RGBA frame and present it. Re-creates
        the streaming texture if the frame's dimensions changed since
        the last call (mirrors the old GTK4 client's
        ``_FrameTexturePaintable.set_texture`` resize handling — same
        "filters can change output size between frames" contract from
        ``DisplayClient``'s docstring).

        Upload path
        -----------
        Uses ``SDL_LockTexture`` to obtain a pointer into the texture's
        own staging memory and writes the frame rows directly into it,
        rather than ``SDL_UpdateTexture`` (which copies from a source
        buffer the driver doesn't own). This removes one CPU-side copy
        per frame on the streaming-texture path LockTexture exists for.

        Contiguity
        ----------
        Only calls ``ascontiguousarray`` when the incoming array is
        *not* already C-contiguous. The common cases — a frame straight
        out of shared memory, or the output of a LUT color filter — are
        already contiguous, so the guard is a no-op flag check rather
        than a silent 3.6 MB re-materialisation every frame. Zero-copy
        filters that return a non-contiguous view (``crop``/``flip``)
        still get the one copy they genuinely need.
        """
        if arr.dtype != np.uint8 or not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr, dtype=np.uint8)
        h, w = arr.shape[0], arr.shape[1]

        # The texture is grow-only, and only the used sub-rectangle is
        # uploaded and blitted. A window that resizes every frame used to
        # destroy and recreate its texture every frame — a GPU allocation
        # plus a driver round-trip per frame, which is what makes a
        # continuously resizing window stutter. Now the texture is
        # reallocated only when the frame grows past the largest size
        # seen so far, which converges after a second or two.
        cap_w, cap_h = self._tex_capacity
        if self._texture is None or w > cap_w or h > cap_h:
            if self._texture:
                sdl2.SDL_DestroyTexture(self._texture)
            cap_w, cap_h = max(w, cap_w), max(h, cap_h)
            self._texture = sdl2.SDL_CreateTexture(
                self.renderer,
                sdl2.SDL_PIXELFORMAT_ABGR8888,
                sdl2.SDL_TEXTUREACCESS_STREAMING,
                cap_w,
                cap_h,
            )
            if not self._texture:
                raise RuntimeError(
                    f"SDL_CreateTexture failed: {sdl2.SDL_GetError().decode(errors='replace')}"
                )
            _alpha.configure_texture(self._texture)
            self._tex_capacity = (cap_w, cap_h)

        resized = (w, h) != self._tex_size
        if resized:
            self._tex_size = (w, h)
            sdl2.SDL_SetWindowSize(self.window, w, h)

        src_pitch = w * 4

        if (w, h) != self._tex_capacity:
            # Partial upload into a larger texture. One SDL call rather
            # than the lock/rowloop below, which would have to honour the
            # texture's pitch across h rows from Python.
            rect = sdl2.SDL_Rect(0, 0, w, h)
            sdl2.SDL_UpdateTexture(
                self._texture, rect, arr.ctypes.data_as(ctypes.c_void_p), src_pitch
            )
            src = sdl2.SDL_Rect(0, 0, w, h)
            sdl2.SDL_RenderClear(self.renderer)
            sdl2.SDL_RenderCopy(self.renderer, self._texture, src, None)
            sdl2.SDL_RenderPresent(self.renderer)
            self._src_rect = src
            return resized

        self._src_rect = None

        pixels_ptr = ctypes.c_void_p()
        pitch = ctypes.c_int()
        locked = sdl2.SDL_LockTexture(
            self._texture, None, ctypes.byref(pixels_ptr), ctypes.byref(pitch)
        )
        if locked != 0:
            # Lock failed (driver quirk) — fall back to UpdateTexture
            # rather than dropping the frame.
            ptr = arr.ctypes.data_as(ctypes.c_void_p)
            sdl2.SDL_UpdateTexture(self._texture, None, ptr, src_pitch)
        else:
            dst_pitch = pitch.value
            src_addr = arr.ctypes.data
            if dst_pitch == src_pitch:
                # Tightly packed: one contiguous copy of the whole frame.
                ctypes.memmove(pixels_ptr, ctypes.c_void_p(src_addr), src_pitch * h)
            else:
                # Padded destination rows: copy row by row honouring the
                # texture's own pitch.
                dst_base = pixels_ptr.value
                for y in range(h):
                    ctypes.memmove(
                        ctypes.c_void_p(dst_base + y * dst_pitch),
                        ctypes.c_void_p(src_addr + y * src_pitch),
                        src_pitch,
                    )
            sdl2.SDL_UnlockTexture(self._texture)

        sdl2.SDL_RenderClear(self.renderer)
        sdl2.SDL_RenderCopy(self.renderer, self._texture, None, None)
        sdl2.SDL_RenderPresent(self.renderer)
        return resized

    def present_again(self) -> None:
        """Re-present the texture already uploaded, with no new upload.

        Costs one clear + one blit + one present; no allocation, no
        pixel copy. This is what an SDL_WINDOWEVENT_EXPOSED, a resize, or
        DisplayClient.request_repaint() needs — the frame has not
        changed, only the window's idea of what is on screen has.
        No-op before the first push_frame().
        """
        if not self._texture:
            return
        sdl2.SDL_RenderClear(self.renderer)
        sdl2.SDL_RenderCopy(self.renderer, self._texture, self._src_rect, None)
        sdl2.SDL_RenderPresent(self.renderer)

    def size(self) -> tuple[int, int]:
        """The window's actual current size per SDL, not the texture's.

        These disagree whenever the window manager refused a
        SDL_SetWindowSize — which is silent, and which SDL_RenderCopy
        then papers over by stretching the texture to fit.
        """
        w = ctypes.c_int()
        h = ctypes.c_int()
        sdl2.SDL_GetWindowSize(self.window, ctypes.byref(w), ctypes.byref(h))
        return (w.value, h.value)

    def window_id(self) -> int:
        return int(sdl2.SDL_GetWindowID(self.window))

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

        if t == sdl2.SDL_WINDOWEVENT:
            if _compat.LEGACY_EXPOSE:
                return None
            we = event.window
            if we.event == sdl2.SDL_WINDOWEVENT_CLOSE:
                return {"kind": "window", "event_type": "close", "window_id": int(we.windowID)}
            if we.event in (
                sdl2.SDL_WINDOWEVENT_EXPOSED,
                sdl2.SDL_WINDOWEVENT_SHOWN,
                sdl2.SDL_WINDOWEVENT_RESTORED,
            ):
                return {"kind": "window", "event_type": "expose", "window_id": int(we.windowID)}
            if we.event in (
                sdl2.SDL_WINDOWEVENT_SIZE_CHANGED,
                sdl2.SDL_WINDOWEVENT_RESIZED,
            ):
                return {
                    "kind": "window",
                    "event_type": "resize",
                    "window_id": int(we.windowID),
                    "width": int(we.data1),
                    "height": int(we.data2),
                }
            return None

        if t == sdl2.SDL_MOUSEBUTTONDOWN or t == sdl2.SDL_MOUSEBUTTONUP:
            b = event.button
            self._mouse.last_x, self._mouse.last_y = float(b.x), float(b.y)
            return {
                "kind": "mouse",
                "event_type": "down" if t == sdl2.SDL_MOUSEBUTTONDOWN else "up",
                "x": float(b.x),
                "y": float(b.y),
                "button": _SDL_BUTTON_MAP.get(b.button, b.button),
                "window_id": int(b.windowID),
            }

        if t == sdl2.SDL_MOUSEMOTION:
            m = event.motion
            self._mouse.last_x, self._mouse.last_y = float(m.x), float(m.y)
            return {
                "kind": "mouse",
                "event_type": "move",
                "x": float(m.x),
                "y": float(m.y),
                "button": 0,
                "window_id": int(m.windowID),
            }

        if t == sdl2.SDL_MOUSEWHEEL:
            w = event.wheel
            # SDL2's wheel.y is "lines/clicks", positive = away from
            # the user (scroll up) — matches the sign convention the
            # old GTK4 EventControllerScroll callback used, so
            # main.py's existing sC() callback needs no changes.
            return {
                "kind": "scroll",
                "dx": float(w.x),
                "dy": float(w.y),
                "window_id": int(w.windowID),
            }

        if t == sdl2.SDL_KEYDOWN or t == sdl2.SDL_KEYUP:
            k = event.key
            name = sdl2.SDL_GetKeyName(k.keysym.sym).decode(errors="replace")
            return {
                "kind": "key",
                "event_type": "down" if t == sdl2.SDL_KEYDOWN else "up",
                "key": name,
                "window_id": int(k.windowID),
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
