"""
DisplayClient — SDL2-backed FrameBuffer display, single-coroutine.

Public API shape matches the old GTK4 DisplayClient (wbb/display.py)
closely enough that main.py's existing call site does not need to
change beyond the three fixes below.

What's different under the hood, and why
-------------------------------------------
* No GTK4, no GLib main loop, no second thread — see _window.py's
  module docstring for the structural argument.
* Always-on-top / absolute position go through a capability-detected
  backend chain (wbb_display.placement) — see placement/chain.py and
  placement/kwin.py.
* Borderless is unconditional and free (SDL_WINDOW_BORDERLESS at
  window-creation time) — no backend dependency.

Fixes applied after first real-desktop testing
-------------------------------------------------
1. **Multi-monitor positioning.** ``position=`` is now resolved
   against a specific monitor's real global origin (via
   ``_window.list_displays()``/``SDL_GetDisplayBounds``) instead of
   being passed straight through as if every setup has one monitor at
   (0, 0). See ``monitor=`` below and ``_resolve_global_position()``.
   This was the actual bug behind "can't move the window into the
   area below monitor 2 but above the bottom of monitor 1" — both
   placement backends operate in one global virtual-desktop
   coordinate space spanning every monitor (see _window.py's
   "Multi-monitor coordinates" docstring section), and nothing was
   previously translating a monitor-local position into that space
   for any monitor other than one happening to sit at (0, 0).
2. **click_through is back**, real (not a no-op) on X11 sessions via
   the Shape extension (placement/x11_ewmh.py); on KWin it's a clean,
   explained failure (KWin's scripting API genuinely has no
   input-transparency mechanism — see placement/kwin.py — this is a
   real platform gap, not something missing from this code).
3. **CPU/filter cost.** Two changes: (a) filters now run in a thread-
   pool executor via ``loop.run_in_executor`` instead of inline on the
   asyncio loop, so a slow filter chain (scale() through Pillow,
   blur() through scipy) no longer blocks event polling or anything
   else sharing this loop; (b) an explicit ``max_fps`` cap (default
   60) throttles the render loop independently of whatever the
   renderer's vsync situation is — the old loop had *no* cap at all
   when SDL's accelerated+vsync renderer wasn't available (see
   _window.py's fallback-to-no-vsync path), so it would spin as fast
   as next_frame()+filters+push_frame could go, with zero backpressure.
   max_fps=0 disables the cap entirely if you want the old (uncapped)
   behavior back for some reason.
4. **Initial placement vs. XWayland's first-map race.** ``set_above()``
   / ``set_position()`` used to be called exactly once each, immediately
   after ``SDLWindow(...)`` was constructed, with no yield to the event
   loop in between. Under XWayland (which is what the X11/EWMH backend
   actually drives — see placement/x11_ewmh.py — even on a Wayland
   session, since SDL2 chose the x11 subsystem here), a freshly created
   toplevel isn't guaranteed to have completed its first
   map/configure round-trip with the compositor by the time
   ``SDL_CreateWindow`` returns. An ``XConfigureWindow`` (what
   ``X11Placement.set_position()`` sends) issued before that round-trip
   finishes can be silently superseded by KWin's own initial-placement
   logic once it actually maps the window — see
   https://github.com/swaywm/wlroots/issues/292 for the same race
   reported independently against wlroots/XWayland. ``always_on_top``
   appeared unaffected only because ``set_above()`` uses an EWMH
   ``_NET_WM_STATE`` client message rather than a raw configure
   request, and client messages get queued and re-delivered once the
   window manager actually starts managing the window, whereas a stale
   ``XConfigureWindow`` does not get retried.

   Fix: rather than guessing a fixed startup delay (fragile — the
   right delay depends on the machine, the compositor, and whatever
   else happens to run before this), ``set_above()``/``set_position()``
   are now re-sent for the first few iterations of the render loop
   itself (see ``_PLACEMENT_RETRY_FRAMES`` below). The render loop
   already pumps SDL events and yields to the event loop every
   iteration, so this naturally lands the retries after the window has
   had real wall-clock time and several event-loop turns to finish
   mapping, while costing only a handful of extra D-Bus/Xlib
   round-trips, only during the first few frames, only when
   always_on_top/position were actually requested.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections.abc import Callable
from typing import Any, Optional

import numpy as np

from .placement import select_backend
from ._window import SDLWindow, list_displays

log = logging.getLogger(__name__)

Filter = Callable[[np.ndarray], np.ndarray]
MouseCallback = Callable[[str, float, float, int], Any]
KeyCallback = Callable[[str, str], Any]
ScrollCallback = Callable[[float, float], Any]

# How many render-loop iterations to re-apply always_on_top/position
# for after startup. See module docstring point 4 — this is what
# absorbs the race against XWayland's first map/configure round-trip
# without guessing a fixed sleep duration. Five frames is on the order
# of ~100ms at a typical first-few-frames pace and has been enough in
# practice; cheap enough to not worry about tuning further.
_PLACEMENT_RETRY_FRAMES = 5


@dataclasses.dataclass(frozen=True, slots=True)
class WindowPosition:
    x: int
    y: int


class DisplayClient:
    """
    Renders a :class:`wbb.buffer.FrameBuffer` into an SDL2 window.

    Parameters
    ----------
    buffer:
        The FrameBuffer to read frames from (anything with an async
        ``next_frame(timeout)`` coroutine and ``.width``/``.height``).
    title:
        Window title (cosmetic).
    wm_class:
        Stable window-class identifier the KWin and X11 placement
        backends match the window by. Give each concurrent
        DisplayClient a distinct value.
    filters:
        Ordered list of ``(np.ndarray) -> np.ndarray`` callables, same
        contract as wbb.filters. Run in a thread-pool executor (see
        module docstring point 3) — must still be safe to call from a
        thread other than the one that constructed them; wbb's own
        filters.py functions all are (pure functions over numpy
        arrays, no shared mutable state).
    on_mouse_event / on_key_event / on_scroll_event:
        Same callback signatures as the old GTK4 client.
    window_size:
        Pin the window to a fixed size. If omitted, auto-sizes to the
        buffer's native dimensions.
    position:
        Initial (x, y) pixel position, **local to the monitor selected
        by `monitor=`** — not local to (0, 0) of the whole virtual
        desktop. See ``monitor=`` below; this is the fix for the
        multi-monitor bug.
    monitor:
        Which monitor ``position`` is local to, as an index into
        ``wbb_display.list_displays()`` (0 = whatever SDL enumerates
        first — not guaranteed to match physical left-to-right order;
        call ``list_displays()`` yourself and check each entry's
        ``.x``/``.y``/``.width``/``.height`` if you need to know which
        index is which monitor on your setup). Defaults to 0.
    always_on_top, borderless, click_through:
        Same semantics as the old GTK4 client. ``click_through`` is
        real on X11 sessions (Shape extension), and a clean, explained
        no-op under KWin's Wayland scripting backend (no mechanism
        exists there — see placement/kwin.py). Check
        ``is_click_through_active()`` after ``run_async()`` starts if
        you need to know whether it actually took effect — see point 2
        in the module docstring for why this one in particular matters
        to check, unlike always_on_top/position.
    max_fps:
        Caps how often frames are pushed to the window, independent of
        the renderer's own vsync (which may silently be unavailable —
        see _window.py). Default 60. 0 disables the cap. See module
        docstring point 3 for why this exists — there was no cap at
        all before.
    """

    def __init__(
        self,
        buffer: Any,
        *,
        title: str = "wbb",
        wm_class: str = "wbb-display",
        filters: Optional[list[Filter]] = None,
        on_mouse_event: Optional[MouseCallback] = None,
        on_key_event: Optional[KeyCallback] = None,
        on_scroll_event: Optional[ScrollCallback] = None,
        window_size: Optional[tuple[int, int]] = None,
        position: tuple[int, int] = (0, 0),
        monitor: int = 0,
        always_on_top: bool = False,
        borderless: bool = False,
        click_through: bool = False,
        max_fps: float = 60.0,
    ) -> None:
        if not (
            isinstance(position, tuple)
            and len(position) == 2
            and all(isinstance(v, int) for v in position)
        ):
            raise ValueError(f"position must be an (x, y) tuple of ints, got {position!r}")

        self._buf = buffer
        self._title = title
        self._wm_class = wm_class
        self._filters = filters or []
        self._on_mouse = on_mouse_event
        self._on_key = on_key_event
        self._on_scroll = on_scroll_event
        self._fixed_window_size = window_size
        self._requested_local_position = position
        self._monitor_index = monitor
        self._always_on_top = always_on_top
        self._borderless = borderless
        self._click_through_requested = click_through
        self._click_through_active = False
        self._max_fps = max_fps
        self._min_frame_interval = (1.0 / max_fps) if max_fps > 0 else 0.0

        self._win: Optional[SDLWindow] = None
        self._placement = None
        self._stop_requested = False
        self._caller_loop: Optional[asyncio.AbstractEventLoop] = None

        # See module docstring point 4 / _PLACEMENT_RETRY_FRAMES.
        # Counts down once run_async()'s loop starts; while > 0, each
        # iteration re-applies always_on_top/position before doing
        # anything else.
        self._placement_retries_remaining = _PLACEMENT_RETRY_FRAMES

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------
    def run(self) -> None:
        asyncio.run(self.run_async())

    async def run_async(self) -> None:
        """
        Open the window and run until ``stop()`` is called, the window
        is closed by the user, or the FrameBuffer iterator ends.
        """
        self._caller_loop = asyncio.get_running_loop()

        init_w, init_h = self._fixed_window_size or (self._buf.width, self._buf.height)
        self._win = SDLWindow(
            init_w,
            init_h,
            title=self._title,
            wm_class=self._wm_class,
            borderless=self._borderless,
        )

        handle = self._win.native_handle()
        self._placement = select_backend(handle, wm_class=self._wm_class)

        # First attempt, same as before — on backends/compositors that
        # don't hit the XWayland first-map race (or that aren't racing
        # at all, e.g. a window that was already mapped from a prior
        # run via set_position()), this is all that's needed. The
        # retries below are what catch the case where this one is lost.
        self._apply_placement(init_w, init_h)

        if self._click_through_requested:
            self._click_through_active = self._placement.set_click_through(True)

        try:
            last_push = 0.0
            while not self._stop_requested:
                if self._placement_retries_remaining > 0:
                    # See module docstring point 4. Re-send
                    # always_on_top/position for the first few frames
                    # so a configure request lost to XWayland's first
                    # map/configure round-trip gets a second (third,
                    # fourth...) chance, instead of betting everything
                    # on the single attempt above landing after the
                    # window is actually mapped.
                    self._placement_retries_remaining -= 1
                    self._apply_placement(init_w, init_h)

                for ev in self._win.poll_events():
                    await self._dispatch_event(ev)
                    if self._stop_requested:
                        break
                if self._stop_requested:
                    break

                # See module docstring point 3: next_frame() does NOT
                # return None on timeout (wbb.buffer.FrameBuffer's
                # cv.wait_for result is discarded internally), it
                # re-reads whatever was last committed — harmless to
                # re-push, not a correctness issue.
                frame = await self._buf.next_frame(timeout=1.0)

                now = time.monotonic()
                if self._min_frame_interval and (now - last_push) < self._min_frame_interval:
                    # Within the fps cap's window since the last push —
                    # skip rendering this frame entirely (filters
                    # included) rather than just skipping the present.
                    # This is the actual fix for unbounded CPU use: the
                    # expensive part is the filter chain + texture
                    # upload, not SDL_RenderPresent, so the cap has to
                    # gate entry into that work, not just the final
                    # blit.
                    continue
                last_push = now

                arr = await self._run_filters(frame.data)
                self._win.push_frame(arr)

                await asyncio.sleep(0)
        finally:
            if self._win is not None:
                self._win.close()
                self._win = None

    async def _run_filters(self, arr: np.ndarray) -> np.ndarray:
        """
        Apply the filter chain off the asyncio loop, in the default
        thread-pool executor. See module docstring point 3: filters
        like filters.scale() (Pillow resize) and filters.blur()
        (scipy uniform_filter) are real CPU work — tens of
        milliseconds is easy to hit at viewport-sized arrays — and
        running that inline on the same loop that's also polling SDL
        events and awaiting next_frame() means every other coroutine
        sharing this loop (BrowserBridge's CDP recv loop, your own
        automation code) stalls for the duration. wbb's own
        filters.py functions are all pure functions over numpy arrays
        with no shared mutable state, so handing them to a thread-pool
        worker is safe without any additional locking.
        """
        if not self._filters:
            return arr

        def _apply() -> np.ndarray:
            result = arr
            for f in self._filters:
                result = f(result)
            return result

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _apply)

    def stop(self) -> None:
        self._stop_requested = True

    # ------------------------------------------------------------------
    # Multi-monitor position resolution
    # ------------------------------------------------------------------
    def _resolve_global_position(
        self, monitor_index: int, local: tuple[int, int]
    ) -> tuple[int, int]:
        """
        Translate a monitor-local (x, y) into the single global
        coordinate space every placement backend actually expects —
        see _window.py's "Multi-monitor coordinates" docstring section
        for why that space, not (0, 0)-per-monitor, is the real
        contract. ``monitor_index`` out of range, or no displays
        enumerable at all, falls back to treating ``local`` as already
        global (i.e. adds (0, 0)) — the same behavior the old, buggy
        code had everywhere, so a misconfigured monitor index degrades
        to the previous behavior rather than a new failure mode.
        """
        displays = list_displays()
        if not displays or not (0 <= monitor_index < len(displays)):
            if displays:
                log.warning(
                    "monitor=%d out of range (found %d displays); treating "
                    "position=%r as already in global coordinates.",
                    monitor_index,
                    len(displays),
                    local,
                )
            return local
        d = displays[monitor_index]
        return (d.x + local[0], d.y + local[1])

    # ------------------------------------------------------------------
    # Placement application (used by both startup and the retry window)
    # ------------------------------------------------------------------
    def _apply_placement(self, width: int, height: int) -> None:
        """
        Re-send always_on_top/position to the active placement
        backend, if either was requested. Idempotent and cheap enough
        to call repeatedly — see module docstring point 4 for why this
        needs to happen more than once early on, rather than exactly
        once at startup.
        """
        if self._placement is None:
            return
        if self._always_on_top:
            self._placement.set_above(True)
        if self._placement.supports_position():
            gx, gy = self._resolve_global_position(
                self._monitor_index, self._requested_local_position
            )
            self._placement.set_position(gx, gy, width, height)

    # ------------------------------------------------------------------
    # Placement passthroughs
    # ------------------------------------------------------------------
    def is_positionable(self) -> bool:
        return self._placement is not None and self._placement.supports_position()

    def is_click_through_active(self) -> bool:
        """
        True only if click_through was requested AND a placement
        backend actually implemented it (currently: X11/EWMH only —
        see placement/x11_ewmh.py). False on KWin's Wayland backend
        even though always_on_top/position work there — this is a
        real, checked platform gap, not a bug; see the module
        docstring's point 2.
        """
        return self._click_through_active

    def set_position(self, position: tuple[int, int], *, monitor: Optional[int] = None) -> None:
        """
        Move the window. ``position`` is monitor-local, same
        convention as the constructor's ``position=`` — pass
        ``monitor=`` to target a different monitor than the one given
        at construction time; omit it to keep using the original.

        No-op if called before ``run_async()`` has created the window
        (``self._win``/``self._placement`` are ``None`` until then) —
        same as before. If you need the window positioned correctly
        immediately on startup, pass ``position=``/``monitor=`` to the
        constructor instead; the startup race that used to make a
        single early ``set_position()`` unreliable is handled
        internally now (see module docstring point 4), not by calling
        this method early.
        """
        if self._win is None or self._placement is None:
            return
        mon = monitor if monitor is not None else self._monitor_index
        w, h = self._fixed_window_size or (self._buf.width, self._buf.height)
        self._requested_local_position = position
        self._monitor_index = mon
        gx, gy = self._resolve_global_position(mon, position)
        self._placement.set_position(gx, gy, w, h)
        # Re-arm a few retries: a position change requested well after
        # startup is not racing the initial map, but it costs nothing
        # to also cover the (rarer) case of a monitor hot-plug or a
        # compositor-side reset racing this particular call.
        self._placement_retries_remaining = max(
            self._placement_retries_remaining, _PLACEMENT_RETRY_FRAMES
        )

    def get_position(self) -> WindowPosition:
        """Returns the last-requested position, monitor-local (same
        convention as set_position/the constructor) — not the global
        coordinate actually sent to the placement backend."""
        return WindowPosition(
            x=self._requested_local_position[0], y=self._requested_local_position[1]
        )

    def set_always_on_top(self, above: bool) -> None:
        if self._placement is not None:
            self._placement.set_above(above)
        self._always_on_top = above

    def set_click_through(self, enabled: bool) -> bool:
        """Toggle click-through after startup. Returns whether it
        actually took effect — same contract as the constructor's
        click_through= flag, see is_click_through_active()."""
        if self._placement is None:
            return False
        self._click_through_active = self._placement.set_click_through(enabled)
        self._click_through_requested = enabled
        return self._click_through_active

    # ------------------------------------------------------------------
    # Input dispatch
    # ------------------------------------------------------------------
    async def _dispatch_event(self, ev: dict) -> None:
        kind = ev["kind"]
        if kind == "quit":
            self._stop_requested = True
        elif kind == "mouse" and self._on_mouse is not None:
            await self._fire_callback(
                self._on_mouse, ev["event_type"], ev["x"], ev["y"], ev["button"]
            )
        elif kind == "scroll" and self._on_scroll is not None:
            await self._fire_callback(self._on_scroll, ev["dx"], ev["dy"])
        elif kind == "key" and self._on_key is not None:
            await self._fire_callback(self._on_key, ev["event_type"], ev["key"])

    async def _fire_callback(self, cb: Callable[..., Any], *args: Any) -> None:
        try:
            result = cb(*args)
        except Exception:
            log.exception("Error in DisplayClient input callback")
            return
        if asyncio.iscoroutine(result):
            await result
