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

   Fix, in two halves. First, a requested position is handed to
   ``SDL_CreateWindow`` rather than ``SDL_WINDOWPOS_UNDEFINED`` (see
   ``run_async``), so the window is *born* in the right place and the
   compositor's own initial-placement logic has nothing to decide. That
   is a better position to negotiate from than any amount of moving
   afterwards. Second, ``set_above()``/``set_position()`` are re-sent
   on a fixed wall-clock cadence for ``placement_settle`` seconds after
   startup (``_settle_placement`` below), covering a WM that
   repositions on map regardless, and the result is read back and
   logged once (``_report_placement``). The readback is the important
   part: every mechanism underneath is fire-and-forget, so before it
   there was no way to tell a refused move from a successful one.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections.abc import Callable
from typing import Any, Optional

import numpy as np

from .. import _compat
from .. import filters_alpha as _filters_alpha
from .geometry import Anchor, anchored_origin
from .placement import select_backend
from .window_type import WindowType
from ._window import SDLWindow, list_displays, preinit_alpha

log = logging.getLogger(__name__)

Filter = Callable[[np.ndarray], np.ndarray]
MouseCallback = Callable[[str, float, float, int], Any]
KeyCallback = Callable[[str, str], Any]
ScrollCallback = Callable[[float, float], Any]

# How long, in wall-clock seconds, to keep re-applying
# always_on_top/position after startup, and how far apart the attempts
# are spaced. See module docstring point 4 for what this absorbs.
#
# This used to be a count of render-loop iterations (five), on the
# assumption that five iterations is "on the order of ~100ms". It is
# not: an iteration lasts until the frame buffer's generation counter
# moves (buffer.py's next_frame), so the same five iterations span
# ~330ms at a 15fps screencast and ~80ms at 60fps. The duration that
# matters here is the window manager's, not the page's, so it is now
# measured in seconds and the attempt rate is fixed independently of
# frame rate.
_PLACEMENT_SETTLE_SECONDS = 0.75
_PLACEMENT_ATTEMPT_INTERVAL = 0.1


# Sentinel for set_position(monitor=...): distinguishes "argument omitted,
# keep the current mode" from an explicit monitor=None (which now means
# global virtual-desktop coordinates, a real value).
_KEEP: object = object()


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
        arrays, no shared mutable state). The last filter's output
        array may be mutated in place before upload (see
        ``premultiply`` and ``_prepare_alpha``), so a filter must not
        return a buffer it intends to reuse on a later frame.
    on_mouse_event / on_key_event / on_scroll_event:
        Same callback signatures as the old GTK4 client.
    window_size:
        Pin the window to a fixed size. If omitted, auto-sizes to the
        buffer's native dimensions.
    position:
        Initial (x, y) pixel position. Its coordinate space is set by
        ``monitor=`` below. With the default (``monitor=None``) it is a
        **global virtual-desktop** pixel — placeable anywhere across all
        monitors, including a monitor whose global origin is offset
        (negative ``x`` to the left of primary, positive ``y`` for a
        taller monitor sitting below a shorter one mounted higher). This
        is what lets you drop the window at an arbitrary desktop
        position.
    monitor:
        Selects how ``position`` is interpreted.

        * ``None`` (default) — ``position`` is in global virtual-desktop
          coordinates. Use this to place the window anywhere on the
          unified desktop. ``list_displays()`` reports each monitor's
          global origin/size if you want to compute a target relative to
          a specific monitor yourself.
        * ``int`` — ``position`` is local to that monitor, as an index
          into ``list_displays()`` (0 = whatever SDL enumerates first,
          not guaranteed left-to-right; check each entry's
          ``.x``/``.y``/``.width``/``.height``). The monitor's origin is
          added to ``position`` to get the global coordinate.

        Note: a previous version defaulted this to ``0`` and *always*
        ran monitor-local, which anchored every position to monitor 0's
        top-left and made it impossible to reach desktop rows outside
        that monitor's vertical span — the multi-monitor placement
        regression. The default is now global to avoid that.
    always_on_top, borderless, click_through:
        Same semantics as the old GTK4 client. ``click_through`` is
        real on X11 sessions (Shape extension), and a clean, explained
        no-op under KWin's Wayland scripting backend (no mechanism
        exists there — see placement/kwin.py). Check
        ``is_click_through_active()`` after ``run_async()`` starts if
        you need to know whether it actually took effect — see point 2
        in the module docstring for why this one in particular matters
        to check, unlike always_on_top/position.
    placement_settle:
        Seconds after startup (and after each ``set_position()``) during
        which always_on_top/position are re-asserted, roughly every
        100ms, before the requested-vs-actual result is logged once.
        Default 0.75. Set to 0 to apply placement exactly once and
        report immediately — useful if you are re-placing the window
        yourself and do not want the library competing with you.
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
        position: Optional[tuple[int, int]] = None,
        monitor: Optional[int] = None,
        always_on_top: bool = False,
        borderless: bool = False,
        click_through: bool = False,
        max_fps: float = 60.0,
        resizable: bool = True,
        window_type: WindowType = WindowType.NORMAL,
        alpha: bool = False,
        gamescope_overlay: bool = False,
        anchor: Anchor = Anchor.TOP_LEFT,
        idle_repaint_interval: float = 0.25,
        x11_display_name: Optional[str] = None,
        debug_frames: bool = False,
        close_on_window_close: bool = False,
        premultiply: bool = True,
        placement_settle: float = _PLACEMENT_SETTLE_SECONDS,
    ) -> None:
        if position is not None and not (
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
        # None => no position requested: let the compositor place the
        # window naturally (do NOT force it to (0,0), which would slam it
        # into the desktop origin — that origin can even be a different
        # monitor than you expect on offset multi-monitor layouts).
        self._position_requested = position is not None
        self._requested_local_position = position if position is not None else (0, 0)
        # None => position is global virtual-desktop coordinates (default
        # when a position IS given); an int => position is local to that
        # monitor's origin. See _resolve_global_position for the two modes.
        self._monitor_index: Optional[int] = monitor
        self._always_on_top = always_on_top
        self._borderless = borderless
        self._click_through_requested = click_through
        self._click_through_active = False
        self._max_fps = max_fps
        self._min_frame_interval = (1.0 / max_fps) if max_fps > 0 else 0.0
        self._resizable = resizable
        self._window_type = window_type
        self._alpha = alpha
        self._gamescope_overlay = gamescope_overlay
        self._anchor = Anchor(anchor)
        self._idle_repaint_interval = 0.0 if _compat.LEGACY_PARK else idle_repaint_interval
        self._debug_frames = debug_frames
        self._debug_pushes = 0
        self._close_on_window_close = close_on_window_close
        self._premultiply = premultiply
        self._pm_scratch: Optional[np.ndarray] = None
        self._x11_display_name = x11_display_name

        # The window's *current* size, kept up to date as push_frame
        # resizes it. Previously the only size anywhere near placement
        # was the construction size, so every re-placement after a
        # resize sent stale geometry.
        self._current_size: tuple[int, int] = (0, 0)
        self._last_frame_id: int = -1
        self._repaint_pending: bool = False
        self._refilter_pending: bool = False
        self._last_frame: Any = None
        # Reusable landing pad for copy_latest(); see _stable_frame().
        self._scratch: Optional[np.ndarray] = None

        self._win: Optional[SDLWindow] = None
        self._placement = None
        self._stop_requested = False
        self._caller_loop: Optional[asyncio.AbstractEventLoop] = None

        # See module docstring point 4 / _PLACEMENT_SETTLE_SECONDS. The
        # deadline is armed when run_async()'s loop starts (and re-armed
        # by set_position); until it passes, each attempt re-applies
        # always_on_top/position. _placement_reported makes the
        # requested-vs-actual summary fire exactly once per settle
        # window rather than once per attempt.
        self._placement_settle = max(0.0, float(placement_settle))
        self._placement_deadline = 0.0
        self._placement_next_attempt = 0.0
        self._placement_reported = True
        # The "nothing moved it" warning is long and actionable exactly
        # once. On a window manager that never honours a move, every
        # later set_position() would otherwise repeat it verbatim, which
        # turns a useful diagnostic into noise that hides the next real
        # one. Fires in full the first time, at debug after that.
        self._placement_failure_logged = False

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
        self._current_size = (init_w, init_h)

        # Resolve the requested position BEFORE creating the window, so
        # it can be handed to SDL_CreateWindow instead of
        # SDL_WINDOWPOS_UNDEFINED. A window born at the right place
        # never gives the window manager's placement policy a chance to
        # decide otherwise, which is a strictly better position to
        # negotiate from than moving it afterwards. The placement
        # backend still re-asserts the position during the settle window
        # below, because a WM may reposition on map regardless.
        init_pos: Optional[tuple[int, int]] = None
        if self._position_requested:
            if self._alpha and self._monitor_index is not None:
                # Monitor-local resolution calls list_displays(), which
                # initialises SDL video — and the ARGB visual has to be
                # chosen before that happens or the window comes out
                # 24-bit (see display/alpha.py's prepare()). Global mode
                # (monitor=None) short-circuits without touching SDL, so
                # this is only needed for the int case.
                preinit_alpha(self._x11_display_name)
            init_pos = self._resolve_global_position(
                self._monitor_index, self._requested_local_position
            )

        self._win = SDLWindow(
            init_w,
            init_h,
            title=self._title,
            wm_class=self._wm_class,
            borderless=self._borderless,
            resizable=self._resizable,
            window_type=self._window_type,
            alpha=self._alpha,
            gamescope_overlay=self._gamescope_overlay,
            x11_display_name=self._x11_display_name,
            position=init_pos,
        )

        handle = self._win.native_handle()
        self._placement = select_backend(handle, wm_class=self._wm_class)

        if self._click_through_requested:
            self._click_through_active = self._placement.set_click_through(True)

        # Arming the settle window schedules the first attempt for
        # `now`, so the loop's first iteration applies placement
        # immediately — there is no separate up-front _apply_placement()
        # call any more. Keeping one would have made the first attempt
        # land before select_backend's window has had a single event-loop
        # turn, which is the attempt most likely to be lost to the
        # first-map race and the least worth spending a round-trip on.
        self._arm_placement_settle()

        try:
            last_push = 0.0
            while not self._stop_requested:
                self._settle_placement()

                for ev in self._win.poll_events():
                    await self._dispatch_event(ev)
                    if self._stop_requested:
                        break
                if self._stop_requested:
                    break

                # next_frame() does NOT signal a timeout — it re-reads
                # whatever was last committed. frame_id is what tells
                # "new frame" apart from "the park expired", so the
                # filter chain does not re-run on a frame already
                # rendered. The park is short (idle_repaint_interval)
                # rather than 1.0s so a window whose page has stopped
                # painting still self-heals after a resize.
                frame = await self._buf.next_frame(
                    timeout=self._idle_repaint_interval or 1.0
                )

                if frame.frame_id == 0:
                    # Nothing has ever been committed to this buffer. The
                    # zeroed segment is not a frame; pushing it paints the
                    # window solid black until the first real frame lands.
                    await asyncio.sleep(0)
                    continue

                if frame.frame_id == self._last_frame_id:
                    if self._refilter_pending:
                        self._refilter_pending = False
                        self._repaint_pending = False
                        arr = await self._run_filters(self._stable_frame(frame))
                        new_h, new_w = int(arr.shape[0]), int(arr.shape[1])
                        if (new_w, new_h) != self._current_size:
                            self._on_size_change(new_w, new_h)
                        self._win.push_frame(self._prepare_alpha(arr))
                    elif self._repaint_pending:
                        self._repaint_pending = False
                        self._win.present_again()
                    await asyncio.sleep(0)
                    continue

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

                data = self._stable_frame(frame)
                arr = await self._run_filters(data)
                self._last_frame_id = frame.frame_id
                self._repaint_pending = False
                self._refilter_pending = False

                # Send the new geometry BEFORE push_frame's
                # SDL_SetWindowSize, so the WM sees one geometry change
                # carrying the new origin and the new size together
                # rather than a resize now and a move a frame later.
                new_h, new_w = int(arr.shape[0]), int(arr.shape[1])
                if (new_w, new_h) != self._current_size:
                    self._on_size_change(new_w, new_h)

                # _prepare_alpha, not push_frame(arr) directly. This is
                # the path every frame takes; the re-filter branch above
                # is only reached after set_filters(). Missing it here
                # meant an alpha window uploaded straight (unassociated)
                # alpha for its entire lifetime, which a premultiplied
                # compositor renders as opaque — see _prepare_alpha.
                arr = self._prepare_alpha(arr)
                resized = self._win.push_frame(arr)
                if self._debug_frames:
                    self._log_push(frame, arr, resized)

                await asyncio.sleep(0)
        finally:
            if self._win is not None:
                self._win.close()
                self._win = None

    def _prepare_alpha(self, arr: np.ndarray) -> np.ndarray:
        """Premultiply the frame if the window has a real alpha visual.

        X11 composites 32-bit ARGB windows with premultiplied alpha —
        XRender's PictOpOver, and `glBlendFunc(GL_ONE,
        GL_ONE_MINUS_SRC_ALPHA)` in KWin's OpenGL backend. Straight
        (unassociated) alpha, which is what every intuitive way of
        writing an alpha channel produces, is misread by that: white
        with alpha 0 means "full-intensity white at zero coverage" and
        renders as opaque white, not as nothing. Chrome's PNG frames hit
        the same rule but happen to survive it, because their
        transparent regions are (0, 0, 0, 0) and black is the one colour
        identical under both conventions.

        Doing this once here rather than inside each alpha filter means
        it cannot be applied twice, and it covers frames whose alpha came
        from the decoder rather than from a filter.

        Cost: nothing when the window is opaque, and nothing when the
        frame is fully opaque (`alpha.min() == 255` short-circuits).
        Otherwise one uint16 multiply over H*W*4 out of a workspace
        cached per frame shape — no per-frame allocation. Pass
        premultiply=False if your frames are already premultiplied, or
        set WBB_LEGACY_PREMULTIPLY=1 to disable it without touching the
        call site.

        Mutates `arr` in place when it is writeable, which it is
        whenever a filter chain produced it. Filters must therefore
        return an array the client may own — every filter in
        wbb.filters does (each returns a fresh `frame.copy()`). A
        third-party filter that returns a buffer it keeps and reuses
        across frames would be corrupted and double-premultiplied.
        """
        if (
            not self._premultiply
            or _compat.LEGACY_PREMULTIPLY
            or self._win is None
            or not self._win.alpha_active
        ):
            return arr
        if arr.flags.writeable:
            _filters_alpha._premultiply_inplace(arr)
            return arr
        # Read-only view straight out of shared memory (no filters
        # configured): premultiply into a reusable scratch array so this
        # stays allocation-free per frame.
        if self._pm_scratch is None or self._pm_scratch.shape != arr.shape:
            self._pm_scratch = np.empty(arr.shape, dtype=np.uint8)
        np.copyto(self._pm_scratch, arr)
        _filters_alpha._premultiply_inplace(self._pm_scratch)
        return self._pm_scratch

    def _log_push(self, frame: Any, arr: np.ndarray, resized: bool) -> None:
        """One line per pushed frame: everything needed to tell a data
        problem from a window problem.

        `mean` is the giveaway. Black window + nonzero mean => the pixels
        are fine and the problem is SDL/WM side. Black window + no lines
        at all => the render loop never reached push_frame.
        """
        self._debug_pushes += 1
        if self._debug_pushes > 1 and self._debug_pushes % 30 != 0:
            return
        win_w, win_h = self._win.size() if self._win is not None else (-1, -1)
        log.warning(
            "push #%d fid=%s arr=%s %s contig=%s mean=%.1f alpha_mean=%.1f "
            "tex=%s win=%dx%d resized=%s",
            self._debug_pushes,
            frame.frame_id,
            arr.shape,
            arr.dtype,
            arr.flags["C_CONTIGUOUS"],
            float(arr[..., :3].mean()),
            float(arr[..., 3].mean()),
            getattr(self._win, "_tex_size", None),
            win_w,
            win_h,
            resized,
        )

    def _stable_frame(self, frame: Any) -> np.ndarray:
        """Frame pixels that will not change under us while filters run.

        `frame.data` is a live view into shared memory, and the writer
        keeps flipping between only two segments — so a filter chain
        running in a thread-pool executor for longer than two frame
        intervals can be reading a segment the writer has re-entered.
        With filters configured, copy once into a reusable scratch array
        (one memcpy, no per-frame allocation); with no filters the array
        goes straight into push_frame's memmove on this same thread, so
        the exposure is microseconds and the copy is not worth paying.
        """
        if not self._filters or _compat.LEGACY_STABLE_FRAME:
            return frame.data
        copy_latest = getattr(self._buf, "copy_latest", None)
        if not callable(copy_latest):
            return frame.data
        if self._scratch is None or self._scratch.shape != frame.data.shape:
            self._scratch = np.empty(frame.data.shape, dtype=np.uint8)
        stable = copy_latest(self._scratch)
        return stable.data if stable is not None else frame.data

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
        self, monitor_index: Optional[int], local: tuple[int, int]
    ) -> tuple[int, int]:
        """
        Translate a requested position into the single global
        coordinate space every placement backend actually expects (see
        _window.py's "Multi-monitor coordinates" docstring section).

        Two modes, selected by *monitor_index*:

        * ``None`` — *global mode*. ``local`` is already in global
          virtual-desktop coordinates; return it unchanged. This is the
          mode that lets you place the window at an arbitrary desktop
          pixel, including anywhere on a monitor whose global origin is
          offset (the common multi-monitor case: a shorter monitor
          mounted higher means the taller monitor's origin has a
          positive ``y``, and a monitor to the left has a negative
          ``x``). Use ``list_displays()`` to read those origins and add
          them yourself if you want, or just pass the global pixel you
          actually mean.

        * ``int`` — *monitor-local mode*. ``local`` is local to that
          monitor; its global origin (from ``list_displays()``) is added.
          An out-of-range index, or no displays enumerable at all, falls
          back to treating ``local`` as already global (adds (0, 0)) and
          warns, so a misconfigured index degrades rather than raises.

        The previous regression was that *every* call went through
        monitor-local mode anchored to ``monitor=0``, so a position's
        ``y`` was always measured from monitor 0's top edge — making it
        impossible to reach desktop rows above (or, depending on layout,
        below) that monitor. Global mode removes that anchoring.
        """
        if monitor_index is None:
            return local

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
    def _apply_placement(self) -> None:
        """
        Re-send always_on_top/position to the active placement backend,
        using the window's CURRENT size — not the construction size.

        Idempotent and cheap enough to call repeatedly; see module
        docstring point 4 for why this happens more than once early on.
        """
        if self._placement is None:
            return
        if self._always_on_top:
            self._placement.set_above(True)
        if self._position_requested and self._placement.supports_position():
            gx, gy = self._resolve_global_position(
                self._monitor_index, self._requested_local_position
            )
            w, h = self._current_size
            self._placement.set_position(gx, gy, w, h)

    def _arm_placement_settle(self) -> None:
        """(Re)open the window during which placement is re-asserted."""
        now = time.monotonic()
        self._placement_deadline = now + self._placement_settle
        self._placement_next_attempt = now
        self._placement_reported = False

    def _settle_placement(self) -> None:
        """Re-apply placement on a fixed wall-clock cadence, then report.

        Runs the full settle window rather than stopping early on a
        matching readback, deliberately. A window created at the right
        position (see run_async) reads back correct from the very first
        check, *before* the window manager has mapped it and had its own
        say — so an early exit on "it matches" would stop re-asserting
        exactly in the case that most needs it. The attempts are
        idempotent and there are about eight of them; the handful of
        round-trips saved by exiting early is not worth reintroducing
        that race.

        Cost: one placement call, plus one geometry readback on the
        final attempt, every _PLACEMENT_ATTEMPT_INTERVAL for
        _placement_settle seconds. Nothing at all after that, and
        nothing per frame.
        """
        if self._placement_reported:
            return
        now = time.monotonic()
        if now < self._placement_next_attempt:
            return
        self._placement_next_attempt = now + _PLACEMENT_ATTEMPT_INTERVAL

        self._apply_placement()

        if now < self._placement_deadline:
            return

        # The window has had a full settle window on the current
        # mechanism and still is not where it was asked to go. If the
        # backend has another mechanism, take it and start a fresh
        # window; the backend keeps the choice, so every later move
        # goes straight to whatever worked and pays nothing for the
        # search. A readback of None (backend cannot measure) is not
        # evidence of failure and must not advance anything.
        if self._placement_matches() is False and self._advance_position_method():
            self._placement_deadline = now + self._placement_settle
            return

        self._placement_reported = True
        self._report_placement()

    def _placement_matches(self) -> Optional[bool]:
        """Is the window where it was asked to be? None if unmeasurable."""
        if not self._position_requested or self._placement is None:
            return None
        if not self._placement.supports_position():
            return None
        got = self._placement.actual_position()
        if got is None:
            return None
        want = self._resolve_global_position(
            self._monitor_index, self._requested_local_position
        )
        return got == want

    def _advance_position_method(self) -> bool:
        """Ask the backend for another way to move the window.

        False when the backend has none left, or is old enough not to
        implement the hook at all — a third-party PlacementBackend
        predating this stays working, it just does not participate in
        the search.
        """
        if self._placement is None:
            return False
        advance = getattr(self._placement, "next_position_method", None)
        if not callable(advance):
            return False
        previous = self._position_method_name()
        if not advance():
            return False
        log.info(
            "Placement: %r did not move the window; retrying with %r",
            previous,
            self._position_method_name(),
        )
        return True

    def _position_method_name(self) -> str:
        name = getattr(self._placement, "position_method", None)
        return name() if callable(name) else "?"

    def _report_placement(self) -> None:
        """Log requested vs actual once the settle window has closed.

        This is the only place the library finds out whether placement
        worked. Every mechanism underneath is fire-and-forget: an EWMH
        client message has no reply, and a KWin script's return value is
        not plumbed back over D-Bus. Without this a refused move and a
        successful one are indistinguishable from inside the process,
        which is what makes "the window is in the wrong place" so
        expensive to diagnose.
        """
        if not self._position_requested or self._placement is None:
            return
        want = self._resolve_global_position(
            self._monitor_index, self._requested_local_position
        )
        got = self._placement.actual_position()
        if got is None:
            log.info(
                "Placement: requested global position %s; the %r backend "
                "cannot read the geometry back, so whether it took is "
                "unknown. Check with: xwininfo -id <window> | grep Absolute",
                want,
                getattr(self._placement, "name", "?"),
            )
        elif got == want:
            log.debug(
                "Placement: settled at %s as requested, via %r",
                got,
                self._position_method_name(),
            )
        elif self._placement_failure_logged:
            log.debug(
                "Placement: requested %s, window is at %s (%r); already "
                "reported, not repeating.",
                want,
                got,
                self._position_method_name(),
            )
        else:
            self._placement_failure_logged = True
            log.warning(
                "Placement: requested global position %s but the window is "
                "at %s. Every positioning mechanism this backend has was "
                "tried (last was %r) and none moved it. Things known to "
                "cause this: a window_type the WM treats as special "
                "(notification/OSD types are commonly excluded from "
                "client-initiated moves), a Window Rule matching "
                "wm_class=%r, or a placement policy applied at map time. "
                "The position passed to the constructor is applied at "
                "window-creation time through a different channel and may "
                "still work where a later move does not.",
                want,
                got,
                self._position_method_name(),
                self._wm_class,
            )

    def actual_position(self) -> Optional[WindowPosition]:
        """Where the window really is, in **global** desktop coordinates.

        Note the convention difference from `get_position()`, which
        echoes back the last position you *requested*, in whatever
        convention you requested it (monitor-local if `monitor=` is an
        int). This one is always global, because it is measured off the
        window rather than remembered, and there is no monitor to
        measure it against until you pick one.

        Returns None before `run_async()` has created the window, and on
        backends that cannot read the geometry back (currently anything
        but X11/EWMH — see each backend's `actual_position()`). None
        means "unknown", not "wrong".
        """
        if self._placement is None:
            return None
        pos = self._placement.actual_position()
        return None if pos is None else WindowPosition(x=pos[0], y=pos[1])

    def _on_size_change(self, new_w: int, new_h: int) -> None:
        """The next frame has a different shape; re-anchor and re-place.

        With anchor=TOP_LEFT this only records the new size (SDL's own
        resize already keeps the top-left fixed). With any other anchor
        it recomputes the origin so the anchored point stays put, and
        issues the move+resize as a single placement call —
        _NET_MOVERESIZE_WINDOW and KWin's frameGeometry both carry
        x/y/w/h together, so it is one geometry change, not two.
        """
        old_w, old_h = self._current_size
        if self._anchor is not Anchor.TOP_LEFT and self._position_requested:
            self._requested_local_position = anchored_origin(
                self._anchor,
                self._requested_local_position[0],
                self._requested_local_position[1],
                old_w,
                old_h,
                new_w,
                new_h,
            )
        self._current_size = (new_w, new_h)
        if self._position_requested and self._placement is not None:
            if self._placement.supports_position():
                gx, gy = self._resolve_global_position(
                    self._monitor_index, self._requested_local_position
                )
                self._placement.set_position(gx, gy, new_w, new_h)

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

    def set_position(
        self,
        position: "tuple[int, int] | int",
        position_y: Optional[int] = None,
        *,
        monitor: "Optional[int] | object" = _KEEP,
    ) -> None:
        """
        Move the window. Accepts either a single ``(x, y)`` tuple or two
        separate ints::

            display.set_position((100, 200))
            display.set_position(100, 200)

        Coordinate convention follows ``monitor``:

        * Omit ``monitor`` (the default) — keep whatever mode was set at
          construction or by a previous call. With the constructor's new
          default that means *global virtual-desktop coordinates*.
        * ``monitor=None`` — explicitly use global virtual-desktop
          coordinates: ``position`` is an absolute desktop pixel,
          placeable anywhere across all monitors (this is what restores
          arbitrary placement on offset multi-monitor layouts).
        * ``monitor=<int>`` — ``position`` is local to that monitor's
          origin (see ``list_displays()``).

        No-op if called before ``run_async()`` has created the window
        (``self._win``/``self._placement`` are ``None`` until then). If
        you need the window positioned correctly immediately on startup,
        pass ``position=``/``monitor=`` to the constructor instead; the
        startup race that used to make a single early ``set_position()``
        unreliable is handled internally now (see module docstring
        point 4).
        """
        if position_y is None:
            if not isinstance(position, tuple):
                raise TypeError("expected an (x, y) tuple or two ints")
            pos = position
        else:
            if not isinstance(position, int):
                raise TypeError("expected two ints when passing x and y separately")
            pos = (position, position_y)

        if self._win is None or self._placement is None:
            return

        # _KEEP sentinel => preserve the current mode; an explicit None
        # => global mode; an int => monitor-local. None is now a real,
        # distinct value (global), which is why "keep current" needed its
        # own sentinel rather than reusing None.
        mon = self._monitor_index if monitor is _KEEP else monitor  # type: ignore[assignment]

        w, h = self._current_size or self._fixed_window_size or (
            self._buf.width,
            self._buf.height,
        )
        self._position_requested = True
        self._requested_local_position = pos
        self._monitor_index = mon  # type: ignore[assignment]
        gx, gy = self._resolve_global_position(mon, pos)  # type: ignore[arg-type]
        self._placement.set_position(gx, gy, w, h)
        # Re-arm the settle window: a position change requested well
        # after startup is not racing the initial map, but it costs
        # nothing to also cover a monitor hot-plug or compositor-side
        # reset racing this particular call — and it means a late move
        # gets the same requested-vs-actual report as the initial one.
        self._arm_placement_settle()

    def get_position(self) -> WindowPosition:
        """Returns the last-requested position in whatever convention was
        last used (global by default, or local to ``self._monitor_index``
        if an int monitor was set) — not necessarily the resolved global
        coordinate actually sent to the placement backend."""
        return WindowPosition(
            x=self._requested_local_position[0], y=self._requested_local_position[1]
        )

    def set_geometry(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        monitor: "Optional[int] | object" = _KEEP,
    ) -> None:
        """Move and resize in one call.

        Both placement backends carry x/y/w/h in a single message
        (_NET_MOVERESIZE_WINDOW on X11; frameGeometry under KWin), so
        this is atomic from the window manager's point of view — unlike
        set_position() followed by waiting for push_frame to resize,
        which gives one frame where position and size disagree.

        The size given here is what gets sent to the WM now; the next
        frame with a different shape will still resize the window
        (that is what push_frame does), re-anchored per `anchor=`.

        No-op before run_async() has created the window.
        """
        if self._win is None or self._placement is None:
            return
        mon = self._monitor_index if monitor is _KEEP else monitor  # type: ignore[assignment]
        self._monitor_index = mon  # type: ignore[assignment]
        self._position_requested = True
        self._requested_local_position = (x, y)
        self._current_size = (width, height)
        gx, gy = self._resolve_global_position(mon, (x, y))  # type: ignore[arg-type]
        if self._placement.supports_position():
            self._placement.set_position(gx, gy, width, height)

    def set_filters(self, filters: "Optional[list]") -> None:
        """Replace the filter chain while the loop is running.

        The list is swapped atomically from the loop's point of view —
        the render loop reads `self._filters` once per frame and the
        assignment is a single bytecode — so a chain is never applied
        half-old and half-new. Takes effect on the next frame, and
        requests a repaint so a static page updates immediately rather
        than waiting for one.
        """
        self._filters = list(filters or [])
        self._scratch = None
        # A repaint alone only re-presents the texture that is already
        # uploaded; the filter chain runs when a *new* frame arrives. On
        # a page that has stopped painting there is no new frame, so a
        # filter swap would never become visible. Flag a re-filter, which
        # re-runs the chain over the frame currently in the buffer.
        self._refilter_pending = True
        self.request_repaint()

    def request_repaint(self) -> None:
        """Ask the render loop to present the current frame again.

        The loop parks in next_frame(); a page that has stopped painting
        produces no screencast frames, so nothing presents the window
        until the park expires. This flags a repaint and wakes the
        buffer so the loop returns immediately instead of waiting out
        idle_repaint_interval.
        """
        self._repaint_pending = True
        wake = getattr(self._buf, "wake", None)
        if callable(wake):
            wake()

    def current_size(self) -> tuple[int, int]:
        """The size the client believes the window is, per the last frame.

        Compare against SDLWindow.size() if you suspect the window
        manager refused a resize — they diverge silently otherwise.
        """
        return self._current_size

    def placement_backend(self) -> str:
        """Name of the placement backend that activated for this window.

        One of "x11-ewmh", "kwin", "none" — or "" before `run_async()`
        has created the window. Which one you get decides what is
        actually available: only "x11-ewmh" can read the window's
        geometry back (`actual_position()`) or do click-through, so this
        is the first thing to check when either of those silently does
        nothing. See placement/chain.py for how it is chosen and for the
        WBB_PLACEMENT override.
        """
        return "" if self._placement is None else getattr(self._placement, "name", "?")

    def position_method(self) -> str:
        """Which positioning mechanism the backend is currently using.

        The backend may have several and switches between them based on
        whether the window actually moved (see chain.py and
        placement/x11_ewmh.py). Cosmetic — for logs and diagnostics.
        """
        return self._position_method_name()

    def is_alpha_active(self) -> bool:
        """True if the window really has a per-pixel alpha visual.

        False when alpha= was not requested, or was requested and could
        not be satisfied (no 32-bit visual, no python-xlib). Same
        contract as is_click_through_active(): a checked capability,
        not a promise.
        """
        return bool(self._win is not None and getattr(self._win, "alpha_active", False))

    def is_window_type_active(self) -> bool:
        """True if the requested window_type was written to the window.

        False on WindowType.NORMAL (nothing to write), on non-X11
        subsystems, and when python-xlib is unavailable.
        """
        return bool(self._win is not None and getattr(self._win, "window_type_active", False))

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
            log.info("render loop stopping: SDL_QUIT")
            self._stop_requested = True
        elif kind == "window":
            et = ev["event_type"]
            if et == "close":
                # Opt-in: before 0.1.5 SDL_WINDOWEVENT was not decoded at
                # all, so a CLOSE could never stop the loop. Making it
                # stop by default turned out to end the loop on windows
                # nobody asked to close, so the old behaviour is the
                # default and the event is still delivered to on_window.
                if self._close_on_window_close:
                    log.info("render loop stopping: window close event")
                    self._stop_requested = True
            elif et in ("expose", "resize"):
                # The window's contents are undefined after an expose or
                # a WM-driven resize. Re-present what is already in the
                # texture instead of leaving it until the next frame,
                # which may be a whole park away.
                if self._win is not None:
                    self._win.present_again()
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
