"""
09_window_positioning.py — putting the window where you meant to.

Two things make this harder than it looks, and both bite silently.

**Coordinate space.** A position is either global or monitor-local, and
`monitor=` is what selects between them:

    DisplayClient(position=(100, 200))               # global desktop pixel
    DisplayClient(position=(100, 200), monitor=1)    # local to monitor 1

Global is the default. On a multi-monitor layout the monitors live in
one shared coordinate space, so a monitor to the left of the primary has
a negative x and a shorter monitor mounted higher pushes its neighbour's
origin to a positive y. `list_displays()` reports each monitor's origin
and size in that space.

The trap: `centered_position()` returns a **monitor-local** origin, so it
must be paired with a matching `monitor=`. Written without it —

    position=centered_position(W, H, monitor=1)      # WRONG, no monitor=

— a monitor-1-local coordinate is read as a global one and the window
lands on whichever monitor holds the desktop origin. On a single-monitor
setup, or one where monitor 0 sits at (0, 0), that mistake is invisible.
It shows up the moment you plug in a second screen. Pass both, always.

**Positioning is a request.** Nothing underneath reports success: an
EWMH client message has no reply and a KWin script's return value does
not come back over D-Bus. A window manager is free to override the
position — window rules, placement policy, or a `window_type` it treats
as special all do — and the only way to find out is to measure.
`actual_position()` reads the geometry back off the server and returns
**global** coordinates, or None when the active backend cannot measure
(currently anything but X11/EWMH). None means unknown, not wrong.

Note `actual_position()` is global while `get_position()` echoes back
what you last requested, in whatever convention you requested it. They
are different numbers on purpose.

    MONITOR=1 py 09_window_positioning.py       # run the tour on monitor 1
    WBB_LOG=debug py 09_window_positioning.py   # the library's own placement log
    WBB_PLACEMENT=x11 py 09_...py               # force a backend (x11|kwin|none)

If a move does nothing, `WBB_LOG=debug` is the first thing to reach for:
the library reports which mechanism it settled on and what the window's
real geometry turned out to be. If the report says the position is not
measurable, check which backend activated — only the X11/EWMH one can
measure, and it is also the only one that can do click-through.

Borderless here so the window's origin and its frame's origin are the
same point; with decorations they differ and the comparison below needs
a correction for the frame.
"""

import asyncio
import logging
import os

from wbb import BrowserBridge, DisplayClient, FrameBuffer, centered_position
from wbb.display import list_displays, video_driver
from _pages import data_url

# The library logs its placement decisions — which mechanism it used,
# requested vs actual geometry — through the standard logging module and
# says nothing at all by default. Nothing here is wbb-specific; this is
# the ordinary way to see any library's log output.
logging.basicConfig(
    level=os.environ.get("WBB_LOG", "WARNING").upper(),
    format="%(levelname)s %(name)s: %(message)s",
)

W, H = 360, 220
MONITOR = int(os.environ.get("MONITOR", "0"))
MARGIN = 40

PAGE = data_url(
    "<body style='margin:0;height:100vh;display:flex;align-items:center;"
    "justify-content:center;color:#e6edf3;font:600 22px system-ui;"
    "background:linear-gradient(135deg,#2b4c7e,#101b2d)'>"
    "<div>window position</div></body>"
)


def report(display: DisplayClient, label: str) -> None:
    """Print requested vs measured for one placement step.

    Worth doing once in your own code after startup, the same way you
    would check `is_click_through_active()`. Not worth doing per frame —
    it is a couple of X round-trips.
    """
    want = display.get_position()
    got = display.actual_position()
    if got is None:
        print(f"  {label:<22} requested {want.x},{want.y} — not measurable "
              f"on this placement backend")
    else:
        print(f"  {label:<22} requested {want.x},{want.y} "
              f"-> actual {got.x},{got.y} (global)")


async def main() -> None:
    displays = list_displays()
    print("monitors, in the global coordinate space:")
    for d in displays:
        print(f"  [{d.index}] origin ({d.x}, {d.y})  size {d.width}x{d.height}")
    if not displays:
        print("  none enumerable — positioning will fall back to global (0, 0)")
        return
    target = displays[MONITOR] if 0 <= MONITOR < len(displays) else displays[0]
    print(f"using monitor {target.index}")
    # Which video driver SDL picked decides what is possible at all. On
    # a Wayland session "x11" means an XWayland window (EWMH available,
    # geometry measurable, click-through available) and "wayland" means
    # a native surface (the compositor owns positioning outright). The
    # session type alone does not tell you which you got.
    print(f"SDL video driver: {video_driver()!r}\n")

    # Corners of the chosen monitor, as monitor-local coordinates. Each
    # one is paired with monitor=MONITOR below; that pairing is the whole
    # discipline this example is about.
    corners = [
        ("top-left", (MARGIN, MARGIN)),
        ("top-right", (target.width - W - MARGIN, MARGIN)),
        ("bottom-right", (target.width - W - MARGIN, target.height - H - MARGIN)),
        ("bottom-left", (MARGIN, target.height - H - MARGIN)),
    ]

    buf = FrameBuffer("position", W, H)
    try:
        async with BrowserBridge(buf, width=W, height=H, screencast_max_fps=15) as br:
            await br.navigate(PAGE)

            display = DisplayClient(
                buf,
                title="wbb position",
                wm_class="wbb-position",
                borderless=True,
                always_on_top=True,
                # Handed to SDL_CreateWindow, so the window is born here
                # rather than being moved after the window manager has
                # already placed it somewhere of its own choosing.
                position=corners[0][1],
                monitor=MONITOR,
                # Seconds spent re-asserting placement after startup and
                # after each move, before the result is logged. The
                # default (0.75) absorbs a compositor that repositions
                # the window when it maps it. Pass 0 if you are driving
                # placement yourself and do not want the library
                # competing with you.
                placement_settle=0.75,
            )
            task = asyncio.create_task(display.run_async())

            await asyncio.sleep(1.5)
            print(f"placement backend: {display.placement_backend()!r} "
                  f"(mechanism {display.position_method()!r})")
            report(display, "at creation")

            # A move after startup. The coordinate convention set at
            # construction carries over, so these stay monitor-local.
            for label, pos in corners[1:]:
                await asyncio.sleep(1.2)
                display.set_position(pos)
                await asyncio.sleep(1.0)
                report(display, label)

            # set_geometry() sends position and size to the window
            # manager as one change, rather than moving now and resizing
            # whenever the next frame happens to arrive. Restating the
            # current size, as here, makes it an atomic move; to change
            # the size as well, change what the filter chain outputs and
            # let the window follow it — see 13.
            await asyncio.sleep(1.2)
            cx, cy = centered_position(W, H, monitor=MONITOR)
            display.set_geometry(cx, cy, W, H)
            await asyncio.sleep(1.0)
            report(display, "centered")

            print(f"\nfinished on mechanism {display.position_method()!r}. "
                  "If actual and requested disagree, the window manager "
                  "overrode the move; WBB_LOG=debug shows which mechanisms "
                  "were tried.")
            await task
    finally:
        buf.close()
        buf.unlink()


if __name__ == "__main__":
    asyncio.run(main())
