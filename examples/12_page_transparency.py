"""
12_page_transparency.py — transparency authored by the page itself.

11 gets its alpha from a filter, which is cheap and always works. This
one takes the alpha straight from the page, so a CSS gradient with a
soft edge or a `border-radius` with antialiasing comes through as real
per-pixel alpha.

The cost is the screencast format. JPEG has no alpha channel, so the
whole stream has to switch to PNG, which also gives up the
libjpeg-turbo decode path. Measured at 1280x720: roughly 15-25 ms to
encode a flat UI page and 45-70 ms one with photographs, plus about
10-16 ms to decode. Budget 30-90 ms per frame on top of everything
else, which is why screencast_max_fps is pinned low here. Below about
640x360 it is comfortable at 30 fps; at 720p with mixed content expect
10-15.

If the overlay comes out opaque, run this with PROBE=1: it skips the
window entirely and prints the alpha value Chrome actually produced.
"""

import asyncio
import os

from wbb import (
    BrowserBridge,
    DisplayClient,
    FrameBuffer,
    WindowType,
    centered_position,
    preinit_alpha,
)
from _pages import data_url

W, H = 420, 220

SOFT = data_url(
    "<style>html,body{margin:0;background:transparent}"
    "div{margin:36px;height:120px;border-radius:60px;"
    "background:radial-gradient(circle at 30% 40%,"
    "rgba(120,190,255,0.95),rgba(20,30,60,0.15));"
    "color:#fff;font:600 24px/120px system-ui;text-align:center}</style>"
    "<div>page alpha</div>"
)


async def probe() -> None:
    """Is Chrome producing alpha at all? No window involved."""
    buf = FrameBuffer("probe", W, H)
    try:
        async with BrowserBridge(
            buf,
            width=W,
            height=H,
            screencast_format="png",
            transparent_background=True,
        ) as br:
            await br.navigate(SOFT)
            await asyncio.sleep(1.5)
            frame = await buf.next_frame(timeout=2.0)
            empty = int(frame.data[3, 3, 3])
            centre = int(frame.data[H // 2, W // 2, 3])
            del frame
            print(f"alpha in the page margin: {empty}")
            print(f"alpha in the shape      : {centre}")
            print(
                "Chrome honours the transparent background."
                if empty == 0
                else "Chrome ignored it — use the filter approach in 11 instead."
            )
    finally:
        buf.close()
        buf.unlink()


async def overlay() -> None:
    # See 11: the ARGB visual must be chosen before SDL video comes up,
    # and centered_position() initialises it.
    preinit_alpha()

    buf = FrameBuffer("pagealpha", W, H)
    try:
        async with BrowserBridge(
            buf,
            width=W,
            height=H,
            screencast_format="png",
            transparent_background=True,
            screencast_max_fps=15,
        ) as br:
            await br.navigate(SOFT)

            display = DisplayClient(
                buf,
                title="wbb page alpha",
                wm_class="wbb-page-alpha",
                borderless=True,
                always_on_top=True,
                click_through=True,
                alpha=True,
                window_type=WindowType.CRITICAL_NOTIFICATION,
                position=centered_position(W, H, monitor=0),
                monitor=0,
            )
            task = asyncio.create_task(display.run_async())
            await asyncio.sleep(1.0)
            if not display.is_alpha_active():
                print("window is opaque; the page's alpha is being composited "
                      "against black. Run PROBE=1 to check Chrome's side "
                      "separately.")
            await task
    finally:
        buf.close()
        buf.unlink()


if __name__ == "__main__":
    asyncio.run(probe() if os.environ.get("PROBE") else overlay())
