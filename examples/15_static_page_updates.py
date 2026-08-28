"""
15_static_page_updates.py — updating the window when the page has
stopped painting.

Chrome only produces screencast frames when the page repaints. A page
that has finished rendering produces none at all, so the render loop
has nothing new to work with. Three mechanisms keep the window correct
anyway:

  * expose and resize events re-present the current frame automatically
    — drag another window across this one and watch it stay correct;
  * `request_repaint()` re-presents the current frame on demand. It does
    NOT re-run filters, so on its own it changes nothing visible; use it
    after moving or resizing the window from outside the loop;
  * `set_filters()` re-runs the filter chain over the frame already in
    the buffer, so a filter change becomes visible immediately even
    though the browser has sent nothing new.

The page here paints once and then goes completely still. The counter
printed at the end proves the browser delivered nothing during the
filter changes — every visible update came from the display side.
"""

import asyncio

from wbb import (
    BrowserBridge,
    DisplayClient,
    FrameBuffer,
    WindowType,
    centered_position,
    filters,
    preinit_alpha,
)
from _pages import data_url

W, H = 480, 300
SECONDS = 2.5

STATIC = data_url(
    "<body style='margin:0;background:#ffffff'>"
    "<div style='height:300px;background:"
    "linear-gradient(140deg,#f2b705,#d94f04);color:#231f20;"
    "font:700 30px/300px system-ui;text-align:center'>"
    "painted once</div></body>"
)

STAGES = [
    ("full frame", []),
    ("rounded rect, 40px inset", [filters.alpha_rounded_rect(inset=40, radius=26)]),
    ("ellipse", [filters.alpha_ellipse(inset=16)]),
    ("50% opacity", [filters.opacity(0.5)]),
    ("bottom fade", [filters.alpha_gradient(direction="down", start=1.0, end=0.0)]),
]


async def main() -> None:
    preinit_alpha()

    buf = FrameBuffer("static", W, H)
    delivered = 0

    def count(frame) -> None:
        nonlocal delivered
        delivered += 1

    try:
        async with BrowserBridge(buf, width=W, height=H) as br:
            br.on("frame", count)
            await br.navigate(STATIC)

            display = DisplayClient(
                buf,
                title="wbb — static page",
                wm_class="wbb-static",
                borderless=True,
                always_on_top=True,
                click_through=True,
                alpha=True,
                window_type=WindowType.CRITICAL_NOTIFICATION,
                position=centered_position(W, H, monitor=0),
                monitor=0,
            )
            task = asyncio.create_task(display.run_async())
            await asyncio.sleep(2.0)

            frames_before = delivered
            print(f"window alpha active: {display.is_alpha_active()}")
            print(f"browser frames so far: {frames_before}\n")

            for label, pipeline in STAGES:
                if task.done():
                    break
                print(f"  {label}")
                display.set_filters(pipeline)
                await asyncio.sleep(SECONDS)

            print(f"\nbrowser frames during the filter changes: "
                  f"{delivered - frames_before}")
            print("every update above came from set_filters(), not from Chrome.")
            print("Ctrl-C to stop; drag a window across this one first — it "
                  "repaints from the expose event.")
            await task
    finally:
        buf.close()
        buf.unlink()


if __name__ == "__main__":
    asyncio.run(main())
