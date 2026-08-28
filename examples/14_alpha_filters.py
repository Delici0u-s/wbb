"""
14_alpha_filters.py — the transparency filters, one at a time.

Cycles through every alpha filter in `wbb.filters` on the same page so
you can see what each does. Change SECONDS to linger longer on each.

Each step calls `DisplayClient.set_filters()`, which re-runs the chain
over the frame already in the buffer — the page here is static, so
nothing new arrives from Chrome and every change you see comes from the
display side.

All of these need `DisplayClient(alpha=True)` and a window that actually
got a 32-bit visual; on an opaque window they are uploaded and
discarded, which looks identical to nothing happening.

    filters.opacity(level)                     whole window, 0.0-1.0
    filters.alpha_rect(x, y, w, h)             keep a rectangle
    filters.alpha_rect(..., inside=0,          punch a hole and leave the
                       outside=None)           rest alone
    filters.alpha_regions([r1, r2, ...])       several rectangles at once
    filters.alpha_rounded_rect(inset=, radius=)
    filters.alpha_ellipse(inset=)
    filters.alpha_color_key(rgb, tolerance=)   key a colour out
    filters.alpha_gradient(direction=)         linear fade

`inside` and `outside` take an alpha in 0-255, or None to leave that
side of the boundary untouched — that is how "keep only this" and "hide
only this" come from the same function.
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

W, H = 480, 320
SECONDS = 3.0

PAGE = data_url(
    "<body style='margin:0;background:#ffffff;color:#10151b;"
    "font:600 22px/1.5 system-ui'>"
    f"<div style='height:{H}px;background:"
    "linear-gradient(135deg,#7fc4ff,#0b3a63);color:#fff;padding:24px'>"
    "alpha filters</div></body>"
)

STAGES = [
    ("no alpha filter (opaque)", []),
    ("opacity(0.45)", [filters.opacity(0.45)]),
    ("alpha_rounded_rect(inset=40, radius=28)",
     [filters.alpha_rounded_rect(inset=40, radius=28)]),
    ("alpha_ellipse(inset=20)", [filters.alpha_ellipse(inset=20)]),
    ("alpha_rect hole in the middle",
     [filters.alpha_rect(160, 100, 160, 120, inside=0, outside=None)]),
    ("alpha_regions: two bands kept",
     [filters.alpha_regions([(0, 40, 480, 80), (0, 200, 480, 80)])]),
    ("alpha_gradient(down)", [filters.alpha_gradient(direction="down")]),
    ("rounded rect + 60% opacity",
     [filters.alpha_rounded_rect(inset=40, radius=28), filters.opacity(0.6)]),
]


async def main() -> None:
    preinit_alpha()

    buf = FrameBuffer("alphatour", W, H)
    try:
        async with BrowserBridge(buf, width=W, height=H, screencast_max_fps=15) as br:
            await br.navigate(PAGE)

            display = DisplayClient(
                buf,
                title="wbb alpha filters",
                wm_class="wbb-alpha-tour",
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
            print(f"window alpha active: {display.is_alpha_active()}")

            while not task.done():
                for label, pipeline in STAGES:
                    if task.done():
                        break
                    print(f"  {label}")
                    display.set_filters(pipeline)
                    await asyncio.sleep(SECONDS)

            await task
    finally:
        buf.close()
        buf.unlink()


if __name__ == "__main__":
    asyncio.run(main())
