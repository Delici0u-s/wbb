"""
11_transparent_overlay.py — an overlay with a transparent background.

Transparency needs two things, and they are independent:

  * a window with a 32-bit ARGB visual   -> DisplayClient(alpha=True)
  * frames that carry a real alpha channel

This example supplies the alpha itself, with the filters in
`wbb.filters`. That keeps the cheap JPEG screencast and costs one mask
gather per frame — the right default when the transparency is
geometric. For alpha authored by the page (soft gradients, antialiased
curves) see 12, which pays for a PNG screencast instead.

Write straight (unassociated) alpha and let `DisplayClient` handle the
rest: X11 composites ARGB windows with *premultiplied* alpha, so a
white pixel with alpha 0 renders as opaque white rather than as
nothing. `DisplayClient(alpha=True)` premultiplies each frame before
upload, which is why the filters below can just set the alpha channel
and be done. Pass `premultiply=False` if your frames already are.

`display.is_alpha_active()` reports whether the window really got the
ARGB visual; it reads the depth back off the X server rather than
trusting the SDL hint. If it says False the frames still render, just
opaquely, and every alpha filter below becomes a no-op.

Try the other shapes by changing PIPELINE:

    [filters.alpha_rounded_rect(inset=MARGIN, radius=RADIUS)]   # keep the card
    [filters.alpha_color_key((255, 255, 255))]                  # drop the page's white
    [filters.opacity(0.6)]                                      # whole window faded
    [filters.alpha_ellipse(inset=20)]                           # elliptical window
    [filters.alpha_rect(0, 0, 200, 200, inside=0, outside=None)] # punch a hole
"""

import asyncio
import os

import numpy as np

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

W, H = 420, 200

# The card sits inside a margin; the filter makes everything outside it
# transparent. Keep these in sync with the CSS — the filter has no idea
# what the page looks like, which is the point: the alpha is yours to
# define, not the page's.
MARGIN, RADIUS = 30, 18

PIPELINE = [
    filters.alpha_rounded_rect(inset=MARGIN, radius=RADIUS),
]

# INVISIBLE=1 replaces the pipeline with opacity(0) — the whole window
# should vanish. If it does, alpha from filters reaches the screen and
# any remaining problem is the shape of PIPELINE. If the window is still
# fully visible, filter alpha is not reaching the screen at all, and the
# two numbers printed at startup say which side to look at.
#
# This test is only meaningful with premultiplication on (the default).
# opacity(0) over a white page produces (255, 255, 255, 0), and a
# premultiplied compositor draws that as opaque white — so with
# WBB_LEGACY_PREMULTIPLY=1 the window stays fully visible even though
# the alpha is correct and is reaching the compositor. Running the two
# against each other is the clean A/B: default should vanish, legacy
# should not.
if os.environ.get("INVISIBLE"):
    PIPELINE = [filters.opacity(0.0)]

CARD = data_url(
    f"<style>html,body{{margin:0;background:#ffffff}}"
    f"div{{margin:{MARGIN}px;height:{H - 2 * MARGIN}px;box-sizing:border-box;"
    f"padding:26px;border-radius:{RADIUS}px;background:#1b2026;color:#e6edf3;"
    f"font:600 26px system-ui}}</style>"
    "<div>transparent overlay</div>"
)


async def main() -> None:
    # Must happen before anything else initialises SDL video — and
    # centered_position() below does, because it queries monitor bounds.
    # SDL resolves the X11 visual once, inside SDL_Init, so the ARGB
    # visual has to be chosen first or the window comes out 24-bit.
    preinit_alpha()

    buf = FrameBuffer("glass", W, H)
    try:
        async with BrowserBridge(buf, width=W, height=H, screencast_max_fps=15) as br:
            await br.navigate(CARD)

            display = DisplayClient(
                buf,
                title="wbb transparent",
                wm_class="wbb-transparent",
                borderless=True,
                always_on_top=True,
                click_through=True,
                alpha=True,
                window_type=WindowType.CRITICAL_NOTIFICATION,
                filters=PIPELINE,
                position=centered_position(W, H, monitor=0),
                monitor=0,
            )
            task = asyncio.create_task(display.run_async())
            await asyncio.sleep(1.5)

            # Two numbers that say which half is at fault if the overlay
            # comes out as a plain rectangle: a transparent fraction near
            # zero means the filters are not shaping anything, and
            # alpha_active False means the window cannot show it.
            frame = await buf.next_frame(timeout=2.0)
            shaped = frame.data
            for f in PIPELINE:
                shaped = f(shaped)
            transparent = float((shaped[..., 3] == 0).mean())
            mean_alpha = float(shaped[..., 3].mean())
            del frame, shaped
            print(
                f"pipeline output: {transparent:.0%} of pixels alpha=0, "
                f"mean alpha {mean_alpha:.0f}/255"
            )
            print(f"window alpha active: {display.is_alpha_active()}")
            if transparent < 0.05:
                print(
                    "  -> the filters are not shaping anything; the pipeline "
                    "is the problem, not the window."
                )
            elif not display.is_alpha_active():
                print(
                    "  -> the frames are shaped correctly but the window is "
                    "opaque, so the alpha is discarded."
                )
            else:
                print(
                    "  -> both sides look right; if the window still looks "
                    "solid, the compositor is ignoring the alpha."
                )

            await task
    finally:
        buf.close()
        buf.unlink()


if __name__ == "__main__":
    asyncio.run(main())
