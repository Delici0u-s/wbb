"""
08_clip_display_to_element.py — Use get_bounds() to clip the display
window to a single page element.

Demonstrates:
  - BrowserBridge.get_bounds() resolving an element by selector/text
    without clicking anything
  - Feeding the resulting box straight into filters.crop()
  - Sizing the DisplayClient window to match the cropped region, the
    same way 01_display_with_filter.py sizes its window to a manually
    chosen crop rectangle — except here the rectangle comes from the
    page itself instead of being hardcoded.

This is a "select an object, then clip the display to that object"
scenario: rather than guessing pixel coordinates up front, ask the page
where the element actually is, then build the crop filter from that.

Caveat
------
The crop region is computed once, right after the element appears. If
the page is responsive/animated and the element later moves or resizes,
this example's window will keep showing the *original* rectangle (now
possibly misaligned) rather than tracking it live — re-querying
get_bounds() on every frame and rebuilding the crop filter is the
straightforward extension if you need that, just at the cost of one
extra Runtime.evaluate() round-trip per frame.
"""

import asyncio
from wbb import BrowserBridge, FrameBuffer, DisplayClient, filters

URL = "https://www.wikipedia.org/"
WIDTH, HEIGHT = 1280, 720

# What we're selecting — Wikipedia's central-featured block (the logo
# + top-10-language links grid) is a clearly bounded sub-region of the
# page, distinct from the full viewport, which makes the "clip" visible
# in the resulting window. (Note: #www-wikipedia-org is the page's
# <body> id, not a sub-region — it would crop to ~the whole viewport,
# which defeats the point of this example.)
SELECTOR = ".central-featured"


async def main() -> None:
    buf = FrameBuffer("ex08", WIDTH, HEIGHT)

    async with BrowserBridge(buf, width=WIDTH, height=HEIGHT) as br:
        await br.navigate(URL)

        # Resolve the element's on-screen box. get_bounds() polls
        # internally (same as wait_for_selector/click_element), so this
        # does not race the page's own load/render timing.
        box = await br.get_bounds(SELECTOR, timeout=10.0)
        if box is None:
            print(f"Never found an element matching {SELECTOR!r} — aborting.")
            buf.close()
            buf.unlink()
            return

        # Bounding boxes from getBoundingClientRect() are floats and can
        # report sub-pixel/edge values; clamp into the buffer's actual
        # pixel grid before handing them to filters.crop(), which slices
        # a numpy array and will raise on an out-of-range region.
        crop_x = max(0, int(box["x"]))
        crop_y = max(0, int(box["y"]))
        crop_w = max(1, min(int(box["width"]), WIDTH - crop_x))
        crop_h = max(1, min(int(box["height"]), HEIGHT - crop_y))

        print(f"Resolved {SELECTOR!r} -> x={crop_x} y={crop_y} w={crop_w} h={crop_h}")

        pipeline = [filters.crop(crop_x, crop_y, crop_w, crop_h)]

        display = DisplayClient(
            buf,
            title=f"wbb — clipped to {SELECTOR}",
            filters=pipeline,
            window_size=(crop_w, crop_h),
        )
        await display.run_async()

    buf.close()
    buf.unlink()


if __name__ == "__main__":
    asyncio.run(main())
