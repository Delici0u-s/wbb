"""
tools_diagnose_display.py — not an example. Run this when a window is
blank or behaving oddly and you want to know which half is at fault.

    py tools_diagnose_display.py
    URL=https://wikipedia.org py tools_diagnose_display.py
    NOFILTERS=1 py tools_diagnose_display.py

It logs one line per pushed frame:

    push #1 fid=7 arr=(648, 768, 4) uint8 contig=False mean=132.8
            alpha_mean=255.0 tex=(768, 648) win=768x648 resized=True

  no lines at all      -> the loop never reached push_frame; the problem
                          is upstream (no frames arriving, or the loop
                          exited). Check the "bridge delivered N frames"
                          count at the end.
  mean ~= 0            -> the pixels really are black. A filter, the
                          decode, or the page.
  mean > 0 but blank   -> pixels are fine; the problem is SDL or the WM.
  win != tex           -> the WM refused the resize and SDL_RenderCopy
                          is stretching the texture to hide it.

If the loop ends by itself it logs why ("render loop stopping: ...").
On KWin/XWayland a decorated window can receive a WM close request
without anyone clicking anything, which is why DisplayClient ignores it
unless you pass close_on_window_close=True.

Each change made in the 0.1.5 review can be turned off individually, so
a regression can be bisected without editing code:

    WBB_LEGACY_RENDER_CONFIG=1   renderer clear colour / blend modes
    WBB_LEGACY_STABLE_FRAME=1    copy_latest() before the filter chain
    WBB_LEGACY_EXPOSE=1          SDL_WINDOWEVENT decoding
    WBB_LEGACY_ACK=1             screencast ack ordering
    WBB_LEGACY_PARK=1            1.0s next_frame() park
"""

import asyncio
import logging
import os

import numpy as np

from wbb import BrowserBridge, DisplayClient, FrameBuffer, filters, _compat

URL = os.environ.get("URL", "https://wikipedia.org")
WIDTH, HEIGHT = 1280, 720
CROP = (128, 0, 768, 648)


def vignette(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    cx, cy = w / 2, h / 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
    mask = (1 - dist.clip(0, 1) ** 2)[..., np.newaxis].astype(np.float32)
    out = (frame.astype(np.float32) * mask).clip(0, 255).astype(np.uint8)
    out[..., 3] = frame[..., 3]
    return out


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print("legacy switches active:", _compat.active() or "none")

    use_filters = os.environ.get("NOFILTERS", "") not in ("1", "true")
    pipeline = (
        [filters.colorize(r=0.95, g=0.95, b=1.1), vignette, filters.crop(*CROP)]
        if use_filters
        else []
    )
    win_size = (CROP[2], CROP[3]) if use_filters else (WIDTH, HEIGHT)

    buf = FrameBuffer("diag", WIDTH, HEIGHT)
    delivered = 0

    def on_frame(frame):
        nonlocal delivered
        delivered += 1
        if delivered <= 3 or delivered % 60 == 0:
            print(f"  [bridge] frame {delivered} id={frame.frame_id} "
                  f"mean={float(frame.data[..., :3].mean()):.1f}")

    try:
        async with BrowserBridge(buf, width=WIDTH, height=HEIGHT) as br:
            br.on("frame", on_frame)
            await br.navigate(URL)
            display = DisplayClient(
                buf,
                title="wbb — diagnose",
                filters=pipeline,
                window_size=win_size,
                debug_frames=True,
            )
            await display.run_async()
    finally:
        print(f"bridge delivered {delivered} frames in total")
        buf.close()
        buf.unlink()


if __name__ == "__main__":
    asyncio.run(main())
