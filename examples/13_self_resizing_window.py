"""
13_self_resizing_window.py — a window that follows the size of its
content, anchored to a corner that does not move.

Handing `push_frame` an array of a different shape resizes the window.
SDL does that by keeping the top-left corner fixed, so a window pinned
to the bottom-right of a monitor walks off the edge as it grows.
`anchor=` recomputes the origin and sends the move and the resize to
the window manager as one geometry change.

`set_geometry()` is the manual version: an atomic move+resize, as
opposed to `set_position()` followed by waiting for the next frame to
change the size, which leaves one frame where the two disagree.

This resizes on every single frame at 60 fps to show that continuous
resizing is smooth, then walks the window around the monitor with
`set_geometry()`. Watch the anchored corner during the pulsing phase —
it should stay put. Set `ANCHOR = Anchor.TOP_LEFT` to see the default
behaviour, where the opposite corner sweeps instead.
"""

import asyncio
import math
import time

import numpy as np

from wbb import Anchor, BrowserBridge, DisplayClient, FrameBuffer, centered_position
from wbb.display import list_displays
from _pages import data_url

W, H = 800, 600
ANCHOR = Anchor.BOTTOM_RIGHT
mon_idx = 0  # main monitor
FPS = 180  # main monitor

PAGE = data_url(
    "<body style='margin:0;color:#fff;font:600 44px/1.3 system-ui;padding:20px;"
    "background:linear-gradient(135deg,#1b6ca8,#0c2d48)'>"
    "<div id=t>resizing</div>"
    "<script>let n=0;setInterval(()=>{t.textContent="
    "new Date().toLocaleTimeString()+' '+(n++);},16)</script></body>"
)


def pulsing_crop(hz: float = 0.6):
    """Crop to a rectangle whose size changes every frame."""
    start = time.monotonic()

    def _filter(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        phase = (time.monotonic() - start) * hz * 2 * math.pi
        k = 0.45 + 0.55 * (1 + math.sin(phase)) / 2
        # Even dimensions keep the texture upload aligned; the & ~1 is
        # cosmetic, not required.
        return frame[: max(160, int(h * k)) & ~1, : max(220, int(w * k)) & ~1]

    return _filter


async def main() -> None:
    buf = FrameBuffer("resizer", W, H)
    try:
        # 60 fps: the resize path is per-frame work, so this is the
        # interesting case. The texture is grow-only and only the used
        # sub-rectangle is uploaded, so a changing frame size does not
        # reallocate anything after the first couple of seconds.
        async with BrowserBridge(buf, width=W, height=H, screencast_max_fps=60) as br:
            await br.navigate(PAGE)

            display = DisplayClient(
                buf,
                title="wbb — self-resizing",
                wm_class="wbb-resizer",
                borderless=True,
                always_on_top=True,
                filters=[pulsing_crop()],
                anchor=ANCHOR,
                position=centered_position(W, H, monitor=mon_idx),
                monitor=mon_idx,
                max_fps=FPS,
            )
            task = asyncio.create_task(display.run_async())

            await asyncio.sleep(6.0)

            # Walk it around the monitor. Each call is one move+resize,
            # not a move followed by a resize a frame later.
            print("walking the window with set_geometry()")
            centre_x, centre_y = centered_position(560, 420, monitor=0)
            offsets = [(-300, -200), (300, -200), (300, 200), (-300, 200), (0, 0)]
            for dx, dy in offsets:
                if task.done():
                    break
                display.set_geometry(max(0, centre_x + dx), max(0, centre_y + dy), 560, 420)
                await asyncio.sleep(1.2)

            print("back to pulsing; Ctrl-C to stop")
            await task
    finally:
        buf.close()
        buf.unlink()


if __name__ == "__main__":
    asyncio.run(main())
