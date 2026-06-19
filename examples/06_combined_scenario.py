"""
06_combined_scenario.py — Display + record + monitor simultaneously.

Demonstrates:
  - Three independent async tasks sharing one FrameBuffer
  - Display window with a live filter
  - Concurrent recording to disk (every N-th frame as PNG)
  - Visual change monitor that logs to stdout
  All composed in a single script, zero library modification.
"""

import asyncio
import time
import numpy as np
from pathlib import Path
from wbb import BrowserBridge, FrameBuffer, DisplayClient, filters

URL = "https://example.com"
WIDTH, HEIGHT = 1280, 720
SNAPSHOTS_DIR = Path("snapshots")
SNAPSHOT_EVERY_N = 60  # frames
CHANGE_THRESHOLD = 0.03


async def record_task(buf: FrameBuffer) -> None:
    """Save every N-th frame as a PNG."""
    SNAPSHOTS_DIR.mkdir(exist_ok=True)
    frame_n = 0
    async for frame in buf:
        frame_n += 1
        if frame_n % SNAPSHOT_EVERY_N == 0:
            path = SNAPSHOTS_DIR / f"frame_{frame_n:06d}.png"
            await asyncio.get_running_loop().run_in_executor(None, frame.save, path)
            print(f"[record] saved {path}")


async def monitor_task(buf: FrameBuffer) -> None:
    """Detect and log visual changes."""
    prev = None
    async for frame in buf:
        curr = frame.data[:HEIGHT // 2, :WIDTH // 2, :3].astype(np.int16)
        if prev is not None:
            diff = np.abs(curr - prev).sum(axis=2)
            frac = (diff > 8).sum() / curr.shape[0] / curr.shape[1]
            if frac > CHANGE_THRESHOLD:
                print(f"[monitor] change {frac:.1%} at {time.strftime('%H:%M:%S')}")
        prev = curr


async def main() -> None:
    buf = FrameBuffer("ex06", WIDTH, HEIGHT)

    pipeline = filters.chain(
        filters.colorize(r=1.0, g=0.9, b=0.9),  # warm tint
        filters.contrast(1.1),
    )

    async with BrowserBridge(buf, width=WIDTH, height=HEIGHT) as br:
        await br.navigate(URL)

        # Run all three consumers concurrently
        display = DisplayClient(buf, title=f"wbb — {URL}", filters=[pipeline])
        t1 = asyncio.create_task(display.run_async(), name="display")
        t2 = asyncio.create_task(record_task(buf), name="record")
        t3 = asyncio.create_task(monitor_task(buf), name="monitor")

        # Wait until display closes; then cancel the others
        await t1
        t2.cancel()
        t3.cancel()
        await asyncio.gather(t2, t3, return_exceptions=True)

    buf.close()
    buf.unlink()


if __name__ == "__main__":
    asyncio.run(main())
