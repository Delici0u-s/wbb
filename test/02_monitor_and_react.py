"""
02_monitor_and_react.py — Monitor a page for visual change and react.
"""

import asyncio
import signal
import time
from pathlib import Path

import numpy as np

from wbb import BrowserBridge, FrameBuffer

URL = "https://www.clocktab.com/"
WIDTH, HEIGHT = 1280, 900
DIFF_THRESHOLD = 0.001

LOG_DIR = Path("change_logging")
LOG_DIR.mkdir(exist_ok=True)

REGION = (
    int(WIDTH / 4),
    int(HEIGHT / 4),
    int(WIDTH * 3 / 4),
    int(HEIGHT * 3 / 4),
)  # (x, y, w, h) — watch middle


def pixel_diff_fraction(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).sum(axis=2)
    changed = (diff > 10).sum()
    return changed / (a.shape[0] * a.shape[1])


async def main() -> None:
    buf = FrameBuffer("ex02", WIDTH, HEIGHT)
    x, y, w, h = REGION

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        # First Ctrl-C: ask the loop to wind down gracefully.
        # Ignored on repeat presses until shutdown finishes — that's
        # what stops a second/third Ctrl-C from interrupting cleanup
        # mid-await and leaking the shared-memory segments.
        if not stop_event.is_set():
            print("\nStopping… (press Ctrl-C again only if this hangs)")
            stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            # Not available on Windows — Ctrl-C still raises KeyboardInterrupt
            # there, just without the repeat-press protection below.
            pass

    try:
        print(f"Now watching {URL} for change. Press ctrl-c to stop")
        RUN_DIR = LOG_DIR / time.strftime("%Y-%m-%d_%H-%M-%S")
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        async with BrowserBridge(buf, width=WIDTH, height=HEIGHT) as br:
            br.on("navigate", lambda url: print(f"[nav] {url}"))
            await br.navigate(URL)
            prev = None
            print("Monitoring for visual changes in region", REGION)

            frame_iter = br.frames()
            while not stop_event.is_set():
                get_frame = asyncio.ensure_future(frame_iter.__anext__())
                stop_wait = asyncio.ensure_future(stop_event.wait())
                done, _pending = await asyncio.wait(
                    {get_frame, stop_wait}, return_when=asyncio.FIRST_COMPLETED
                )
                if stop_wait in done:
                    get_frame.cancel()
                    break
                stop_wait.cancel()
                frame = get_frame.result()

                region = frame.crop(x, y, w, h).copy()
                if prev is not None:
                    frac = pixel_diff_fraction(prev, region)
                    if frac > DIFF_THRESHOLD:
                        ts = time.strftime("%H:%M:%S")
                        print(f"[{ts}] Change detected ({frac:.1%} pixels differ) — reacting")
                        await br.click(WIDTH / 2, HEIGHT - 40)
                        # frame.save(f"change_logging/change_{int(time.time())}.png")
                        frame.save(RUN_DIR / f"change_{int(time.time())}.png")
                prev = region
            # br.stop() runs here automatically via __aexit__
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, ValueError):
                pass
        buf.close()
        buf.unlink()
        print("Cleaned up. Bye.")


if __name__ == "__main__":
    asyncio.run(main())
