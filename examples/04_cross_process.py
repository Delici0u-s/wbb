"""
04_cross_process.py — BrowserBridge in one process, DisplayClient in another.

Demonstrates:
  - Named shared-memory buffer as the cross-process IPC primitive
  - Process A: renders into the buffer (no display)
  - Process B: attaches to the buffer and opens a window (no browser)

Run with two terminals:
    # Terminal A (renderer):
    python examples/04_cross_process.py renderer https://example.com

    # Terminal B (viewer):
    python examples/04_cross_process.py viewer
"""

import asyncio
import sys
from wbb import BrowserBridge, FrameBuffer, DisplayClient

NAME = "wbb_cross_proc"
WIDTH, HEIGHT = 1280, 720


async def run_renderer(url: str) -> None:
    """Process A: owns the buffer, drives the browser."""
    buf = FrameBuffer(NAME, WIDTH, HEIGHT, attach=False)
    print(f"[renderer] Buffer '{NAME}' created. Launching browser…")

    async with BrowserBridge(buf, width=WIDTH, height=HEIGHT) as br:
        await br.navigate(url)
        print(f"[renderer] Rendering {url} — press Ctrl-C to stop")
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
    
    buf.close()
    buf.unlink()
    print("[renderer] Stopped.")


async def run_viewer() -> None:
    """Process B: attaches to the buffer and opens a display window."""
    buf = FrameBuffer(NAME, WIDTH, HEIGHT, attach=True)
    print(f"[viewer] Attached to buffer '{NAME}'. Opening window…")
    display = DisplayClient(buf, title=f"wbb viewer — {NAME}")
    await display.run_async()
    buf.close()


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "renderer":
        url = sys.argv[2] if len(sys.argv) > 2 else "https://example.com"
        try:
            asyncio.run(run_renderer(url))
        except KeyboardInterrupt:
            pass
    elif mode == "viewer":
        asyncio.run(run_viewer())
    else:
        print(f"Unknown mode: {mode}. Use 'renderer' or 'viewer'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
