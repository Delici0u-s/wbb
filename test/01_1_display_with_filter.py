"""
01_display_with_filter.py — Display a page with a custom filter applied.

Demonstrates:
  - BrowserBridge + FrameBuffer lifecycle
  - DisplayClient with a filter pipeline
  - Composing built-in filters with a user-defined filter
"""

import asyncio
import numpy as np
from wbb import BrowserBridge, FrameBuffer, DisplayClient, filters

URL = "https://example.com"
WIDTH, HEIGHT = 1280, 720


def vignette(frame: np.ndarray) -> np.ndarray:
    """User-defined filter: radial darkening toward the edges."""
    h, w = frame.shape[:2]
    cx, cy = w / 2, h / 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
    mask = (1 - dist.clip(0, 1) ** 2)[..., np.newaxis].astype(np.float32)
    out = (frame.astype(np.float32) * mask).clip(0, 255).astype(np.uint8)
    out[..., 3] = frame[..., 3]  # preserve alpha
    return out


async def main() -> None:
    buf = FrameBuffer("ex03", WIDTH, HEIGHT)

    pipeline = [
        filters.colorize(r=0.95, g=0.95, b=1.1),  # slight cool tint
        vignette,
        filters.crop(0, 0, WIDTH, HEIGHT),
    ]

    async with BrowserBridge(buf, width=WIDTH, height=HEIGHT) as br:
        await br.navigate(URL)
        display = DisplayClient(buf, title="wbb — filtered view", filters=pipeline)
        await display.run_async()

    buf.close()
    buf.unlink()


if __name__ == "__main__":
    asyncio.run(main())
