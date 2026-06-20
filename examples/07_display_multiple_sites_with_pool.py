"""
08_display_multiple_sites_with_pool_composited.py — Render N websites in one window.

Demonstrates:
  - BrowserPool reuse across multiple BrowserBridge instances
  - N independent browser buffers
  - A compositor that writes all site frames into one larger buffer
  - DisplayClient rendering only the composed buffer
"""

from __future__ import annotations

import asyncio
import math
from contextlib import suppress

import numpy as np

from wbb import BrowserPool, DisplayClient, FrameBuffer

URLS: tuple[str, ...] = (
    "https://example.com",
    "https://www.python.org",
    "https://github.com",
    "https://www.clocktab.com/",
    "https://animate.style/",
)

SITES_AMOUNT = len(URLS)

SITE_WIDTH = 640
SITE_HEIGHT = 360
GAP = 16
PADDING = 16

MAX_IDLE_BROWSERS = min(3, SITES_AMOUNT)
COMPOSE_FPS = 30


def choose_layout(n: int, tile_w: int, tile_h: int) -> tuple[int, int]:
    """
    Return (cols, rows) for n tiles.

    Heuristic:
      - try every possible grid
      - penalize empty cells
      - prefer overall aspect ratios that make sense for landscape tiles
      - bias slightly toward wider layouts
    """
    if n <= 0:
        raise ValueError("n must be >= 1")

    tile_aspect = tile_w / tile_h
    target_aspect = tile_aspect * min(2.0, max(1.0, n / 4.0))

    best_score = float("inf")
    best = (n, 1)

    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)

        used = rows * cols
        empty = used - n

        grid_w = cols * tile_w + (cols - 1) * GAP
        grid_h = rows * tile_h + (rows - 1) * GAP
        grid_aspect = grid_w / grid_h

        # Lower is better.
        score = (
            empty * 1.0
            + abs(math.log(grid_aspect / target_aspect))
            + (0.05 if cols < rows else 0.0)
        )

        if score < best_score:
            best_score = score
            best = (cols, rows)

    return best


def cell_origin(col: int, row: int) -> tuple[int, int]:
    x = PADDING + col * (SITE_WIDTH + GAP)
    y = PADDING + row * (SITE_HEIGHT + GAP)
    return x, y


async def compose_sites(
    site_buffers: list[FrameBuffer],
    display_buffer: FrameBuffer,
    cols: int,
    rows: int,
) -> None:
    out_w = PADDING * 2 + cols * SITE_WIDTH + (cols - 1) * GAP
    out_h = PADDING * 2 + rows * SITE_HEIGHT + (rows - 1) * GAP

    canvas = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    canvas[..., :3] = 24
    canvas[..., 3] = 255

    try:
        while True:
            canvas[..., :3] = 24
            canvas[..., 3] = 255

            for i, buf in enumerate(site_buffers):
                col = i % cols
                row = i // cols
                if row >= rows:
                    break

                x, y = cell_origin(col, row)
                frame = buf.read()
                canvas[y : y + SITE_HEIGHT, x : x + SITE_WIDTH] = frame.data

            display_buffer.write(canvas)
            await asyncio.sleep(1 / COMPOSE_FPS)
    except asyncio.CancelledError:
        raise


async def main() -> None:
    cols, rows = choose_layout(SITES_AMOUNT, SITE_WIDTH, SITE_HEIGHT)

    out_w = PADDING * 2 + cols * SITE_WIDTH + (cols - 1) * GAP
    out_h = PADDING * 2 + rows * SITE_HEIGHT + (rows - 1) * GAP

    pool = BrowserPool(max_idle=MAX_IDLE_BROWSERS)

    site_buffers = [
        FrameBuffer(f"site_{i:02d}", SITE_WIDTH, SITE_HEIGHT) for i in range(SITES_AMOUNT)
    ]
    display_buffer = FrameBuffer("dashboard", out_w, out_h)

    bridges = []
    try:
        for buf, url in zip(site_buffers, URLS, strict=True):
            br = await pool.acquire(buffer=buf, width=SITE_WIDTH, height=SITE_HEIGHT)
            bridges.append(br)

        await asyncio.gather(*(br.navigate(url) for br, url in zip(bridges, URLS, strict=True)))

        display = DisplayClient(
            display_buffer,
            title=f"wbb — {SITES_AMOUNT}-site dashboard",
            window_size=(out_w, out_h),
        )

        compose_task = asyncio.create_task(compose_sites(site_buffers, display_buffer, cols, rows))
        display_task = asyncio.create_task(display.run_async())

        try:
            await display_task
        finally:
            compose_task.cancel()
            with suppress(asyncio.CancelledError):
                await compose_task
            display.stop()

    finally:
        for br in bridges:
            await pool.release(br)

        for buf in site_buffers:
            buf.close()
            buf.unlink()

        display_buffer.close()
        display_buffer.unlink()


if __name__ == "__main__":
    asyncio.run(main())
