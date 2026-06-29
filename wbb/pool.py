"""
ChromePool — reuse warm Chrome processes across BrowserBridge instances.

What pooling saves, and what it does not
----------------------------------------
**Saves:** the dominant cost of a fresh bridge is launching the Chrome
process and completing the CDP attach/enable handshake (hundreds of ms).
Keeping the process warm and re-pointing it at a new buffer skips that.

**Does NOT save — and the reset logic below exists because of this:**

* *Session/screencast state.* A released bridge still has a live
  screencast running against whatever it last rendered. Naively handing
  it back out leaves it pushing frames into the *previous* buffer (or a
  blank page) and firing the previous owner's hooks. ``release()`` now
  pauses the screencast and clears hooks; ``acquire()`` rebinds the
  buffer, re-applies the viewport, and restarts the screencast against
  the new consumer.

* *Renderer memory.* Chrome does not shrink its renderer heap back down
  after a heavy page — a process that once rendered a 4K-image-laden
  site keeps that high-water-mark RSS for its whole life. So a pool of
  ``max_idle`` *warm* processes is also a pool of ``max_idle`` *peak-RSS*
  processes. ``about:blank`` frees DOM/JS references but not the
  allocator's retained pages. If memory matters more than launch
  latency for your workload, use a smaller ``max_idle`` (or 0 to disable
  reuse). There is no way to reclaim that memory short of restarting the
  process, which is exactly what pooling is avoiding.

* *Unbounded concurrent acquires.* ``acquire`` is reuse-or-spawn: it
  does not cap how many live bridges exist at once, only how many *idle*
  ones are retained. A burst of N concurrent acquires spawns up to N
  Chromes. ``max_idle`` bounds the resting footprint, not the peak.
"""

from __future__ import annotations

from wbb import BrowserBridge, FrameBuffer


class ChromePool:
    """Reuses warm Chrome processes across BrowserBridge instances."""

    def __init__(self, max_idle: int = 3) -> None:
        self._idle: list[BrowserBridge] = []
        self._max_idle = max_idle

    async def acquire(self, buffer: FrameBuffer, **kwargs) -> BrowserBridge:
        if self._idle:
            br = self._idle.pop()

            # Rebind the bridge to the new buffer before reuse.
            br._buf = buffer

            # Apply requested viewport (if any) BEFORE restarting the
            # screencast, so the screencast's maxWidth/maxHeight and the
            # device-metrics override agree on the new size.
            if "width" in kwargs or "height" in kwargs:
                width = kwargs.get("width", br._width)
                height = kwargs.get("height", br._height)
                await br.set_viewport(width, height)

            # Clean page state, then restart the screencast so the new
            # consumer actually receives frames (the idle bridge had its
            # screencast paused by release()).
            await br.navigate("about:blank")
            await br._restart_screencast()

            return br

        br = BrowserBridge(buffer, **kwargs)
        await br.start()
        return br

    async def release(self, br: BrowserBridge) -> None:
        if len(self._idle) < self._max_idle:
            # Reset session state so a reused bridge doesn't leak the
            # previous owner's hooks or keep decoding frames into a
            # now-detached buffer while it sits idle:
            #   * pause the screencast (no point decoding about:blank),
            #   * clear event hooks,
            #   * return the page to a clean state.
            br._clear_hooks()
            from contextlib import suppress

            with suppress(Exception):
                await br._send("Page.stopScreencast")
            await br.navigate("about:blank")
            self._idle.append(br)
        else:
            await br.stop()


BrowserPool = ChromePool
