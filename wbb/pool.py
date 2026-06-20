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
            br._buf = buffer  # or replace with br.set_buffer(buffer) if you add that method

            # Keep the existing Chrome process warm, but make sure the
            # page is back to a clean state.
            await br.navigate("about:blank")

            # Optional: if width/height changed between uses, update viewport.
            if "width" in kwargs or "height" in kwargs:
                width = kwargs.get("width", br._width)
                height = kwargs.get("height", br._height)
                await br.set_viewport(width, height)

            return br

        br = BrowserBridge(buffer, **kwargs)
        await br.start()
        return br

    async def release(self, br: BrowserBridge) -> None:
        if len(self._idle) < self._max_idle:
            await br.navigate("about:blank")
            self._idle.append(br)
        else:
            await br.stop()


BrowserPool = ChromePool
