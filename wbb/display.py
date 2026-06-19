"""
DisplayClient — an optional SDL2 window that consumes a FrameBuffer.

``pygame`` is a soft dependency. Importing this module succeeds even
without it; the error is deferred to ``DisplayClient.run()``.

Filter pipeline
---------------
A filter is any callable with signature ``(np.ndarray) -> np.ndarray``.
The pipeline is applied in order to each frame before it is blitted to
the SDL surface. Filters are run in the event loop (they should be fast);
offload heavy work with ``asyncio.run_in_executor`` inside a custom filter
wrapper if needed.

Input forwarding
----------------
``DisplayClient`` exposes a ``on_mouse_event`` and ``on_key_event``
callback slot. The caller decides whether and how to forward events to
``BrowserBridge``; the display layer is deliberately decoupled.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any, Optional

import numpy as np

from wbb.buffer import FrameBuffer

log = logging.getLogger(__name__)

Filter = Callable[[np.ndarray], np.ndarray]
MouseCallback = Callable[[str, int, int, int], Any]  # (event_type, x, y, button)
KeyCallback = Callable[[str, str], Any]  # (event_type, key_name)


class DisplayClient:
    """
    An SDL2 window that displays the contents of a :class:`FrameBuffer`.

    Parameters
    ----------
    buffer:
        The buffer to read frames from.
    title:
        Window title.
    filters:
        Ordered list of filter callables applied to each frame.
    on_mouse_event:
        Optional callback fired on mouse press/release/move inside the window.
    on_key_event:
        Optional callback fired on key press/release inside the window.
    """

    def __init__(
        self,
        buffer: FrameBuffer,
        *,
        title: str = "wbb",
        filters: Optional[list[Filter]] = None,
        on_mouse_event: Optional[MouseCallback] = None,
        on_key_event: Optional[KeyCallback] = None,
        window_size: Optional[tuple[int, int]] = None,
    ) -> None:
        self._buf = buffer
        self._title = title
        self._filters = filters or []
        self._on_mouse = on_mouse_event
        self._on_key = on_key_event
        self._stop_event = asyncio.Event()
        self._window_size = window_size

    def stop(self) -> None:
        """Signal the display loop to exit."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Blocking runner
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Open the SDL2 window and block until it is closed.

        Raises ``ImportError`` if ``pygame`` is not installed.
        """
        asyncio.run(self.run_async())

    # ------------------------------------------------------------------
    # Async runner — can be awaited from an existing event loop
    # ------------------------------------------------------------------

    async def run_async(self) -> None:
        """
        Open the SDL2 window and run until ``stop()`` is called or the
        window is closed by the user.

        Can be scheduled as a task alongside other coroutines::

            task = asyncio.create_task(display.run_async())
            ...
            display.stop()
            await task
        """
        try:
            import pygame  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("DisplayClient requires pygame: pip install wbb[display]") from exc

        pygame.init()
        # width, height = self._buf.width, self._buf.height
        width, height = self._window_size or (self._buf.width, self._buf.height)
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(self._title)
        clock = pygame.time.Clock()

        try:
            while not self._stop_event.is_set():
                # Process SDL events
                for evt in pygame.event.get():
                    if evt.type == pygame.QUIT:
                        return
                    self._dispatch_sdl_event(evt)

                # Non-blocking frame read
                frame = self._buf.read()
                arr = frame.data

                # Apply filter pipeline
                for f in self._filters:
                    arr = f(arr)

                # Blit: pygame expects (W, H, C) surface from RGB(A)
                # pygame.surfarray uses (W, H, 3) for RGB
                rgb = arr[:, :, :3]  # drop alpha channel
                surface = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
                screen.blit(surface, (0, 0))
                pygame.display.flip()

                # Yield to the event loop so other tasks can run
                await asyncio.sleep(0)
                clock.tick(60)
        finally:
            pygame.quit()

    def _dispatch_sdl_event(self, evt: Any) -> None:
        import pygame  # noqa: PLC0415

        if evt.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            kind = "down" if evt.type == pygame.MOUSEBUTTONDOWN else "up"
            if self._on_mouse:
                self._on_mouse(kind, evt.pos[0], evt.pos[1], evt.button)
        elif evt.type == pygame.MOUSEMOTION:
            if self._on_mouse:
                self._on_mouse("move", evt.pos[0], evt.pos[1], 0)
        elif evt.type in (pygame.KEYDOWN, pygame.KEYUP):
            kind = "down" if evt.type == pygame.KEYDOWN else "up"
            if self._on_key:
                self._on_key(kind, pygame.key.name(evt.key))
