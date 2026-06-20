"""
BrowserBridge — owns the headless Chrome process and its CDP connection.

Frame acquisition model
-----------------------
Chrome's ``Page.startScreencast`` pushes JPEG frames over the CDP
WebSocket as they are rendered — no polling, no screenshot RPC round-
trips. Each push is decoded to RGBA and written into the FrameBuffer.

Thread model
------------
* The main public API is ``async``; callers drive it from an event loop.
* Internally a single ``asyncio`` task drives the WebSocket connection.
* FrameBuffer writes happen in a thread-pool executor so they do not
  block the event loop (numpy decode is CPU-bound).

Chrome detection
----------------
The library checks common install paths and the PATH for ``google-chrome``,
``chromium``, ``chromium-browser``, and ``chrome``. The first match wins.
Override with ``CHROME_PATH`` env var.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import suppress
from pathlib import Path
from typing import Any, Optional

import numpy as np
import websockets
from PIL import Image

from wbb.buffer import FrameBuffer
from wbb.frame import Frame

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chrome detection
# ---------------------------------------------------------------------------

_CANDIDATES = [
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "chrome",
]

_PLATFORM_PATHS: dict[str, list[str]] = {
    "darwin": [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ],
    "win32": [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Chromium\Application\chrome.exe",
    ],
}


def _find_chrome() -> str:
    if env := os.environ.get("CHROME_PATH"):
        return env
    for name in _CANDIDATES:
        if found := shutil.which(name):
            return found
    for path in _PLATFORM_PATHS.get(sys.platform, []):
        if Path(path).exists():
            return path
    raise FileNotFoundError(
        "Chrome/Chromium not found. Install it or set the CHROME_PATH environment variable."
    )


# ---------------------------------------------------------------------------
# Event hook registry
# ---------------------------------------------------------------------------

EventCallback = Callable[..., Coroutine[Any, Any, None] | None]


class _HookRegistry:
    def __init__(self) -> None:
        self._hooks: dict[str, list[EventCallback]] = {}

    def register(self, event: str, cb: EventCallback) -> None:
        self._hooks.setdefault(event, []).append(cb)

    async def fire(self, event: str, **kwargs: Any) -> None:
        for cb in self._hooks.get(event, []):
            result = cb(**kwargs)
            if asyncio.iscoroutine(result):
                await result


# ---------------------------------------------------------------------------
# BrowserBridge
# ---------------------------------------------------------------------------


class BrowserBridge:
    """
    Owns a headless Chrome instance and exposes it as an async Python API.

    Parameters
    ----------
    buffer:
        The :class:`FrameBuffer` that receives rendered frames.
    width, height:
        Viewport dimensions.
    screencast_quality:
        JPEG quality (1–100) for the CDP screencast stream.
    screencast_max_fps:
        Maximum frames per second to request from Chrome.
    enable_input:
        If False, all ``click`` / ``key`` / ``type`` methods raise.
        Setting this explicitly to True enables input; False disables
        it; None (default) enables it.
    headless_args:
        Extra Chrome command-line arguments.
    """

    def __init__(
        self,
        buffer: FrameBuffer,
        *,
        width: int = 1280,
        height: int = 720,
        screencast_quality: int = 80,
        screencast_max_fps: int = 30,
        enable_input: Optional[bool] = None,
        headless_args: Optional[list[str]] = None,
    ) -> None:
        self._buf = buffer
        self._width = width
        self._height = height
        self._sq = screencast_quality
        self._sfps = screencast_max_fps
        self._input_enabled = enable_input is not False
        self._extra_args = headless_args or []

        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._ws: Optional[websockets.ClientConnection] = None
        self._cmd_id = 0
        self._session_id: Optional[str] = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._hooks = _HookRegistry()
        self._recv_task: Optional[asyncio.Task[None]] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch Chrome and connect the CDP WebSocket."""
        chrome = _find_chrome()
        args = [
            chrome,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            f"--window-size={self._width},{self._height}",
            "--remote-debugging-port=0",  # OS-assigned port
            "about:blank",
            *self._extra_args,
        ]
        log.debug("Launching: %s", " ".join(args))
        self._proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for Chrome to print its DevTools URL
        ws_url = await asyncio.get_event_loop().run_in_executor(None, self._read_devtools_url)
        log.debug("DevTools URL: %s", ws_url)

        self._ws = await websockets.connect(ws_url, max_size=None)
        self._running = True
        self._recv_task = asyncio.create_task(self._recv_loop(), name="cdp-recv")

        # Discover the page target Chrome created for "about:blank" and
        # attach to it with a flattened session, so Page.*/Runtime.*/Input.*
        # commands have somewhere to land.
        targets = await self._send("Target.getTargets")
        page_target = next(t for t in targets["targetInfos"] if t["type"] == "page")
        attach_result = await self._send(
            "Target.attachToTarget",
            targetId=page_target["targetId"],
            flatten=True,
        )
        self._session_id = attach_result["sessionId"]

        # Enable domains
        await self._send("Page.enable")
        await self._send("Runtime.enable")
        await self._send("Network.enable")
        await self._send("Target.setDiscoverTargets", discover=True)

        # Set viewport
        await self._send(
            "Emulation.setDeviceMetricsOverride",
            width=self._width,
            height=self._height,
            deviceScaleFactor=1,
            mobile=False,
        )

        # Start screencast
        await self._send(
            "Page.startScreencast",
            format="jpeg",
            quality=self._sq,
            maxWidth=self._width,
            maxHeight=self._height,
            everyNthFrame=1,
        )

    def _read_devtools_url(self) -> str:
        """Parse Chrome stderr for 'DevTools listening on ws://...'"""
        assert self._proc and self._proc.stderr
        for line in self._proc.stderr:
            decoded = line.decode(errors="replace")
            if "DevTools listening on" in decoded:
                return decoded.split("DevTools listening on")[-1].strip()
        raise RuntimeError("Chrome did not emit a DevTools URL")

    async def stop(self) -> None:
        """Stop the screencast, close the WebSocket, terminate Chrome."""
        if not self._running:
            return
        self._running = False
        with suppress(Exception):
            await self._send("Page.stopScreencast")
        if self._recv_task:
            self._recv_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._recv_task
        if self._ws:
            await self._ws.close()
        if self._proc:
            self._proc.terminate()
            loop = asyncio.get_event_loop()
            try:
                await asyncio.wait_for(loop.run_in_executor(None, self._proc.wait), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()
                # Still need to reap the zombie, but don't let a stuck
                # kill() hang shutdown forever — cap it too.
                with suppress(Exception):
                    await asyncio.wait_for(loop.run_in_executor(None, self._proc.wait), timeout=2)

    # async def stop(self) -> None:
    #     """Stop the screencast, close the WebSocket, terminate Chrome."""
    #     if not self._running:
    #         return
    #     self._running = False
    #     with suppress(Exception):
    #         await self._send("Page.stopScreencast")
    #     if self._recv_task:
    #         self._recv_task.cancel()
    #         with suppress(asyncio.CancelledError):
    #             await self._recv_task
    #     if self._ws:
    #         await self._ws.close()
    #     if self._proc:
    #         self._proc.terminate()
    #         try:
    #             self._proc.wait(timeout=5)
    #         except subprocess.TimeoutExpired:
    #             self._proc.kill()

    async def __aenter__(self) -> "BrowserBridge":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # CDP plumbing
    # ------------------------------------------------------------------

    async def _send(self, method: str, **params: Any) -> Any:
        self._cmd_id += 1
        cid = self._cmd_id
        payload: dict[str, Any] = {"id": cid, "method": method, "params": params}
        if self._session_id is not None:
            payload["sessionId"] = self._session_id
        msg = json.dumps(payload)
        fut: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        self._pending[cid] = fut
        assert self._ws
        await self._ws.send(msg)
        return await fut

    # async def _send(self, method: str, **params: Any) -> Any:
    #     self._cmd_id += 1
    #     cid = self._cmd_id
    #     msg = json.dumps({"id": cid, "method": method, "params": params})
    #     fut: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
    #     self._pending[cid] = fut
    #     assert self._ws
    #     await self._ws.send(msg)
    #     return await fut

    async def _recv_loop(self) -> None:
        assert self._ws
        async for raw in self._ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # debug test
            # a_s = msg.get("id")
            # a = msg.get("method", f"<response id={a_s}>")
            # print(f"[CDP] {a}", file=sys.stderr)  # TEMP

            if "id" in msg:
                fut = self._pending.pop(msg["id"], None)
                if fut and not fut.done():
                    if "error" in msg:
                        fut.set_exception(RuntimeError(msg["error"].get("message", "CDP error")))
                    else:
                        fut.set_result(msg.get("result", {}))
            elif "method" in msg:
                asyncio.create_task(self._dispatch_event(msg["method"], msg.get("params", {})))

    async def _dispatch_event(self, method: str, params: dict[str, Any]) -> None:
        if method == "Page.screencastFrame":
            await self._on_screencast_frame(params)
        elif method == "Page.navigatedWithinDocument" or method == "Page.frameNavigated":
            await self._hooks.fire("navigate", url=params.get("url", ""))
        elif method == "Page.loadEventFired":
            await self._hooks.fire("load")
        elif method == "Runtime.consoleAPICalled":
            args = [a.get("value", "") for a in params.get("args", [])]
            await self._hooks.fire("console", level=params.get("type", "log"), args=args)
        elif method == "Runtime.exceptionThrown":
            desc = params.get("exceptionDetails", {}).get("text", "")
            await self._hooks.fire("error", description=desc)

    async def _on_screencast_frame(self, params: dict[str, Any]) -> None:
        # Ack immediately so Chrome keeps pushing
        session_id = params.get("sessionId", 0)
        asyncio.create_task(self._send("Page.screencastFrameAck", sessionId=session_id))

        data = params.get("data", "")
        ts = params.get("metadata", {}).get("timestamp", time.monotonic())

        loop = asyncio.get_running_loop()
        rgba = await loop.run_in_executor(None, _jpeg_to_rgba, data, self._width, self._height)
        fid = self._buf.write(rgba)
        frame = Frame(data=rgba, width=self._width, height=self._height, frame_id=fid, timestamp=ts)
        await self._hooks.fire("frame", frame=frame)

    # ------------------------------------------------------------------
    # Navigation & page control
    # ------------------------------------------------------------------

    # async def navigate(
    #     self, url: str, *, wait_for_load: bool = True, timeout: float = 30.0
    # ) -> None:
    #     """Navigate to *url*. By default, blocks until the page's load event fires."""
    #     if wait_for_load:
    #         evt = asyncio.Event()
    #         self.on("load", lambda: evt.set())
    #         await self._send("Page.navigate", url=url)
    #         try:
    #             await asyncio.wait_for(evt.wait(), timeout=timeout)
    #         except asyncio.TimeoutError:
    #             log.warning("navigate() timed out waiting for load after %.1fs", timeout)
    #     else:
    #         await self._send("Page.navigate", url=url)

    async def navigate(self, url: str) -> None:
        """Navigate to *url* and wait for the page to stop loading."""
        await self._send("Page.navigate", url=url)

    async def reload(self, *, ignore_cache: bool = False) -> None:
        await self._send("Page.reload", ignoreCache=ignore_cache)

    async def eval(self, expression: str, *, await_promise: bool = False) -> Any:
        """Evaluate *expression* in the page context and return the result."""
        result = await self._send(
            "Runtime.evaluate",
            expression=expression,
            awaitPromise=await_promise,
            returnByValue=True,
        )
        return result.get("result", {}).get("value")

    async def wait_for_selector(
        self, selector: str, *, timeout: float = 10.0, poll_interval: float = 0.15
    ) -> bool:
        """Poll the DOM until *selector* matches at least one element."""
        elapsed = 0.0
        while elapsed < timeout:
            found = await self.eval(f"document.querySelector({selector!r}) !== null")
            if found:
                return True
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        return False

    # async def wait_for_load(self, timeout: float = 30.0) -> None:
    #     """Block until the page fires its load event."""
    #     evt: asyncio.Event = asyncio.Event()
    #     self.on("load", lambda: evt.set())
    #     try:
    #         await asyncio.wait_for(evt.wait(), timeout=timeout)
    #     except asyncio.TimeoutError:
    #         log.warning("wait_for_load timed out after %.1fs", timeout)

    async def set_viewport(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        await self._send(
            "Emulation.setDeviceMetricsOverride",
            width=width,
            height=height,
            deviceScaleFactor=1,
            mobile=False,
        )

    # ------------------------------------------------------------------
    # Input (all raise if enable_input=False)
    # ------------------------------------------------------------------

    def _require_input(self) -> None:
        if not self._input_enabled:
            raise RuntimeError("Input is disabled (enable_input=False)")

    async def click(self, x: float, y: float, *, button: str = "left") -> None:
        self._require_input()
        for kind in ("mousePressed", "mouseReleased"):
            await self._send(
                "Input.dispatchMouseEvent",
                type=kind,
                x=x,
                y=y,
                button=button,
                clickCount=1,
            )

    async def click_element(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[str] = None,
        nth: int = 0,
        scroll_into_view: bool = False,
        timeout: float = 5.0,
        poll_interval: float = 0.15,
        button: str = "left",
    ) -> bool:
        """
        Click an element identified by *selector* and/or *text*, instead
        of a hardcoded pixel position.

        Resolution order
        -----------------
        1. If *selector* is given, ``document.querySelectorAll(selector)``
           is evaluated. If it yields at least one match, *nth* of those
           matches is used directly (selector matches are NOT redirected
           to a clickable ancestor — if you wrote the selector, you meant
           that element).
        2. Otherwise — or if *selector* matched nothing — *text* is used
           as a case-insensitive substring match against every element's
           *own* text content (direct text-node children only, not
           descendants' concatenated text). Each match is then resolved
           to its nearest clickable ancestor (``a``, ``button``, ``input``,
           ``select``, ``textarea``, ``[role="button"]``, ``[onclick]``,
           ``summary``, ``label``, or the matched element itself if it's
           already one of those) and the result list is de-duplicated.
           This matters in practice: matching text like "Login" inside
           ``<button><b>Login</b></button>`` should click the button, not
           the inert ``<b>`` wrapping it — verified against jsdom, see
           module docstring.
        3. If neither resolves to an element within *timeout* seconds,
           returns False without raising.

        Coordinates
        -----------
        The resolved element's ``getBoundingClientRect()`` center is used
        as the click point, passed straight to ``self.click()`` — same
        Input.dispatchMouseEvent path as a manual hardcoded-coordinate
        click.

        Parameters
        ----------
        selector:
            CSS selector, tried first.
        text:
            Substring to match against element text content; fallback if
            *selector* is None or matches nothing.
        nth:
            Index into the match list if more than one element matches
            (default: first match, 0).
        scroll_into_view:
            If True, calls ``element.scrollIntoView()`` and re-reads the
            bounding box before clicking. Default False, per your call —
            this does not scroll unless asked, since scrolling can shift
            unrelated layout and cause a different mis-click.
        timeout, poll_interval:
            Same semantics as ``wait_for_selector`` — poll until the
            element appears or *timeout* elapses.
        button:
            Forwarded to ``self.click()``.

        Returns
        -------
        bool
            True if an element was found and clicked, False on timeout.

        Raises
        ------
        RuntimeError
            If input is disabled — raised by the underlying ``click()``
            call's ``self._require_input()`` check.
        ValueError
            If neither *selector* nor *text* is given.
        """
        if selector is None and text is None:
            raise ValueError("click_element requires at least one of selector or text")

        finder_js = _build_finder_js(selector, text, nth)

        elapsed = 0.0
        while True:
            box = await self.eval(finder_js)
            if box is not None:
                break
            if elapsed >= timeout:
                return False
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        if scroll_into_view:
            scroll_js = _build_finder_js(selector, text, nth, scroll_into_view=True)
            box = await self.eval(scroll_js)
            if box is None:
                # Element disappeared between the find and the scroll —
                # treat as not found rather than clicking stale coords.
                return False

        cx = box["x"] + box["width"] / 2
        cy = box["y"] + box["height"] / 2
        await self.click(cx, cy, button=button)
        return True

    async def move(self, x: float, y: float) -> None:
        self._require_input()
        await self._send("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)

    async def scroll(self, x: float, y: float, delta_x: float = 0, delta_y: float = 100) -> None:
        self._require_input()
        await self._send(
            "Input.dispatchMouseEvent",
            type="mouseWheel",
            x=x,
            y=y,
            deltaX=delta_x,
            deltaY=delta_y,
        )

    async def key(
        self,
        key: str,
        *,
        kind: str = "keyDown",
        text: Optional[str] = None,
    ) -> None:
        """
        Dispatch a key event. *key* is a DOM key value (e.g. 'Enter', 'a').
        NOTICE: This function is a bit wonkey and doesnt always work correctly, or as Expected!!

        *text* is the character CDP should report as generated by this
        keypress — required for Chrome to actually insert a character into
        a focused editable field. If omitted, defaults to *key* itself when
        *key* is a single printable character, and to "" otherwise (e.g.
        for 'Enter', 'Tab', 'Backspace', which shouldn't insert text).
        """
        self._require_input()
        if text is None:
            text = key if len(key) == 1 else ""
        await self._send(
            "Input.dispatchKeyEvent",
            type=kind,
            key=key,
            text=text if kind == "keyDown" else "",  # keyUp shouldn't carry text
        )

    # async def key(self, key: str, *, kind: str = "keyDown") -> None:
    #     """Dispatch a raw key event. *key* is a DOM key value (e.g. 'Enter')."""
    #     self._require_input()
    #     await self._send("Input.dispatchKeyEvent", type=kind, key=key)

    async def type(self, text: str) -> None:
        """Insert *text* as if typed by the user."""
        self._require_input()
        await self._send("Input.insertText", text=text)

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    @property
    def buffer(self) -> FrameBuffer:
        return self._buf

    async def frames(self) -> AsyncIterator[Frame]:
        """Async iterator of frames as they arrive from the screencast."""
        async for frame in self._buf:
            yield frame

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------

    def on(self, event: str, cb: EventCallback) -> None:
        """
        Register *cb* for *event*.

        Supported events: ``"frame"``, ``"navigate"``, ``"load"``,
        ``"console"``, ``"error"``.

        *cb* may be a plain function or a coroutine function; both work.
        """
        self._hooks.register(event, cb)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jpeg_to_rgba(data_b64: str, width: int, height: int) -> np.ndarray:
    """Decode a base64 JPEG payload to an H×W×4 RGBA uint8 array."""
    raw = base64.b64decode(data_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)  # type: ignore[attr-defined]
    return np.asarray(img, dtype=np.uint8)


# Elements considered "clickable" when resolving a text match to its
# nearest interactive ancestor. Selector matches skip this entirely.
_CLICKABLE_SEL = 'a,button,input,select,textarea,[role="button"],[onclick],summary,label'


def _build_finder_js(
    selector: Optional[str],
    text: Optional[str],
    nth: int,
    *,
    scroll_into_view: bool = False,
) -> str:
    """
    Build a JS expression for Runtime.evaluate that resolves an element by
    selector (preferred) or text substring (fallback) and returns its
    bounding box as ``{x, y, width, height}`` (viewport-relative — the
    coordinate space `Input.dispatchMouseEvent` expects), or ``null`` if
    nothing resolves.

    *selector* and *text* are JSON-encoded into the expression so
    arbitrary strings (quotes, backslashes, unicode) embed safely —
    `json.dumps` rather than Python `repr`, since this is JS, not Python
    (the existing `wait_for_selector` uses `repr` because it calls
    `querySelector` directly with a Python-side f-string; this builds a
    bigger expression so a JSON literal is cleaner).
    """
    sel_json = json.dumps(selector) if selector is not None else "null"
    text_json = json.dumps(text) if text is not None else "null"
    scroll_json = "true" if scroll_into_view else "false"

    return f"""
(() => {{
    const sel = {sel_json};
    const textNeedle = {text_json};
    const nth = {nth};
    const doScroll = {scroll_json};
    const CLICKABLE_SEL = {json.dumps(_CLICKABLE_SEL)};

    function nearestClickable(el) {{
        if (el.matches && el.matches(CLICKABLE_SEL)) return el;
        const anc = el.closest ? el.closest(CLICKABLE_SEL) : null;
        return anc || el;
    }}

    let matches = [];
    if (sel !== null) {{
        matches = Array.from(document.querySelectorAll(sel));
    }}
    if (matches.length === 0 && textNeedle !== null) {{
        const needle = textNeedle.toLowerCase();
        matches = Array.from(document.querySelectorAll('*')).filter(el => {{
            let own = '';
            for (const node of el.childNodes) {{
                if (node.nodeType === Node.TEXT_NODE) own += node.textContent;
            }}
            return own.toLowerCase().includes(needle);
        }});
        matches = matches.map(nearestClickable);
        matches = Array.from(new Set(matches));
    }}

    const el = matches[nth];
    if (!el) return null;

    if (doScroll) {{
        el.scrollIntoView({{ block: 'center', inline: 'center' }});
    }}

    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) return null;  // detached/hidden
    return {{ x: r.x, y: r.y, width: r.width, height: r.height }};
}})()
"""
