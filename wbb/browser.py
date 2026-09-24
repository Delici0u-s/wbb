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
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import suppress
from pathlib import Path
from typing import Any, Optional, Union, overload

import websockets

from wbb import _compat
from wbb.buffer import FrameBuffer
from wbb.codec import decode_frame, turbojpeg_available
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
}


def _windows_paths() -> tuple[list[Path], list[Path]]:
    """(chrome_like, edge) install locations on Windows.

    Built from the environment rather than hard-coded drive letters:
    Chrome's default *per-user* install (no admin rights) lives under
    %LOCALAPPDATA%, which a fixed Program Files list never finds,
    and Program Files is not always on C:.
    """
    roots = [
        os.environ.get(var)
        for var in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")
    ]
    bases = [Path(r) for r in roots if r]
    chrome = [
        b / sub / "Application" / "chrome.exe"
        for b in bases
        for sub in (Path("Google/Chrome"), Path("Chromium"))
    ]
    edge = [b / "Microsoft/Edge/Application/msedge.exe" for b in bases]
    return chrome, edge


def _windows_app_path(exe: str) -> Optional[str]:
    """Look `exe` up in the registry's App Paths, where installers register it.

    Catches installs in non-default locations that neither PATH nor the
    fixed directories cover.
    """
    try:
        import winreg  # noqa: PLC0415 — Windows-only stdlib module
    except ImportError:
        return None
    key = rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe}"
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, key) as k:
                value, _ = winreg.QueryValueEx(k, None)
        except OSError:
            continue
        if value and Path(value).exists():
            return str(value)
    return None


def _find_chrome() -> str:
    """Locate a Chromium-based browser that speaks CDP.

    Order: CHROME_PATH, then Chrome/Chromium (PATH, platform install
    dirs, Windows registry), and on Windows Microsoft Edge last. Edge is
    Chromium-based and ships with every Windows 10/11 install, so it is
    what makes wbb work out of the box there; it is only picked when no
    Chrome/Chromium exists, and it is logged so the choice is visible.
    Set CHROME_PATH to force a specific browser.
    """
    if env := os.environ.get("CHROME_PATH"):
        return env
    checked: list[str] = []
    for name in _CANDIDATES:
        if found := shutil.which(name):
            return found
    checked.append(f"PATH names {_CANDIDATES}")

    for path in _PLATFORM_PATHS.get(sys.platform, []):
        if Path(path).exists():
            return path
        checked.append(path)

    if sys.platform == "win32":
        chrome_paths, edge_paths = _windows_paths()
        for p in chrome_paths:
            if p.exists():
                return str(p)
            checked.append(str(p))
        if found := _windows_app_path("chrome.exe"):
            return found
        checked.append(r"registry App Paths\chrome.exe")

        edge = shutil.which("msedge") or _windows_app_path("msedge.exe")
        if edge is None:
            edge = next((str(p) for p in edge_paths if p.exists()), None)
        if edge is not None:
            log.info(
                "Chrome/Chromium not found; using Microsoft Edge (%s). Set "
                "CHROME_PATH to use a different browser.",
                edge,
            )
            return edge
        checked.append("Microsoft Edge (PATH, registry, install dirs)")

    raise FileNotFoundError(
        "Chrome/Chromium not found. Install it or set the CHROME_PATH "
        "environment variable. Checked:\n  " + "\n  ".join(checked)
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
        # One raising callback must not take out the CDP recv loop, which
        # is what happened before: fire() is awaited inline from
        # _recv_loop, so an exception here killed the socket drain and
        # every pending _send() hung forever with no error anywhere.
        for cb in self._hooks.get(event, []):
            try:
                result = cb(**kwargs)
                if asyncio.iscoroutine(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Error in %r event hook %r", event, cb)


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
        Upper bound on frames per second. Enforced by pacing the
        screencast acknowledgements in wall-clock time, so a page that
        paints rarely still delivers every frame it produces. (Chrome's
        own ``everyNthFrame`` counts compositor frames instead, which
        silently starves static pages; it is only used on the legacy
        ``ack_after_write=False`` path.)
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
        extra_headers: Optional[dict[str, str]] = None,
        screencast_format: str = "jpeg",
        transparent_background: bool = False,
        ack_after_write: bool = True,
    ) -> None:
        if screencast_format not in ("jpeg", "png"):
            raise ValueError(f"screencast_format must be 'jpeg' or 'png', got {screencast_format!r}")
        self._buf = buffer
        self._width = width
        self._height = height
        self._sq = screencast_quality
        self._sfps = screencast_max_fps
        self._input_enabled = enable_input is not False
        self._extra_args = headless_args or []
        self._screencast_format = screencast_format
        self._transparent = transparent_background
        self._ack_after_write = ack_after_write and not _compat.LEGACY_ACK
        self._min_screencast_interval = (
            1.0 / screencast_max_fps if screencast_max_fps and self._ack_after_write else 0.0
        )
        self._last_ack_at = 0.0
        if transparent_background and screencast_format != "png":
            log.warning(
                "transparent_background=True with screencast_format=%r: JPEG has "
                "no alpha channel, so every decoded pixel will be opaque. Use "
                "screencast_format='png'.",
                screencast_format,
            )

        # Header groups: id -> {header_name: value, ...}. Re-merged and
        # pushed to CDP as a single Network.setExtraHTTPHeaders call on
        # every add/remove, since the protocol has no incremental API.
        self._header_groups: dict[str, dict[str, str]] = {}
        if extra_headers:
            self._header_groups["__initial__"] = dict(extra_headers)

        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._ws: Optional[websockets.ClientConnection] = None
        self._cmd_id = 0
        self._session_id: Optional[str] = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._hooks = _HookRegistry()
        self._recv_task: Optional[asyncio.Task[None]] = None
        self._bg_tasks: set[asyncio.Task[Any]] = set()
        self._running = False
        self._drainer: Optional[threading.Thread] = None

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
        ]
        # Note: --default-background-color= is deliberately NOT passed.
        # Its value format is undocumented, it is a Blink switch rather
        # than a browser one, and passing it correlated with Chrome
        # failing to publish a page target at all. The documented path is
        # Emulation.setDefaultBackgroundColorOverride, sent below once
        # the session is attached.
        args += ["about:blank", *self._extra_args]
        log.debug("Launching: %s", " ".join(args))
        self._proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for Chrome to print its DevTools URL
        ws_url = await asyncio.get_running_loop().run_in_executor(None, self._read_devtools_url)
        log.debug("DevTools URL: %s", ws_url)

        # From here on nobody reads Chrome's pipes. A 64 KiB pipe buffer
        # fills in seconds on a page that logs, and Chrome then blocks in
        # write() forever — the browser appears to freeze with no error.
        # Drain both pipes into the logger for the process's lifetime.
        self._start_pipe_drain()

        self._ws = await websockets.connect(ws_url, max_size=None)
        self._running = True
        self._recv_task = asyncio.create_task(self._recv_loop(), name="cdp-recv")

        # Discover the page target Chrome created for "about:blank" and
        # attach to it with a flattened session, so Page.*/Runtime.*/Input.*
        # commands have somewhere to land.
        page_target = await self._await_page_target()
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

        if self._header_groups:
            await self._push_headers()

        if self._transparent:
            # Documented for Page.captureScreenshot; whether the screencast
            # path honours it is NOT confirmed here — see the open questions.
            # It is a cheap call either way and does nothing harmful if the
            # screencast ignores it.
            await self._send(
                "Emulation.setDefaultBackgroundColorOverride",
                color={"r": 0, "g": 0, "b": 0, "a": 0},
            )

        # Set viewport
        await self._send(
            "Emulation.setDeviceMetricsOverride",
            width=self._width,
            height=self._height,
            deviceScaleFactor=1,
            mobile=False,
        )

        # Start screencast
        await self._start_screencast()

    async def _start_screencast(self) -> None:
        # everyNthFrame counts *compositor* frames, not wall-clock time,
        # and a page that stops animating stops producing them. With
        # screencast_max_fps=10 that becomes everyNthFrame=6, so a static
        # page which paints two or three times on load and then goes
        # quiet never reaches the sixth frame and the screencast delivers
        # nothing at all — the page looks like it never loaded.
        #
        # Where ack pacing is available (ack_after_write, the default)
        # ask for every frame and rate-limit by delaying the ack instead:
        # Chrome will not send the next frame until we ack, so the cap is
        # enforced in wall-clock time and a static page still delivers
        # the frames it does produce. everyNthFrame is only used on the
        # legacy pipelined path, where there is no ack to delay.
        every_nth = 1
        if not self._ack_after_write:
            every_nth = max(1, round(60 / self._sfps)) if self._sfps else 1
        params: dict[str, Any] = {
            "format": self._screencast_format,
            "maxWidth": self._width,
            "maxHeight": self._height,
            "everyNthFrame": every_nth,
        }
        if self._screencast_format == "jpeg":
            # quality is meaningless for PNG; CDP ignores it, but sending
            # it anyway makes the log confusing when debugging a
            # transparency problem.
            params["quality"] = self._sq
        if self._screencast_format == "png" and turbojpeg_available():
            log.debug(
                "screencast_format='png': the libjpeg-turbo fast path does not "
                "apply, decode goes through Pillow."
            )
        await self._send("Page.startScreencast", **params)

    async def _restart_screencast(self) -> None:
        """Stop and restart the screencast against the current page/viewport.

        Used by the pool on reuse: a bridge that was paused on
        about:blank needs its screencast re-pointed at the new viewport
        and re-started so the new consumer actually receives pushes.
        """
        with suppress(Exception):
            await self._send("Page.stopScreencast")
        await self._start_screencast()

    def _clear_hooks(self) -> None:
        """Drop all registered event callbacks.

        The pool calls this on release so a reused bridge doesn't keep
        firing the previous owner's ``frame``/``load``/etc. callbacks at
        the next, unrelated consumer.
        """
        self._hooks = _HookRegistry()

    async def _await_page_target(self, timeout: float = 10.0) -> dict[str, Any]:
        """Poll Target.getTargets until Chrome's initial page target exists.

        The page target for the startup URL is not always present the
        instant the DevTools socket accepts a connection — Chrome
        publishes the browser target first. A bare
        `next(t for t in ... if t["type"] == "page")` therefore raises
        StopIteration, which inside a coroutine surfaces as the useless
        `RuntimeError: coroutine raised StopIteration`.
        """
        deadline = time.monotonic() + timeout
        delay = 0.02
        last: list[dict[str, Any]] = []
        while True:
            targets = await self._send("Target.getTargets")
            last = targets.get("targetInfos", [])
            for t in last:
                if t.get("type") == "page":
                    return t
            if time.monotonic() >= deadline:
                kinds = sorted({t.get("type", "?") for t in last})
                raise RuntimeError(
                    f"Chrome exposed no page target within {timeout:.0f}s "
                    f"(saw: {', '.join(kinds) or 'nothing'}). Chrome may have "
                    f"rejected a command-line flag; run with logging at DEBUG "
                    f"to see its stderr."
                )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 0.25)

    def _start_pipe_drain(self) -> None:
        """Consume Chrome's stdout/stderr so its pipe buffers never fill."""

        def _drain() -> None:
            proc = self._proc
            if proc is None:
                return
            for stream, level in ((proc.stderr, logging.DEBUG), (proc.stdout, logging.DEBUG)):
                if stream is None:
                    continue
                try:
                    for line in stream:
                        log.log(level, "chrome: %s", line.decode(errors="replace").rstrip())
                except Exception:
                    return

        self._drainer = threading.Thread(
            target=_drain, name="wbb-chrome-drain", daemon=True
        )
        self._drainer.start()

    def _read_devtools_url(self) -> str:
        """Parse Chrome stderr for 'DevTools listening on ws://...'"""
        assert self._proc and self._proc.stderr
        for line in self._proc.stderr:
            decoded = line.decode(errors="replace")
            if "DevTools listening on" in decoded:
                return decoded.split("DevTools listening on")[-1].strip()
        raise RuntimeError("Chrome did not emit a DevTools URL")

    def _fail_pending(self, exc: BaseException) -> None:
        """Resolve every outstanding _send() future with *exc*.

        Without this a socket close (Chrome crashed, tab closed, stop()
        called) leaves every `await self._send(...)` parked forever with
        no exception and no timeout — the single worst failure mode in
        this file, because it looks like a hang, not an error.
        """
        pending, self._pending = self._pending, {}
        for fut in pending.values():
            if fut.done():
                continue
            fut.set_exception(exc)
            # Read it back so asyncio does not print "Future exception was
            # never retrieved" at GC time for a command whose awaiter was
            # already cancelled — the common shape on Ctrl-C, where the
            # traceback shown points confusingly at _recv_loop rather than
            # at whatever issued the command. An awaiter that is still
            # alive gets the exception exactly as before.
            fut.exception()
        if pending:
            log.debug("failed %d pending CDP command(s): %r", len(pending), exc)

    async def stop(self) -> None:
        """Stop the screencast, close the WebSocket, terminate Chrome.

        Safe to call after a failed start(): a Chrome process that was
        spawned before the failure is still terminated, where the old
        `if not self._running: return` guard leaked it.
        """
        was_running = self._running
        self._running = False
        if was_running:
            with suppress(Exception):
                await self._send("Page.stopScreencast")
        # Drain any in-flight screencast decode/write tasks so we don't
        # cancel a FrameBuffer write half-done. Bounded — screencast is
        # stopped, so no new ones arrive; give them a moment to finish.
        if self._bg_tasks:
            with suppress(Exception):
                await asyncio.wait(set(self._bg_tasks), timeout=2)
        if self._recv_task:
            self._recv_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._recv_task
        self._fail_pending(RuntimeError("BrowserBridge stopped"))
        if self._ws:
            await self._ws.close()
        if self._proc:
            self._proc.terminate()
            loop = asyncio.get_running_loop()
            try:
                await asyncio.wait_for(loop.run_in_executor(None, self._proc.wait), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()
                # Still need to reap the zombie, but don't let a stuck
                # kill() hang shutdown forever — cap it too.
                with suppress(Exception):
                    await asyncio.wait_for(loop.run_in_executor(None, self._proc.wait), timeout=2)

    async def __aenter__(self) -> "BrowserBridge":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # CDP plumbing
    # ------------------------------------------------------------------

    async def send_cdp(self, method: str, **params: Any) -> Any:
        """Send a raw CDP command on this bridge's session and await its result.

        The escape hatch for anything the typed API does not cover
        (Emulation.*, Animation.*, a newer CDP method than this release
        knows about). Same session, same id space, same error mapping as
        every internal call — previously the only way to do this was to
        reach into the private `_send`.

        Raises RuntimeError with Chrome's message on a CDP error, or if
        the connection closes while the command is outstanding.
        """
        if not self._running:
            raise RuntimeError("BrowserBridge is not running; call start() first")
        return await self._send(method, **params)

    async def _send(self, method: str, **params: Any) -> Any:
        self._cmd_id += 1
        cid = self._cmd_id
        payload: dict[str, Any] = {"id": cid, "method": method, "params": params}
        if self._session_id is not None:
            payload["sessionId"] = self._session_id
        msg = json.dumps(payload)
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[cid] = fut
        assert self._ws
        await self._ws.send(msg)
        return await fut

    async def _send_nowait(self, method: str, **params: Any) -> None:
        """Fire-and-forget CDP send for commands whose response we never
        consume (currently just Page.screencastFrameAck, sent once per
        frame). Unlike ``_send`` this allocates no Future and registers
        no pending entry — it just awaits the websocket write. The
        command still carries an ``id`` so Chrome's reply is well-formed;
        ``_recv_loop`` finds no pending Future for it and drops it.

        Awaited from inside the per-event dispatch task that's already
        running, so it adds no *second* task per frame the way the old
        ``create_task(self._send("...Ack"))`` did.
        """
        self._cmd_id += 1
        payload: dict[str, Any] = {
            "id": self._cmd_id,
            "method": method,
            "params": params,
        }
        if self._session_id is not None:
            payload["sessionId"] = self._session_id
        msg = json.dumps(payload)
        assert self._ws
        await self._ws.send(msg)

    async def _recv_loop(self) -> None:
        assert self._ws
        try:
            await self._recv_loop_inner()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # socket error, protocol error, anything
            log.debug("CDP recv loop ended: %r", exc)
            self._fail_pending(exc)
            return
        # Normal iterator exhaustion == the socket closed.
        self._fail_pending(RuntimeError("CDP connection closed"))

    async def _recv_loop_inner(self) -> None:
        assert self._ws
        async for raw in self._ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if "id" in msg:
                fut = self._pending.pop(msg["id"], None)
                if fut and not fut.done():
                    if "error" in msg:
                        fut.set_exception(RuntimeError(msg["error"].get("message", "CDP error")))
                    else:
                        fut.set_result(msg.get("result", {}))
            elif "method" in msg:
                method = msg["method"]
                params = msg.get("params", {})
                if method == "Page.screencastFrame":
                    # Decode is CPU work handed to an executor and must
                    # not block draining the socket, so this one event
                    # gets its own task. Track it so teardown can await
                    # in-flight frames instead of orphaning them.
                    self._fire_and_forget(self._on_screencast_frame(params))
                else:
                    # Navigate/load/console/error hooks are cheap and
                    # already-async; await them inline rather than
                    # spawning a task each. Keeps the socket draining
                    # (these don't do heavy work) at one fewer task per
                    # event than the old create_task-everything path.
                    await self._dispatch_event(method, params)

    def _fire_and_forget(self, coro: Coroutine[Any, Any, None]) -> None:
        """Schedule *coro* as a tracked background task.

        Tasks are held in ``self._bg_tasks`` until done so they aren't
        garbage-collected mid-flight (a documented asyncio footgun), and
        so ``stop()`` can drain in-flight screencast decodes rather than
        cancelling them out from under the FrameBuffer writer.
        """
        task = asyncio.ensure_future(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _dispatch_event(self, method: str, params: dict[str, Any]) -> None:
        # Page.screencastFrame is handled directly in _recv_loop (it needs
        # its own task for the off-loop decode); everything here is a
        # cheap hook fire awaited inline by the recv loop.
        if method == "Page.navigatedWithinDocument" or method == "Page.frameNavigated":
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
        """Decode one screencast frame and commit it to the FrameBuffer.

        Ack ordering
        ------------
        `Page.screencastFrameAck` is Chrome's flow control: it does not
        push the next frame until the previous one is acked. Acking
        *before* the decode (the old behaviour) throws that away — frame
        N+1 arrives while N is still in the executor, both tasks race to
        `_buf.write()`, and the default executor's completion order is
        not the arrival order, so an older frame can land on top of a
        newer one. With `ack_after_write=True` (the default) exactly one
        frame is ever in flight, which makes ordering structural rather
        than lucky. Set `ack_after_write=False` to restore the old
        pipelined behaviour if you would rather have throughput than
        ordering.
        """
        session_id = params.get("sessionId", 0)
        if not self._ack_after_write:
            await self._send_nowait("Page.screencastFrameAck", sessionId=session_id)

        data = params.get("data", "")
        ts = params.get("metadata", {}).get("timestamp", time.monotonic())
        width, height = self._width, self._height

        try:
            loop = asyncio.get_running_loop()
            rgba = await loop.run_in_executor(None, decode_frame, data, width, height)
            fid = self._buf.write(rgba)
        except Exception:
            log.exception("screencast frame dropped")
            return
        finally:
            if self._ack_after_write:
                # In `finally` so a decode/write failure does not stall the
                # stream permanently with no ack outstanding.
                if self._min_screencast_interval:
                    now = time.monotonic()
                    wait = self._min_screencast_interval - (now - self._last_ack_at)
                    if wait > 0:
                        await asyncio.sleep(wait)
                    self._last_ack_at = time.monotonic()
                with suppress(Exception):
                    await self._send_nowait(
                        "Page.screencastFrameAck", sessionId=session_id
                    )

        frame = Frame(
            data=rgba,
            width=width,
            height=height,
            frame_id=fid,
            timestamp=ts,
        )
        await self._hooks.fire("frame", frame=frame)

    # ------------------------------------------------------------------
    # Navigation & page control
    # ------------------------------------------------------------------

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

    async def wait_for_selector(self, selector: str, *, timeout: float = 10.0) -> bool:
        js = f"""
        new Promise((resolve) => {{
            const sel = {selector!r};
            if (document.querySelector(sel)) return resolve(true);
            const obs = new MutationObserver(() => {{
                if (document.querySelector(sel)) {{
                    obs.disconnect();
                    resolve(true);
                }}
            }});
            obs.observe(document.body, {{ childList: true, subtree: true }});
            setTimeout(() => {{ obs.disconnect(); resolve(false); }}, {int(timeout * 1000)});
        }})
        """
        return bool(await self.eval(js, await_promise=True))

    async def set_viewport(self, width: int, height: int) -> None:
        """Change the emulated viewport and re-point the screencast at it.

        The FrameBuffer is fixed-size, so a viewport that no longer
        matches it makes every subsequent `_buf.write()` raise inside a
        background task. That is warned about here rather than
        discovered as a stream that silently stopped.
        """
        buf_w = getattr(self._buf, "width", width)
        buf_h = getattr(self._buf, "height", height)
        if (width, height) != (buf_w, buf_h):
            log.warning(
                "set_viewport(%d, %d) does not match the FrameBuffer (%dx%d); "
                "frames will be resampled to the buffer's size on decode.",
                width,
                height,
                buf_w,
                buf_h,
            )
        self._width = width
        self._height = height
        await self._send(
            "Emulation.setDeviceMetricsOverride",
            width=width,
            height=height,
            deviceScaleFactor=1,
            mobile=False,
        )
        if self._running:
            # maxWidth/maxHeight were baked in at startScreencast time.
            await self._restart_screencast()

    # ------------------------------------------------------------------
    # Extra HTTP headers
    # ------------------------------------------------------------------
    #
    # CDP's Network.setExtraHTTPHeaders replaces the *entire* header set
    # on every call — there is no incremental add/remove at the protocol
    # level. To expose an add/remove API anyway, headers are tracked
    # locally as named groups (one id -> one dict of headers, added
    # together and removed together) and the full merged dict is
    # recomputed and re-sent to CDP on every mutation.
    #
    # Merge order is insertion order of the groups; if two groups define
    # the same header name, the most-recently-added group wins.

    async def _push_headers(self) -> None:
        merged: dict[str, str] = {}
        for group in self._header_groups.values():
            merged.update(group)
        await self._send("Network.setExtraHTTPHeaders", headers=merged)

    async def header_add(self, headers: dict[str, str]) -> str:
        """
        Register a group of one or more extra HTTP headers and push the
        merged header set to Chrome via CDP.

        Returns an opaque ``id``. Pass it to ``header_remove`` later to
        remove exactly this group (other groups are unaffected).

        Example
        -------
        >>> gid = await bridge.header_add({"X-Trace-Id": "abc123"})
        >>> await bridge.header_remove(gid)
        """
        group_id = uuid.uuid4().hex
        self._header_groups[group_id] = dict(headers)
        await self._push_headers()
        return group_id

    async def header_remove(self, group_id: str) -> None:
        """Remove a header group by the id returned from ``header_add``,
        and push the updated header set to Chrome. No-op (does not raise)
        if the id is unknown."""
        self._header_groups.pop(group_id, None)
        await self._push_headers()

    async def header_remove_all(self) -> None:
        """Clear all extra headers and push the (now empty) set to Chrome."""
        self._header_groups.clear()
        await self._push_headers()

    def headers_current(self) -> dict[str, str]:
        """Return the currently merged header dict (local state, no CDP call)."""
        merged: dict[str, str] = {}
        for group in self._header_groups.values():
            merged.update(group)
        return merged

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

        Resolution order, parameters, and return semantics for *finding*
        the element are identical to :meth:`get_bounds` — both methods
        share one implementation (``_build_finder_js``), so "what would
        click_element click?" and "what does get_bounds report?" never
        disagree. See :meth:`get_bounds` for the full resolution-order
        docs (selector first, text-substring-with-nearest-clickable-
        ancestor fallback second). This docstring only covers what's
        click-specific.

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

        box = await self.get_bounds(
            selector,
            text=text,
            nth=nth,
            scroll_into_view=scroll_into_view,
            timeout=timeout,
            poll_interval=poll_interval,
        )
        if box is None:
            return False

        cx = box["x"] + box["width"] / 2
        cy = box["y"] + box["height"] / 2
        await self.click(cx, cy, button=button)
        return True

    # ------------------------------------------------------------------
    # Element bounds (no click)
    # ------------------------------------------------------------------
    #
    # get_bounds() is one real implementation with two @overload stubs
    # layered on top, so a type checker can narrow the return type from
    # *which literal value of nth was passed* instead of seeing one big
    # `dict | list[dict] | None` union on every call site. Without this,
    # `box = await br.get_bounds(sel)` followed by `box["x"]` makes a
    # type checker consider the `list[dict]` branch of the union too —
    # `list.__getitem__` only accepts int/slice, so `box["x"]` looks like
    # an error there even though, at the default `nth=0`, the *runtime*
    # return is always a single dict (or None). The two stubs below tell
    # the checker that an int (or the implicit default) `nth` means "you
    # get a dict or None back" and `nth=None` means "you get a list
    # back" — matching the actual runtime contract documented on the
    # implementation itself.
    #
    # These overloads are type-checking-only — they're never called at
    # runtime (the `...` bodies are never executed); only the final,
    # non-decorated `get_bounds` below has a real body.

    @overload
    async def get_bounds(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[str] = None,
        nth: int = 0,
        detail: bool = False,
        scroll_into_view: bool = False,
        timeout: float = 5.0,
        poll_interval: float = 0.15,
    ) -> Optional[dict[str, Any]]: ...

    @overload
    async def get_bounds(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[str] = None,
        nth: None,
        detail: bool = False,
        scroll_into_view: bool = False,
        timeout: float = 5.0,
        poll_interval: float = 0.15,
    ) -> list[dict[str, Any]]: ...

    async def get_bounds(
        self,
        selector: Optional[str] = None,
        *,
        text: Optional[str] = None,
        nth: Optional[int] = 0,
        detail: bool = False,
        scroll_into_view: bool = False,
        timeout: float = 5.0,
        poll_interval: float = 0.15,
    ) -> Union[dict[str, Any], list[dict[str, Any]], None]:
        """
        Resolve element(s) by *selector* and/or *text* and return their
        bounding box(es), without clicking anything.

        Resolution order
        -----------------
        1. If *selector* is given, ``document.querySelectorAll(selector)``
           is evaluated. If it yields at least one match, those matches
           are used directly (selector matches are NOT redirected to a
           clickable ancestor — if you wrote the selector, you meant
           that element).
        2. Otherwise — or if *selector* matched nothing — *text* is used
           as a case-insensitive substring match against every element's
           *own* text content (direct text-node children only, not
           descendants' concatenated text). Each match is then resolved
           to its nearest clickable ancestor (``a``, ``button``,
           ``input``, ``select``, ``textarea``, ``[role="button"]``,
           ``[onclick]``, ``summary``, ``label``, or the matched element
           itself if it's already one of those) and the result list is
           de-duplicated. This is the exact same resolution
           ``click_element`` uses (both methods share one
           implementation, ``_build_finder_js``), so "what would
           click_element click?" and "what does get_bounds report?"
           never disagree.
        3. Elements with a zero-area box (detached/hidden) are excluded
           from the result entirely — they never appear in the list and
           never count toward *nth*.

        This call polls (same as ``wait_for_selector``/``click_element``)
        until at least one element resolves or *timeout* elapses; it does
        not fail just because the page hasn't finished rendering yet.

        Parameters
        ----------
        selector:
            CSS selector, tried first.
        text:
            Substring to match against element text content; fallback if
            *selector* is None or matches nothing.
        nth:
            - ``int`` (default ``0``): return a single box — the box at
              that index in the match list, or ``None`` if there are
              fewer than ``nth + 1`` matches (or zero matches at all).
            - ``None``: return *every* matching box as a list (``[]`` if
              nothing matched). Useful when you don't know in advance
              how many elements a selector/text will match — e.g.
              collecting every row in a table or every link in a nav.
        detail:
            If True, each returned dict also includes ``tag`` (lowercased
            tag name) and ``text`` (a trimmed, truncated-to-200-chars
            snippet of the element's own visible text), in addition to
            ``x``/``y``/``width``/``height``. Default False — the plain
            geometry dict, which is all ``click_element`` itself needs
            internally.
        scroll_into_view:
            If True, scrolls the *nth* match into view and re-measures
            every box against the post-scroll layout before returning.
            Only meaningful when *nth* is an int — scrolling "the nth
            element" has no defined meaning when you've asked for the
            whole list. Combining ``scroll_into_view=True`` with
            ``nth=None`` raises ``ValueError`` rather than silently
            ignoring the flag, since silently dropping a parameter the
            caller explicitly set is exactly the kind of thing that
            bites someone later in a debugging session.
        timeout, poll_interval:
            Same semantics as ``wait_for_selector``/``click_element`` —
            poll until at least one element appears or *timeout*
            elapses.

        Returns
        -------
        dict | None
            When *nth* is an int: ``{"x", "y", "width", "height"}``
            (plus ``"tag"``/``"text"`` if *detail* is True), or ``None``
            if no element at that index was found within *timeout*.
        list[dict]
            When *nth* is ``None``: every matching box, in the order
            described above. Empty list if nothing matched within
            *timeout* — note this is a plain empty list, not ``None``,
            since "the list of matches" and "no list at all" are
            different things when *nth* itself wasn't asking for one
            specific element.

        Raises
        ------
        ValueError
            If neither *selector* nor *text* is given, or if
            ``scroll_into_view=True`` is combined with ``nth=None``.

        Examples
        --------
        Single element, the common case (mirrors what ``click_element``
        would click)::

            box = await br.get_bounds("#submit-button")
            if box is not None:
                print(box["x"], box["y"])

        Every row in a table, to drive your own iteration::

            rows = await br.get_bounds("table.results tr")
            for row in rows:
                cx = row["x"] + row["width"] / 2
                cy = row["y"] + row["height"] / 2
                await br.click(cx, cy)

        With detail, to disambiguate which text match you got::

            box = await br.get_bounds(text="Submit", detail=True)
            print(box["tag"], box["text"])
        """
        if selector is None and text is None:
            raise ValueError("get_bounds requires at least one of selector or text")
        if scroll_into_view and nth is None:
            raise ValueError(
                "scroll_into_view=True has no defined meaning with nth=None "
                "(there is no single element to scroll to) — pass an int nth, "
                "or drop scroll_into_view."
            )

        scroll_idx = nth if scroll_into_view else None
        finder_js = _build_finder_js(selector, text, scroll_into_view_nth=scroll_idx, detail=detail)

        elapsed = 0.0
        while True:
            boxes = await self.eval(finder_js, await_promise=False)
            if boxes:  # non-empty list — at least one element resolved
                break
            if elapsed >= timeout:
                boxes = []
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        if nth is None:
            return boxes
        if 0 <= nth < len(boxes):
            return boxes[nth]
        return None

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


#: Decode moved to wbb/codec.py (it now dispatches on the payload's magic
#: bytes, because the screencast format is per-bridge). Re-exported so
#: anything importing wbb.browser._jpeg_to_rgba keeps working.
_jpeg_to_rgba = decode_frame


# Elements considered "clickable" when resolving a text match to its
# nearest interactive ancestor. Selector matches skip this entirely.
_CLICKABLE_SEL = 'a,button,input,select,textarea,[role="button"],[onclick],summary,label'


def _build_finder_js(
    selector: Optional[str],
    text: Optional[str],
    *,
    scroll_into_view_nth: Optional[int] = None,
    detail: bool = False,
) -> str:
    """
    Build a JS expression for Runtime.evaluate that resolves elements by
    selector (preferred) or text substring (fallback) and returns a JSON
    array of every match's bounding box, in document order after the
    selector/text/nearest-clickable-ancestor resolution described below.
    Returns ``[]`` (not ``null``) if nothing resolves — callers handle
    the "empty vs missing index" distinction on the Python side
    (``get_bounds``/``click_element``), since whether an empty list means
    "still loading, keep polling" or "definitely absent" depends on
    elapsed time the JS itself has no visibility into.

    This single function is shared by both ``click_element`` and
    ``get_bounds`` — there is exactly one implementation of the
    selector/text resolution logic, not one per caller, which is what
    guarantees they never disagree about which element a given
    selector/text resolves to.

    Resolution order
    -----------------
    1. If *selector* is given, ``document.querySelectorAll(selector)`` is
       evaluated. If it yields at least one match, those matches are used
       directly (selector matches are NOT redirected to a clickable
       ancestor — if you wrote the selector, you meant that element).
    2. Otherwise — or if *selector* matched nothing — *text* is used as a
       case-insensitive substring match against every element's *own*
       text content (direct text-node children only, not descendants'
       concatenated text). Each match is then resolved to its nearest
       clickable ancestor (``a``, ``button``, ``input``, ``select``,
       ``textarea``, ``[role="button"]``, ``[onclick]``, ``summary``,
       ``label``, or the matched element itself if it's already one of
       those) and the result list is de-duplicated. This matters in
       practice: matching text like "Login" inside
       ``<button><b>Login</b></button>`` should resolve to the button,
       not the inert ``<b>`` wrapping it — verified against jsdom.

    Each returned box is ``{x, y, width, height}`` (viewport-relative —
    the coordinate space ``Input.dispatchMouseEvent`` expects), or, if
    *detail* is True, additionally ``{tag, text}`` where ``tag`` is the
    lowercased tag name and ``text`` is a trimmed, whitespace-collapsed,
    200-char-truncated snippet of the element's own text content (same
    "direct text-node children only" rule as the text-match step above).
    Elements with a zero-area box (detached or ``display: none``) are
    excluded from the output entirely — they never occupy a slot in the
    returned array, so they never shift or get assigned an *nth* index.

    *scroll_into_view_nth*, if given, calls ``element.scrollIntoView()``
    on exactly that index of the (deduplicated) match list before any
    boxes are measured, so every returned box — not just that one
    element's — reflects the post-scroll layout. Mirrors the original
    click_element behavior of re-reading the bounding box after a
    scroll, generalized to the fact that this function now always
    returns the full list rather than one pre-selected box.

    *selector* and *text* are JSON-encoded into the expression so
    arbitrary strings (quotes, backslashes, unicode) embed safely —
    ``json.dumps`` rather than Python ``repr``, since this is JS, not
    Python (the existing ``wait_for_selector`` uses ``repr`` because it
    calls ``querySelector`` directly with a Python-side f-string; this
    builds a bigger expression so a JSON literal is cleaner).
    """
    sel_json = json.dumps(selector) if selector is not None else "null"
    text_json = json.dumps(text) if text is not None else "null"
    detail_json = "true" if detail else "false"
    scroll_idx_json = str(int(scroll_into_view_nth)) if scroll_into_view_nth is not None else "null"

    return f"""
(() => {{
    const sel = {sel_json};
    const textNeedle = {text_json};
    const wantDetail = {detail_json};
    const scrollIdx = {scroll_idx_json};
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

    if (scrollIdx !== null && matches[scrollIdx]) {{
        matches[scrollIdx].scrollIntoView({{ block: 'center', inline: 'center' }});
    }}

    const out = [];
    for (const el of matches) {{
        const r = el.getBoundingClientRect();
        if (r.width === 0 && r.height === 0) continue;  // detached/hidden
        const box = {{ x: r.x, y: r.y, width: r.width, height: r.height }};
        if (wantDetail) {{
            box.tag = el.tagName ? el.tagName.toLowerCase() : '';
            let own = '';
            for (const node of el.childNodes) {{
                if (node.nodeType === Node.TEXT_NODE) own += node.textContent;
            }}
            own = own.trim().replace(/\\s+/g, ' ');
            box.text = own.length > 200 ? own.slice(0, 200) : own;
        }}
        out.push(box);
    }}
    return out;
}})()
"""
