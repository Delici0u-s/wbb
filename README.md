# wbb — WebView Buffer Bridge

Render a live website into an off-screen pixel buffer and expose it as a programmable primitive.

`wbb` drives headless Chrome over the Chrome DevTools Protocol (CDP), decodes its screencast stream into RGBA frames, and writes those frames into a POSIX shared-memory double-buffer. Any process, in the same script or a separate one, can attach to that buffer: read frames with zero-copy views, display them in a window, record them to disk or ffmpeg, or run a filter pipeline over them. Mouse, keyboard, and DOM-query input can be forwarded back into the page, so the buffer is a full interactive surface, not just a screenshot source. The point is to get pixels out of a real, JS-executing browser into your own process at low latency, without polling screenshot RPCs and without tying the renderer to the same process as the consumer.

## Installation

```bash
pip install wbb
```

System requirements:

- **Chrome or Chromium** on `PATH` (checked names: `google-chrome`, `google-chrome-stable`, `chromium`, `chromium-browser`, `chrome`; common macOS/Windows install paths are also checked). Override with the `CHROME_PATH` environment variable.
- **ffmpeg**, only if you pipe frames to it yourself (see `examples/05_record_to_ffmpeg.py`). Not a library dependency.

Optional extras:

```bash
pip install wbb[display]    # PyGObject, for legacy GTK4 layer-shell display mode
pip install wbb[fast-jpeg]  # PyTurboJPEG, faster screencast JPEG decoding
pip install wbb[dev]        # pytest, mypy, ruff
```

`wbb.display` (the SDL2 `DisplayClient`) also needs **PySDL2** and the **SDL2** system library. Neither is pulled in by any extra above, so install them yourself (`pip install pysdl2` plus your OS's `libsdl2` package). The KWin placement backend needs `pydbus`; the X11/EWMH backend needs `python-xlib`. Both are optional. `DisplayClient` falls back to a plain, unpositioned window if neither is present.

## Quickstart

```python
import asyncio
from wbb import BrowserBridge, FrameBuffer

async def main():
    buf = FrameBuffer("my_buf", 1280, 720)
    async with BrowserBridge(buf) as br:
        await br.navigate("https://example.com")
        frame = await buf.next_frame()
        frame.save("screenshot.png")
    buf.close()
    buf.unlink()

asyncio.run(main())
```

## Features

- CDP screencast capture, no screenshot polling
- Shared-memory double-buffer with zero-copy frame reads
- Cross-process display: render in one process, view in another
- Composable filter pipelines (crop, scale, color, blur, custom)
- SDL2 window with always-on-top, absolute position, and click-through, where the desktop supports it
- DOM-aware interaction: `wait_for_selector`, `click_element` (selector or text fallback)
- Browser process pooling for reuse across multiple sites
- Frame recording to PNG sequences or an ffmpeg pipe
- Per-bridge extra HTTP header management (add/remove by group)
- Event hooks: `frame`, `navigate`, `load`, `console`, `error`

## Documentation

### `wbb.buffer`

#### `class FrameBuffer(name, width, height, *, attach=False)`

Shared, mutable RGBA pixel buffer backed by two POSIX shared-memory segments (`<name>_a`, `<name>_b`) plus a small metadata segment that tracks which one is current. The writer alternates buffers and flips a pointer after each write, so a reader never sees a torn frame.

- **Parameters**
  - `name: str` — shared-memory identifier. Two `FrameBuffer` instances (same process or different processes) with the same name attach to the same data.
  - `width, height: int` — frame dimensions. The writer's `write()` calls must match exactly.
  - `attach: bool` — `False` (default) creates and owns the segments; `True` attaches to segments created elsewhere. Only the owner should call `unlink()`.

```python
buf = FrameBuffer("ex01", 1280, 720)          # creates segments
viewer_buf = FrameBuffer("ex01", 1280, 720, attach=True)  # attaches, e.g. in another process
```

##### `FrameBuffer.write(rgba: np.ndarray) -> int`

Writer-side. Copies an H×W×4 `uint8` RGBA array into the inactive buffer, then flips the active pointer. Returns the new monotonically increasing `frame_id`. Raises `ValueError` if `rgba.shape != (height, width, 4)`. Called internally by `BrowserBridge`. You generally don't call this yourself unless compositing frames manually (see `examples/07_display_multiple_sites_with_pool.py`).

##### `FrameBuffer.read() -> Frame`

Returns the latest frame as a **zero-copy view** into shared memory. The view goes stale the instant the writer flips again. It doesn't update in place; it just silently starts showing old pixels once a newer frame exists elsewhere. Call `frame.copy()` if you need to keep it past the next write.

```python
frame = buf.read()
frame.save("latest.png")
del frame  # release the view
```

##### `async FrameBuffer.next_frame(timeout: float = 5.0) -> Frame`

Blocks, without busy-polling (backed by a `threading.Condition`), until a frame newer than the last one observed arrives, or `timeout` seconds pass, then returns `read()`. On timeout it does **not** raise or return `None`. It just returns whatever is currently committed, so repeated calls under a slow or idle screencast are harmless.

```python
frame = await buf.next_frame(2)  # wait up to 2s for a new frame
```

##### `async for frame in buf:` (`FrameBuffer.__aiter__`)

Infinite async iterator, equivalent to calling `next_frame()` in a loop with the default timeout. Used by consumers like `examples/06_combined_scenario.py`'s `record_task`/`monitor_task`.

##### `FrameBuffer.close() -> None`

Releases this process's own memory mappings. **Drop every `Frame` (and any array derived from one, e.g. via `.crop()`) before calling this.** CPython won't unmap memory while a buffer-protocol export is still live on it, the same rule `mmap` follows generally. If a view is still outstanding, `close()` logs a debug message and returns instead of raising. The segment releases once you drop the last reference and GC runs.

##### `FrameBuffer.unlink() -> None`

Destroys the underlying named shared-memory segments. Only the creator (`attach=False`) should call this. Safe to call even if `close()` couldn't fully release a mapping: `unlink()` only removes the name so no new process can attach. This process's own mapping is freed by the OS at exit regardless.

##### Context manager

`with FrameBuffer(...) as buf:` calls `close()` on exit, and `unlink()` too if the buffer wasn't created with `attach=True`.

### `wbb.frame`

#### `class Frame` (frozen dataclass)

A snapshot of one rendered frame.

- **Fields**
  - `data: np.ndarray` — H×W×4 `uint8` RGBA. Read-only view into shared memory unless produced by `.copy()`.
  - `width, height: int`
  - `frame_id: int` — unique within one `FrameBuffer` session. Two `Frame`s with the same id are the same underlying frame.
  - `timestamp: float` — `time.monotonic()` at write time.

##### `Frame.copy() -> np.ndarray`

Detaches the data from shared memory entirely. A writable array, safe to keep past `FrameBuffer.close()`.

##### `Frame.crop(x, y, w, h) -> Frame`

Returns a new `Frame` whose `data` is a zero-copy sub-view. `frame_id`/`timestamp` are inherited from the source. Still pins shared memory, same lifecycle rule as the parent.

##### `Frame.save(path, *, format=None) -> None`

Writes the frame to disk via Pillow. Format is inferred from the extension unless given explicitly (e.g. `"PNG"`, `"JPEG"`).

```python
frame = await buf.next_frame()
frame.crop(0, 0, 640, 360).save("corner.png")
```

### `wbb.browser`

#### `class BrowserBridge(buffer, *, width=1280, height=720, screencast_quality=80, screencast_max_fps=30, enable_input=None, headless_args=None, extra_headers=None)`

Owns one headless Chrome process and its CDP WebSocket connection. Frames arrive via `Page.startScreencast` push, no polling. Each JPEG payload is decoded to RGBA off the event loop (PyTurboJPEG if installed, else Pillow) and written into `buffer`.

- **Parameters**
  - `buffer: FrameBuffer` — destination for decoded frames.
  - `width, height: int` — viewport size, also passed to `Emulation.setDeviceMetricsOverride`.
  - `screencast_quality: int` — JPEG quality 1-100 for the CDP stream.
  - `screencast_max_fps: int` — upper bound Chrome is asked to push at.
  - `enable_input: Optional[bool]` — `None` or anything but `False` enables `click`/`key`/`type`/`scroll`/`move`/`click_element`; `False` makes all of them raise `RuntimeError`.
  - `headless_args: list[str] | None` — extra Chrome CLI flags, appended after the built-ins.
  - `extra_headers: dict[str, str] | None` — seeds an initial header group (see `header_add` below).

```python
async with BrowserBridge(buf, width=1280, height=720, enable_input=True) as br:
    await br.navigate("https://example.com")
```

##### `await BrowserBridge.start()` / `await BrowserBridge.stop()`

Manual lifecycle pair behind the `async with` context manager. `start()` launches Chrome, attaches a flattened CDP session to its `about:blank` page target, enables `Page`/`Runtime`/`Network`, pushes any seeded headers, sets the viewport, and starts the screencast. `stop()` stops the screencast, cancels the receive loop, closes the socket, and terminates (then kills, if unresponsive after 5s) the Chrome process.

##### `await BrowserBridge.navigate(url: str) -> None`

Sends `Page.navigate`. Does **not** wait for load itself. Use `wait_for_selector`, the `"load"` event hook, or your own `asyncio.sleep`/polling, depending on what "loaded" means for your page.

##### `await BrowserBridge.reload(*, ignore_cache: bool = False) -> None`

##### `await BrowserBridge.eval(expression: str, *, await_promise: bool = False) -> Any`

Evaluates `expression` via `Runtime.evaluate` with `returnByValue=True` and returns the resulting JSON-compatible value.

##### `await BrowserBridge.wait_for_selector(selector: str, *, timeout: float = 10.0) -> bool`

Polls via a `MutationObserver` (not a busy loop) until `document.querySelector(selector)` matches or `timeout` elapses. Returns whether it matched.

```python
found = await br.wait_for_selector('i[data-jsl10n="search-input-button"]')
```

##### `await BrowserBridge.set_viewport(width: int, height: int) -> None`

Updates `width`/`height` and re-issues `Emulation.setDeviceMetricsOverride`. Does **not** resize the screencast's `maxWidth`/`maxHeight`. Restart the bridge if you need that to change too.

##### Extra HTTP headers: `header_add`, `header_remove`, `header_remove_all`, `headers_current`

CDP's `Network.setExtraHTTPHeaders` replaces the *entire* header set on every call. There's no incremental add/remove at the protocol level. `BrowserBridge` works around this by tracking headers as named groups and re-merging plus re-sending the full set on every mutation.

- `await header_add(headers: dict[str, str]) -> str` — registers a group, pushes the merged set, returns an opaque group id.
- `await header_remove(group_id: str) -> None` — removes one group by id (others unaffected), re-pushes. No-op on an unknown id.
- `await header_remove_all() -> None` — clears every group, pushes an empty header set.
- `headers_current() -> dict[str, str]` — local merged state, no CDP round-trip.

```python
gid = await br.header_add({"X-Trace-Id": "abc123"})
...
await br.header_remove(gid)
```

##### Input: `click`, `move`, `scroll`, `key`, `type`

All raise `RuntimeError` if the bridge was constructed with `enable_input=False`.

- `await click(x: float, y: float, *, button: str = "left") -> None` — dispatches a `mousePressed`+`mouseReleased` pair at pixel coordinates.
- `await move(x: float, y: float) -> None`
- `await scroll(x: float, y: float, delta_x: float = 0, delta_y: float = 100) -> None`
- `await key(key: str, *, kind: str = "keyDown", text: Optional[str] = None) -> None` — `key` is a DOM key value (`"Enter"`, `"a"`, etc.). `text` defaults to `key` itself for single printable characters and to `""` otherwise. **Known rough edge:** a manual `keyDown`/`keyUp` pair for `"Enter"` doesn't reliably submit forms in practice. Sending `"\r"` through `type()`, or calling `eval()` to submit the form directly, both work as documented workarounds (see `examples/03_multi_step_automation.py`).
- `await type(text: str) -> None` — inserts `text` via `Input.insertText`, as if pasted or typed.

##### `await BrowserBridge.click_element(selector=None, *, text=None, nth=0, scroll_into_view=False, timeout=5.0, poll_interval=0.15, button="left") -> bool`

Clicks an element resolved by selector and/or text instead of a hardcoded pixel position.

- **Resolution order:** if `selector` matches anything, the `nth` match is used as-is (no ancestor redirection: a selector match is exactly what you asked for). Otherwise `text` is matched case-insensitively against each element's own direct text-node content, and each match is redirected to its nearest clickable ancestor (`a`, `button`, `input`, `select`, `textarea`, `[role="button"]`, `[onclick]`, `summary`, `label`, or itself), then de-duplicated.
- Polls (`timeout`/`poll_interval`) until something resolves; returns `False` on timeout rather than raising.
- Clicks the resolved element's `getBoundingClientRect()` center via `click()`.
- `scroll_into_view=True` re-resolves and calls `scrollIntoView({block: 'center', inline: 'center'})` first. If the element vanished between find and scroll, returns `False` instead of clicking stale coordinates.
- Raises `ValueError` if neither `selector` nor `text` is given; raises `RuntimeError` if input is disabled.

```python
await br.click_element("button.pure-button.pure-button-primary-progressive")
await br.click_element(text="Deutsch")  # text fallback, no selector
```

##### `BrowserBridge.on(event: str, cb) -> None`

Registers a callback for `"frame"`, `"navigate"`, `"load"`, `"console"`, or `"error"`. `cb` may be sync or a coroutine function; both are awaited correctly.

```python
br.on("load", lambda: print("loaded"))
br.on("navigate", lambda url: print(f"[nav] {url}"))
```

##### `async BrowserBridge.frames() -> AsyncIterator[Frame]`

Thin pass-through async iterator over the underlying `FrameBuffer`.

##### `BrowserBridge.buffer -> FrameBuffer`

The buffer this bridge writes into.

### `wbb.pool`

#### `class BrowserPool(max_idle: int = 3)` (exported as `BrowserPool`, defined as `ChromePool`)

Reuses warm Chrome processes across `BrowserBridge` instances instead of paying full process-launch cost per site.

##### `await BrowserPool.acquire(buffer: FrameBuffer, **kwargs) -> BrowserBridge`

Pops an idle bridge if one exists: rebinds it to `buffer`, navigates it to `about:blank`, updates the viewport if `width`/`height` are in `kwargs`. Otherwise constructs and starts a fresh `BrowserBridge(buffer, **kwargs)`.

##### `await BrowserPool.release(br: BrowserBridge) -> None`

Returns `br` to the idle pool (navigating it to `about:blank` first) if there's room under `max_idle`; otherwise calls `br.stop()`.

```python
pool = BrowserPool(max_idle=3)
br = await pool.acquire(buffer=buf, width=640, height=360)
await br.navigate(url)
...
await pool.release(br)
```

### `wbb.filters`

Every filter has the signature `(np.ndarray) -> np.ndarray` over an H×W×4 `uint8` RGBA array. Built-ins are pure: they return new arrays or views and never mutate input, so user-defined filters of the same shape plug in without modification.

| Function | Signature | Behavior |
|---|---|---|
| `crop` | `crop(x, y, width, height) -> Filter` | Zero-copy sub-region slice. |
| `scale` | `scale(width, height) -> Filter` | Pillow `LANCZOS` resize. |
| `flip` | `flip(*, horizontal=False, vertical=False) -> Filter` | Zero-copy slice-based flip. |
| `grayscale` | `grayscale() -> Filter` | ITU-R BT.601 luminance; alpha preserved. |
| `colorize` | `colorize(r=1.0, g=1.0, b=1.0, a=1.0) -> Filter` | Per-channel multiply, clipped to [0, 255]. |
| `brightness` | `brightness(delta: int) -> Filter` | Additive RGB shift; alpha untouched. |
| `contrast` | `contrast(factor: float) -> Filter` | Scales RGB around midpoint 128. |
| `blur` | `blur(radius: int = 2) -> Filter` | Separable box blur via `scipy.ndimage.uniform_filter`; falls back to a 5-tap neighbor average if scipy isn't installed. |
| `compose` | `compose(first, second) -> Filter` | `second(first(x))`. |
| `chain` | `chain(*filters) -> Filter` | Left-to-right composition of any number of filters. |
| `identity` | `identity() -> Filter` | Pass-through. |

```python
pipeline = filters.chain(
    filters.colorize(r=0.95, g=0.95, b=1.1),
    filters.contrast(1.1),
)
```

### `wbb.display` (SDL2 `DisplayClient`)

Requires PySDL2 plus system SDL2 (see Installation). Renders a `FrameBuffer` into a window, running everything (SDL event polling and frame pull) as a single coroutine on the caller's own asyncio loop. No second thread, no second main loop.

#### `class DisplayClient(buffer, *, title="wbb", wm_class="wbb-display", filters=None, on_mouse_event=None, on_key_event=None, on_scroll_event=None, window_size=None, position=(0, 0), monitor=0, always_on_top=False, borderless=False, click_through=False, max_fps=60.0)`

- **Parameters**
  - `buffer` — anything with `.width`, `.height`, and an async `next_frame(timeout)` (i.e. a `FrameBuffer`).
  - `wm_class: str` — give each concurrent `DisplayClient` a distinct value. The KWin and X11 placement backends match windows by this.
  - `filters: list[Filter] | None` — run in a thread-pool executor, not inline on the event loop, so a slow chain (Pillow resize, scipy blur) never stalls CDP receive or input dispatch.
  - `position: tuple[int, int]` — **monitor-local**, not global-desktop. Must be a 2-tuple of `int`; raises `ValueError` otherwise.
  - `monitor: int` — index into `list_displays()` that `position` is local to. Index order is whatever the OS reports, not guaranteed left-to-right. Call `list_displays()` to check.
  - `always_on_top, borderless` — best-effort. Silently no-op if no placement backend supports them on the current session. `borderless` is unconditional (`SDL_WINDOW_BORDERLESS` at window creation) and always works.
  - `click_through: bool` — real on X11 sessions (via the Shape extension's input region); a clean, logged no-op under KWin's Wayland scripting backend, which has no input-transparency mechanism at all. Check `is_click_through_active()` to know which happened.
  - `max_fps: float` — caps how often frames are pushed (gates entry into the filter chain, not just the final present). `0` disables the cap.

```python
display = DisplayClient(buf, title="wbb: filtered view", filters=pipeline, window_size=(640, 360))
await display.run_async()
```

##### `await DisplayClient.run_async() -> None` / `DisplayClient.run() -> None`

`run_async()` opens the window and runs until `stop()` is called, the user closes the window, or `next_frame()` errors out. `run()` is a sync wrapper (`asyncio.run(self.run_async())`) for non-async call sites.

**XWayland first-map race, and why placement is retried:** on KWin/XWayland, a freshly created toplevel isn't guaranteed to have finished its first map/configure round-trip with the compositor the instant `SDL_CreateWindow` returns. A position-setting `XConfigureWindow` sent before that round-trip completes can be silently superseded by the compositor's own initial placement once it actually maps the window. `always_on_top` (sent as an EWMH client message, which gets queued and redelivered) is unaffected, but absolute `position` is. The fix: `DisplayClient` re-applies `always_on_top`/`position` for the first 5 render-loop iterations. Not a fixed sleep, since the right delay depends on the machine and compositor. This costs a handful of extra D-Bus/Xlib round-trips, only during startup.

##### `DisplayClient.stop() -> None`

Sets a flag the running `run_async()` loop checks every iteration.

##### `DisplayClient.set_position(position, *, monitor=None) -> None`

Moves the window after startup. No-op if called before `run_async()` has created the window. Re-arms the placement retry window (5 frames) to also cover monitor hot-plug or compositor-reset edge cases.

##### `DisplayClient.get_position() -> WindowPosition`

Returns the last-*requested* monitor-local position, not necessarily the resolved global coordinate actually sent to the backend. `WindowPosition` is a frozen dataclass with `x`, `y`.

##### `DisplayClient.set_always_on_top(above: bool) -> None`

##### `DisplayClient.set_click_through(enabled: bool) -> bool`

Returns whether it actually took effect, same contract as the constructor flag.

##### `DisplayClient.is_positionable() -> bool` / `DisplayClient.is_click_through_active() -> bool`

Query the currently active placement backend's real capabilities, post-startup.

##### `wbb.display.list_displays() -> list[DisplayBounds]`

Enumerates monitors via `SDL_GetDisplayBounds`. Safe to call before any window exists. `DisplayBounds` is a frozen dataclass: `index, x, y, width, height`. `x`/`y` are in SDL's single global virtual-desktop coordinate space (a monitor to the left of your primary can have a negative `x`), which is also what every placement backend expects. This is why `position=`/`monitor=` exist, so you don't have to compute global coordinates yourself.

**Placement backend chain** (`wbb.display.placement`, not typically used directly): tried in order. KWin D-Bus scripting first (works on Plasma regardless of X11/Wayland; requires `pydbus`), then X11/EWMH (`_NET_WM_STATE_ABOVE` plus `XMoveWindow`/Shape extension for click-through; requires `python-xlib`; naturally inert under Wayland), then a terminal no-op fallback that logs once and makes `always_on_top`/`position`/`click_through` silently do nothing. Each backend is required to fail cleanly rather than raise. An unexpected exception is caught and treated as a declined activation.

## Examples

The `examples/` directory in the source repository is the practical reference for composing the primitives above:

- **`01_display_with_filter.py`**: `BrowserBridge` + `DisplayClient` with a filter pipeline (built-in `colorize`/`crop` chained with a user-defined vignette).
- **`02_monitor_and_react.py`**: polls frames, computes a pixel-diff fraction over a region, and triggers a `click()` plus a saved screenshot on change. Graceful Ctrl-C shutdown via signal handlers.
- **`03_1_wait_for_selector.py`**: `wait_for_selector()` and `click_element()` in place of fixed sleeps.
- **`03_multi_step_automation.py`**: a full navigate, type, click, eval, screenshot pipeline, including selector and text-fallback clicking on the same page.
- **`04_cross_process.py`**: one process renders into a named `FrameBuffer` (`attach=False`), a second process attaches (`attach=True`) and displays it. Run as `renderer`/`viewer` from two terminals.
- **`05_record_to_ffmpeg.py`**: pipes raw RGBA frames into an `ffmpeg` subprocess at a fixed FPS with real-time pacing.
- **`06_combined_scenario.py`**: one `FrameBuffer` shared concurrently by a live `DisplayClient`, a PNG-snapshot recorder, and a change monitor.
- **`07_display_multiple_sites_with_pool.py`**: `BrowserPool` driving several sites at once, composited into a single grid and shown through one `DisplayClient`.

## License

MIT. See `LICENSE`.

## Contributing

Issues and pull requests welcome. For non-trivial changes, open an issue first to discuss the approach.

## personal notes
pip install "setuptools>=68" wheel "numpy>=1.24" "websockets>=12.0" "Pillow>=10.0" "aiohttp>=3.9" "PyGObject>=3.50" "PyTurboJPEG>=1.7"
rm -rf dist/ build/ *.egg-info
update version in pyproject.toml
python -m build
twine check dist/*
twine upload dist/*
git tag v0.1.1
git push origin v0.1.1
