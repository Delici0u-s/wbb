# wbb — WebView Buffer Bridge

Renders a live website into an off-screen shared-memory pixel buffer and
exposes that buffer as a composable, scriptable async Python primitive.

```
[ headless Chrome ] ──CDP screencast──▶ [ BrowserBridge ]
                                               │
                                         JPEG decode
                                         RGBA write
                                               │
                                               ▼
                                   [ FrameBuffer / SHM ]
                                               │
                                    zero-copy numpy view
                                    filter pipeline
                                               │
                                   ┌───────────┴───────────┐
                                   ▼                       ▼
                            [ DisplayClient ]       [ user script ]
                              SDL2 window           any consumer
```

[github repository](https://github.com/Delici0u-s/wbb)

## Requirements

- Python 3.10+
- Chrome or Chromium installed on the host
  (override path with `CHROME_PATH` env var)
- `numpy`, `Pillow`, `websockets`, `aiohttp` (auto-installed)
- `pygame` only if you use `DisplayClient` (`pip install wbb[display]`)
- `scipy` optional — `filters.blur(radius > 1)` uses it if present and
  falls back to a cheap roll-average otherwise

## Installation

```bash
pip install wbb                   # core
pip install "wbb[display]"        # + pygame for DisplayClient
pip install "wbb[fast-jpeg]"      # + PyTurboJPEG for faster computation.
```

## Quick start

```python
import asyncio
from wbb import BrowserBridge, FrameBuffer

async def main():
    buf = FrameBuffer("my_buf", 1280, 720)
    async with BrowserBridge(buf) as br:
        await br.navigate("https://example.com")
        frame = await buf.next_frame()
        frame.save("screenshot.png")
        del frame  # release the zero-copy view before the buffer closes
    buf.close()
    buf.unlink()

asyncio.run(main())
```

---

## Public API

### `FrameBuffer`

The central primitive. Owns two named shared-memory segments that form a
double-buffer; a metadata segment tracks which side is current.

```python
FrameBuffer(name, width, height, *, attach=False)
```

| Method / property | Description |
|---|---|
| `write(rgba)` | Write an H×W×4 uint8 array into the inactive buffer, then flip. Returns the new `frame_id`. Called internally by `BrowserBridge`. |
| `read()` | Return the latest `Frame` as a zero-copy, read-only view. |
| `async next_frame(timeout=5.0)` | Wait up to `timeout` seconds for a new frame; on timeout, returns the latest frame via `read()` instead of raising. |
| `async for frame in buf` | Yield one `Frame` per rendered frame, indefinitely (built on `next_frame`). |
| `close()` | Release this process's memory mappings. |
| `unlink()` | Destroy the underlying OS-level segments. Only the creating process should call this. |
| `with FrameBuffer(...) as buf` | Context manager: `close()`s on exit, and also `unlink()`s if this process created the buffer (`attach=False`). |

**Lifetime contract — read this before debugging a hang:**
`read()` and async iteration return **zero-copy views** into shared
memory. CPython will not unmap memory while any array still holds a live
buffer-protocol export on it — the same rule as `mmap` generally. In
practice: **drop every `Frame` (and anything derived from one, e.g. via
`.crop()`) before calling `close()`** — reassign the variable, let it go
out of scope, or `del frame`. If you need the pixel data to outlive the
buffer, call `frame.copy()` first to detach it entirely. `close()` itself
is best-effort: if a view is still outstanding it logs and returns rather
than raising, and the segment is freed once that reference is dropped.
`unlink()` always succeeds regardless, since it only removes the *name*.

**Cross-process attach:**

```python
# Process A — creates the buffer
buf = FrameBuffer("shared", 1280, 720, attach=False)

# Process B — connects to the same buffer, no BrowserBridge needed
buf = FrameBuffer("shared", 1280, 720, attach=True)
frame = buf.read()
```

---

### `Frame`

A snapshot of one rendered frame.

```python
Frame(data, width, height, frame_id, timestamp)
```

| Method | Description |
|---|---|
| `data` | `np.ndarray` H×W×4 uint8 RGBA. Read-only view into shared memory. |
| `copy()` | Return a writable, detached copy of the array. |
| `crop(x, y, w, h)` | Zero-copy sub-region view. Returns a new `Frame` (same `frame_id`/`timestamp`). |
| `save(path, format=None)` | Save to file (PNG, JPEG, …); format inferred from extension unless given. |
| `frame_id` | Monotonically increasing integer, unique within a `FrameBuffer` session. |
| `timestamp` | `time.monotonic()` at write time. |

---

### `BrowserBridge`

Owns the headless Chrome process and CDP connection.

```python
BrowserBridge(
    buffer,
    *,
    width=1280,
    height=720,
    screencast_quality=80,
    screencast_max_fps=30,
    enable_input=True,
    headless_args=None,
)
```

**Lifecycle**

```python
async with BrowserBridge(buf) as br:
    ...
# or manually
br = BrowserBridge(buf)
await br.start()
...
await br.stop()
```

`stop()` is a no-op (returns immediately) if `start()` was never called.

**Navigation**

```python
await br.navigate(url)                      # does not block on load — see note below
await br.reload(ignore_cache=False)
await br.set_viewport(width, height)
result = await br.eval("document.title", await_promise=False)
found = await br.wait_for_selector("#some-id", timeout=10.0, poll_interval=0.15)
```

> **Note:** `navigate()` sends `Page.navigate` and returns immediately;
> it does **not** wait for the load event. Pace multi-step automation
> with `wait_for_selector()`, `click_element()`'s own polling, or an
> explicit `asyncio.sleep(...)` between steps — see
> `examples/03_multi_step_automation.py`.

**Input** (requires `enable_input=True`, which is the default; all of
these raise `RuntimeError` if input was disabled)

```python
await br.click(x, y, button="left")
await br.move(x, y)
await br.scroll(x, y, delta_x=0, delta_y=100)
await br.key("Enter")          # DOM key name; see caveat below
await br.type("hello")         # inserts text via Input.insertText
found = await br.click_element(selector=None, text=None, nth=0,
                                scroll_into_view=False, timeout=5.0,
                                poll_interval=0.15, button="left")
```

> **`key()` is known to be flaky.** It dispatches a raw
> `Input.dispatchKeyEvent` and Chrome doesn't always treat it as
> Wikipedia's search box does, for instance — sending `"\r"` as a key
> reliably submits a form where `"Enter"` does not, in practice. If a
> key event doesn't do what you expect, try the alternative character,
> or fall back to `eval()`-driven form submission.

`click_element` resolves an element by CSS *selector* (tried first; if it
matches anything, the *nth* match is used directly — no redirect to a
clickable ancestor) or, failing that, by case-insensitive *text*
substring match against an element's own direct text nodes, with each
text match resolved to its nearest clickable ancestor (`a`, `button`,
`input`, `select`, `textarea`, `[role="button"]`, `[onclick]`, `summary`,
`label`) and de-duplicated. It polls every `poll_interval` seconds up to
`timeout`, then clicks the resolved bounding-box center via the same path
as `click()`. Returns `False` on timeout rather than raising; raises
`ValueError` if neither `selector` nor `text` is given.

**Frame access**

```python
frame = br.buffer.read()    # latest frame, immediately
async for frame in br.frames():
    ...                     # one Frame per screencast push
```

**Event hooks**

```python
br.on("frame",    lambda frame: ...)      # every rendered frame
br.on("navigate", lambda url: ...)        # page navigation
br.on("load",     lambda: ...)            # load event fired
br.on("console",  lambda level, args: ...)
br.on("error",    lambda description: ...)
```

Callbacks may be plain functions or coroutine functions.

---

### `DisplayClient`

Optional SDL2 window. Requires `pygame`.

```python
DisplayClient(
    buffer,
    *,
    title="wbb",
    filters=None,           # list of Filter callables
    on_mouse_event=None,    # (event_type, x, y, button) -> Any
    on_key_event=None,      # (event_type, key_name) -> Any
    window_size=None,       # (width, height); defaults to buffer size
)
```

`window_size` is useful when a filter changes the frame's output
dimensions (e.g. a crop) — set it to match so the SDL surface isn't
mismatched against the buffer's native size.

```python
# Blocking
display.run()

# Async task (composable with other coroutines)
task = asyncio.create_task(display.run_async())
display.stop()   # signal shutdown from another coroutine
await task
```

Input forwarding to the browser:

```python
def on_mouse(kind, x, y, button):
    if kind == "down":
        asyncio.create_task(br.click(x, y))

display = DisplayClient(buf, on_mouse_event=on_mouse)
```

---

### `filters` module

Every filter is `(np.ndarray) -> np.ndarray` — plain callables, fully
composable. Built-ins return factory functions so parameters are explicit:

```python
from wbb import filters

pipeline = [
    filters.crop(0, 0, 640, 360),    # top-left quadrant
    filters.scale(1280, 720),        # scale back up
    filters.grayscale(),
    filters.blur(radius=2),          # uses scipy if installed, else a cheap fallback
    filters.brightness(+20),
    filters.contrast(1.2),
    filters.colorize(r=1.1, g=0.9, b=0.9),
    filters.flip(horizontal=True),
]

# Combine into one callable
f = filters.chain(*pipeline)
result = f(frame.data)
```

User-defined filters plug in identically:

```python
def my_filter(frame: np.ndarray) -> np.ndarray:
    return frame.copy()   # or any transformation

pipeline = [filters.grayscale(), my_filter]
display = DisplayClient(buf, filters=pipeline)
```

---

## Scriptability model

A user scenario is a plain `.py` file that imports `wbb` and composes its
primitives. The library places no constraints on structure or complexity.

### Using the CLI script runner

```bash
python -m wbb script my_scenario.py --url https://example.com
```

The script receives pre-initialized objects as module-level names:
`wbb_buffer`, `wbb_browser`, `wbb_url`, plus all public symbols
(`BrowserBridge`, `FrameBuffer`, `Frame`, `DisplayClient`, `filters`).

`wbb_browser` is **not** started for you — the script owns its lifecycle,
exactly like every `BrowserBridge` in `examples/` (`async with
BrowserBridge(...) as br:` or manual `start()`/`stop()`). This avoids
launching a Chrome process the script may never touch if it builds its
own objects instead, which is equally valid. If the script never starts
`wbb_browser`, the runner's cleanup `stop()` call is a safe no-op.

If the script defines an `async def main()`, the runner calls it.
Otherwise the script executes at import time.

### CLI commands

```bash
# Open a URL in a window (requires pygame)
python -m wbb display https://example.com

# Save a screenshot and exit
python -m wbb screenshot https://example.com output.png --wait 2.0

# Run a user scenario script
python -m wbb script path/to/scenario.py --url https://example.com \
    --width 1920 --height 1080
```

Shared flags (`--width`, `--height`, `--buffer-name`) belong to the top-
level parser and go *before* the subcommand name.

---

## Examples

| Script | What it demonstrates |
|---|---|
| `initial_test.py` | Minimal smoke test: navigate, grab one frame, save it |
| `01_display_with_filter.py` | DisplayClient + user-defined vignette filter |
| `01_1_display_with_filter.py` | Same, with a normalized crop region and a matching `window_size` |
| `02_monitor_and_react.py` | Pixel diff loop → conditional click + screenshot, with graceful Ctrl-C shutdown |
| `03_multi_step_automation.py` | Navigate → type → key → `click_element` (selector and text fallback) → screenshot pipeline |
| `04_cross_process.py` | BrowserBridge in process A, DisplayClient in process B |
| `05_record_to_ffmpeg.py` | Headless recording via ffmpeg pipe, real-time frame pacing, no window |
| `06_combined_scenario.py` | Display + record + monitor as concurrent async tasks sharing one buffer |

---

## Architecture notes

**No Playwright, Puppeteer, or CEF.** wbb drives Chrome directly via the
Chrome DevTools Protocol (CDP) WebSocket. This makes it pip-installable
everywhere and lets it fit inside a single small package.

**Screencast, not polling.** `Page.startScreencast` pushes JPEG frames as
they are rendered. There is no polling loop and no screenshot RPC overhead.

**Double-buffered shared memory.** Two segments alternate; readers always
see a complete frame. The metadata segment (active index + frame_id +
timestamp) is the only synchronisation point. Cross-process reading
requires no locking — see `FrameBuffer`'s lifetime contract above for the
one thing that *does* require care: dropping zero-copy views before
`close()`.

**Decoupled layers.** Render, buffer, and display each run independently.
A slow display does not drop render frames; a slow render does not block
the display. User tasks run alongside both without any additional glue.

**Resource-tracker hygiene.** `wbb._shm.ShmSegment` wraps
`multiprocessing.shared_memory.SharedMemory` and immediately unregisters
attached (non-owning) segments from the process's resource tracker, so
an attaching/reader process never races the owning process's `unlink()`
on exit (see `wbb/_shm.py` for the CPython issue this works around).

---

## License

MIT

## personal notes
pip install "setuptools>=68" wheel "numpy>=1.24" "websockets>=12.0" "Pillow>=10.0" "aiohttp>=3.9" "pygame>2.5"
rm -rf dist/ build/ *.egg-info
python -m build
twine check dist/*
twine upload dist/*
git tag v0.1.1
git push origin v0.1.1
