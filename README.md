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

## Requirements

- Python 3.10+
- Chrome or Chromium installed on the host
  (override path with `CHROME_PATH` env var)
- `numpy`, `Pillow`, `websockets`, `aiohttp` (auto-installed)
- `pygame` only if you use `DisplayClient` (`pip install wbb[display]`)

## Installation

```bash
pip install wbb                   # core
pip install "wbb[display]"        # + pygame for DisplayClient
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
| `write(rgba)` | Write an H×W×4 uint8 array; returns the new `frame_id`. Called internally by `BrowserBridge`. |
| `read()` | Return the latest `Frame` as a zero-copy view. |
| `async next_frame()` | Block until the next new frame, then return it. |
| `async for frame in buf` | Yield one `Frame` per rendered frame, indefinitely. |
| `close()` | Release memory mappings. |
| `unlink()` | Destroy OS-level segments. Only the creating process should call this. |

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
| `copy()` | Return a writable copy of the array. |
| `crop(x, y, w, h)` | Zero-copy sub-region view. Returns a new `Frame`. |
| `save(path, format=None)` | Save to file (PNG, JPEG, …). |
| `frame_id` | Monotonically increasing integer. |
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

**Navigation**

```python
await br.navigate(url)
await br.reload(ignore_cache=False)
await br.wait_for_load(timeout=30.0)
await br.set_viewport(width, height)
result = await br.eval("document.title", await_promise=False)
```

**Input** (requires `enable_input=True`, which is the default)

```python
await br.click(x, y, button="left")
await br.move(x, y)
await br.scroll(x, y, delta_x=0, delta_y=100)
await br.key("Enter")       # DOM key name
await br.type("hello")      # inserts text
```

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
    filters=None,          # list of Filter callables
    on_mouse_event=None,   # (event_type, x, y, button) -> Any
    on_key_event=None,     # (event_type, key_name) -> Any
)
```

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
    filters.crop(0, 0, 640, 360),   # top-left quadrant
    filters.scale(1280, 720),        # scale back up
    filters.grayscale(),
    filters.blur(radius=2),
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

If the script defines an `async def main()`, the runner calls it. Otherwise
the script executes at import time.

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

---

## Examples

| Script | What it demonstrates |
|---|---|
| `01_display_with_filter.py` | DisplayClient + user-defined vignette filter |
| `02_monitor_and_react.py` | Pixel diff loop → conditional click + screenshot |
| `03_multi_step_automation.py` | Navigate → type → eval → screenshot pipeline |
| `04_cross_process.py` | BrowserBridge in process A, DisplayClient in process B |
| `05_record_to_ffmpeg.py` | Headless recording via ffmpeg pipe, no window |
| `06_combined_scenario.py` | Display + record + monitor as concurrent async tasks |

---

## Architecture notes

**No Playwright, Puppeteer, or CEF.** wbb drives Chrome directly via the
Chrome DevTools Protocol (CDP) WebSocket. This makes it pip-installable
everywhere and lets it fit inside a single small package.

**Screencast, not polling.** `Page.startScreencast` pushes JPEG frames as
they are rendered. There is no polling loop and no screenshot RPC overhead.

**Double-buffered shared memory.** Two segments alternate; readers always
see a complete frame. The metadata segment (8 bytes) is the only
synchronisation point. Cross-process reading requires no locking.

**Decoupled layers.** Render, buffer, and display each run independently.
A slow display does not drop render frames; a slow render does not block
the display. User tasks run alongside both without any additional glue.

---

## License

MIT
