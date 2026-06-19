"""
python -m wbb — thin CLI demonstration consumer of the wbb public API.

Usage
-----
::

    # Display a page in a window (requires pygame)
    python -m wbb display https://example.com

    # Run a user scenario script
    python -m wbb script path/to/scenario.py [--url URL] [--width W] [--height H]

    # Headless: save one screenshot and exit
    python -m wbb screenshot https://example.com output.png

The CLI does not implement any scenario logic; it only bootstraps the
library primitives and hands them to the user script or the built-in demo
commands.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m wbb",
        description="WebView Buffer Bridge — scriptable headless browser buffer",
    )
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--buffer-name", default="wbb_default")

    sub = p.add_subparsers(dest="command", required=True)

    # display
    d = sub.add_parser("display", help="Display a URL in a window (requires pygame)")
    d.add_argument("url")
    d.add_argument("--quality", type=int, default=80)

    # screenshot
    s = sub.add_parser("screenshot", help="Save a screenshot and exit")
    s.add_argument("url")
    s.add_argument("output", help="Output file path (PNG, JPEG, …)")
    s.add_argument("--wait", type=float, default=2.0, help="Seconds to wait after load")

    # script
    sc = sub.add_parser("script", help="Run a user scenario script")
    sc.add_argument("script_path", metavar="SCRIPT")
    sc.add_argument("--url", default="about:blank")

    return p


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


async def _cmd_display(args: argparse.Namespace) -> None:
    from wbb import BrowserBridge, DisplayClient, FrameBuffer  # noqa: PLC0415

    buf = FrameBuffer(args.buffer_name, args.width, args.height)
    try:
        async with BrowserBridge(buf, width=args.width, height=args.height) as br:
            await br.navigate(args.url)
            display = DisplayClient(buf, title=f"wbb — {args.url}")
            await display.run_async()
    finally:
        buf.close()
        buf.unlink()


async def _cmd_screenshot(args: argparse.Namespace) -> None:
    from wbb import BrowserBridge, FrameBuffer  # noqa: PLC0415

    buf = FrameBuffer(args.buffer_name, args.width, args.height)
    try:
        async with BrowserBridge(buf, width=args.width, height=args.height) as br:
            await br.navigate(args.url)
            # await br.wait_for_load()
            await asyncio.sleep(args.wait)
            frame = await buf.next_frame()
            frame = buf.read()
            frame.save(args.output)
            print(f"Saved {args.width}×{args.height} screenshot → {args.output}")
            del frame  # release the zero-copy view before the buffer closes
    finally:
        buf.close()
        buf.unlink()


async def _cmd_script(args: argparse.Namespace) -> None:
    from wbb import BrowserBridge, DisplayClient, Frame, FrameBuffer  # noqa: PLC0415
    from wbb import filters  # noqa: PLC0415

    path = Path(args.script_path).resolve()
    if not path.exists():
        sys.exit(f"Script not found: {path}")

    # Pre-initialise objects the script can use via wbb_buffer / wbb_browser.
    # wbb_browser is NOT started — the script owns its lifecycle, same as
    # every BrowserBridge in examples/ (`async with BrowserBridge(...) as br`).
    # This avoids launching a Chrome process the script may never touch if
    # it builds its own objects instead, which is equally valid.
    buf = FrameBuffer(args.buffer_name, args.width, args.height)
    br = BrowserBridge(buf, width=args.width, height=args.height)

    try:
        spec = importlib.util.spec_from_file_location("_wbb_user_script", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)

        # Inject library primitives into the module namespace
        module.wbb_buffer = buf  # type: ignore[attr-defined]
        module.wbb_browser = br  # type: ignore[attr-defined]
        module.wbb_url = args.url  # type: ignore[attr-defined]
        module.BrowserBridge = BrowserBridge  # type: ignore[attr-defined]
        module.FrameBuffer = FrameBuffer  # type: ignore[attr-defined]
        module.Frame = Frame  # type: ignore[attr-defined]
        module.DisplayClient = DisplayClient  # type: ignore[attr-defined]
        module.filters = filters  # type: ignore[attr-defined]

        spec.loader.exec_module(module)

        # If the script defines an async main(), call it; otherwise it ran at import
        if hasattr(module, "main") and asyncio.iscoroutinefunction(module.main):
            await module.main()
    finally:
        # br.stop() is a no-op (returns immediately) if the script never
        # started it — see BrowserBridge.stop()'s _running guard.
        await br.stop()
        buf.close()
        buf.unlink()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dispatch = {
        "display": _cmd_display,
        "screenshot": _cmd_screenshot,
        "script": _cmd_script,
    }

    try:
        asyncio.run(dispatch[args.command](args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
