"""
Windows smoke test for the win32 placement backend and layered alpha.

No browser involved: draws a gradient whose alpha fades left to right
into an SDL window, moves it around, toggles always-on-top and
click-through, and prints what the readback reports at each step.

    python examples/tools_win32_check.py

Expected:
  * a borderless 400x200 window, left edge opaque blue, fading to fully
    see-through on the right, with the desktop visible behind it
  * positions printed as requested == actual
  * during the click-through phase, clicks land on whatever is behind it
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np

from wbb.display._window import SDLWindow, list_displays
from wbb.display.placement import select_backend


def gradient(w: int, h: int) -> np.ndarray:
    a = np.linspace(255, 0, w, dtype=np.float32)[None, :].repeat(h, 0)
    img = np.zeros((h, w, 4), np.uint8)
    img[..., 2] = 255  # blue
    img[..., 3] = a.astype(np.uint8)
    # Premultiply, as DisplayClient does for alpha windows.
    img[..., :3] = (img[..., :3].astype(np.uint16) * img[..., 3:4] // 255).astype(np.uint8)
    return img


def pump(win: SDLWindow, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        for ev in win.poll_events():
            if ev["kind"] == "mouse" and ev["event_type"] == "down":
                print(f"   click received at ({ev['x']:.0f}, {ev['y']:.0f})")
        time.sleep(0.01)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if sys.platform != "win32":
        print("This check is for Windows only.")
        return 1

    d = list_displays()[0]
    print(f"display 0: {d}")

    win = SDLWindow(400, 200, title="wbb win32 check", wm_class="wbb-check",
                    borderless=True, alpha=True, position=(d.x + 100, d.y + 100))
    handle = win.native_handle()
    print(f"subsystem={handle.subsystem} hwnd=0x{handle.win32_hwnd or 0:x} "
          f"alpha_active={win.alpha_active}")
    win.push_frame(gradient(400, 200))

    placement = select_backend(handle, wm_class="wbb-check")
    print(f"placement backend: {placement.name}")

    placement.set_above(True)
    for target in [(d.x + 100, d.y + 100), (d.x + 600, d.y + 300), (d.x + 50, d.y + 500)]:
        placement.set_position(*target, 400, 200)
        pump(win, 0.3)
        print(f"requested {target} -> actual {placement.actual_position()}")
        pump(win, 1.0)

    print("click-through ON for 5 s: clicks should reach what is behind the window")
    print("   ok:", placement.set_click_through(True))
    pump(win, 5.0)
    print("click-through OFF for 5 s: clicks on the opaque (left) part should print")
    print("   ok:", placement.set_click_through(False))
    pump(win, 5.0)

    win.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
