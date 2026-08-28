"""
10_overlay_above_fullscreen.py — a click-through overlay that stays
visible over a fullscreen application.

The one thing to copy from this file is `window_type=`. Always-on-top
alone is not enough on KDE: `_NET_WM_STATE_ABOVE` puts the window in
KWin's AboveLayer, and a focused fullscreen window is promoted to
ActiveLayer, which is higher. `WindowType.CRITICAL_NOTIFICATION` asks
for CriticalNotificationLayer instead, which sits above ActiveLayer.
On any non-KDE window manager it degrades to a plain notification
window, and off X11 it is a no-op.

Note that notification-type windows do not receive keyboard focus on
most window managers. That is what you want for an overlay and wrong
for anything that needs key input.
"""

import asyncio

from wbb import BrowserBridge, DisplayClient, FrameBuffer, WindowType, centered_position
from wbb.display import diagnose_overlay
from _pages import data_url

# This is the browser viewport as well as the window size: the page is
# rendered at exactly W x H, not scaled down to fit. A real website in a
# 480x140 viewport shows the top-left corner of its layout and nothing
# else. To overlay a real page, either render it at a sensible size and
# crop with `filters.crop(...)`, or scale it in the page itself
# (`Emulation.setPageScaleFactor` via `br.send_cdp`, or CSS `zoom`).
W, H = 480, 140

CLOCK = data_url(
    "<body style='margin:0;background:#101418;color:#e6edf3;"
    "font:600 34px/140px system-ui;text-align:center'>"
    "<span id=t>--:--:--</span>"
    "<script>setInterval(()=>t.textContent="
    "new Date().toLocaleTimeString(),250)</script></body>"
)


async def main() -> None:
    # Things no client window can work around — worth checking once at
    # startup rather than wondering later why the overlay vanished.
    for issue in diagnose_overlay():
        print(f"[{issue.key}] {issue.message}\n    -> {issue.remedy}\n")

    buf = FrameBuffer("overlay", W, H)
    try:
        async with BrowserBridge(buf, width=W, height=H, screencast_max_fps=10) as br:
            await br.navigate(CLOCK)

            display = DisplayClient(
                buf,
                title="wbb overlay",
                wm_class="wbb-overlay",
                borderless=True,
                always_on_top=True,
                click_through=True,
                window_type=WindowType.CRITICAL_NOTIFICATION,
                # Monitor-relative, so this lands correctly whatever the
                # monitor's resolution is. Absolute coordinates put the
                # window off-screen on a mixed-DPI multi-monitor setup.
                position=centered_position(W, H, monitor=0),
                monitor=0,
            )
            await display.run_async()
    finally:
        buf.close()
        buf.unlink()


if __name__ == "__main__":
    asyncio.run(main())
