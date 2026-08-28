"""
overlay_diag.py — report the environment conditions that bury an
always-on-top overlay even when its window type and stacking are correct.

None of these are things a client window can fix. They are properties of
the compositor or of the *other* application, so this module reports
rather than repairs. It is pure filesystem/env inspection: no D-Bus, no
X11, no imports beyond the stdlib, and safe to call before any window
exists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True, slots=True)
class OverlayIssue:
    key: str
    message: str
    remedy: str


def _kwinrc_value(section: str, key: str) -> Optional[str]:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    path = Path(base) / "kwinrc"
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    current = ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
        elif current == section and "=" in line:
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip()
    return None


def diagnose_overlay() -> list[OverlayIssue]:
    """Conditions that will hide an overlay regardless of its window type."""
    issues: list[OverlayIssue] = []

    blocking = _kwinrc_value("Compositing", "WindowsBlockCompositing")
    # KWin's default is "true" (allow apps to block compositing) and the
    # key is often absent from kwinrc entirely when unchanged.
    if blocking is None or blocking.lower() != "false":
        issues.append(
            OverlayIssue(
                key="block-compositing",
                message=(
                    "KWin allows applications to block compositing "
                    "(kwinrc [Compositing] WindowsBlockCompositing is not 'false'). "
                    "A fullscreen window that sets _NET_WM_BYPASS_COMPOSITOR=1 — "
                    "most games do — suspends the compositor, and while compositing "
                    "is off nothing can be drawn over it. Window type and stacking "
                    "layer are irrelevant in that state."
                ),
                remedy=(
                    "System Settings > Display and Monitor > Compositor > "
                    "uncheck 'Allow applications to block compositing', or set the "
                    "game to borderless-windowed instead of exclusive fullscreen."
                ),
            )
        )

    if os.environ.get("GAMESCOPE_WAYLAND_DISPLAY") or os.environ.get("GAMESCOPE_XWAYLAND"):
        issues.append(
            OverlayIssue(
                key="gamescope",
                message=(
                    "A gamescope nested compositor is present. A window created "
                    "against the outer session is not composited by gamescope and "
                    "will not appear over content gamescope renders."
                ),
                remedy=(
                    "Run wbb against gamescope's own nested X display "
                    "(DISPLAY=:<n>) and pass gamescope_overlay=True, or accept "
                    "that the overlay only covers the outer session."
                ),
            )
        )

    session = os.environ.get("XDG_SESSION_TYPE", "")
    driver = os.environ.get("SDL_VIDEODRIVER", "")
    if session == "wayland" and driver in ("", "x11"):
        issues.append(
            OverlayIssue(
                key="xwayland",
                message=(
                    "Wayland session, but SDL is using (or defaulting to) the x11 "
                    "video driver, so the window is an XWayland client. Native "
                    "Wayland windows of other applications stack against it through "
                    "XWayland's proxy surface, and window_type= is honoured through "
                    "the X11 property path."
                ),
                remedy=(
                    "This is the supported configuration for window_type= and "
                    "click_through=. Setting SDL_VIDEODRIVER=wayland turns both "
                    "into documented no-ops."
                ),
            )
        )

    return issues
