"""
NoPlacement — terminal fallback. Always "activates" (there's nothing
to fail), every call is a no-op. Logs exactly once, at activation
time, so callers who asked for always_on_top/position get a single
clear explanation instead of silent nothing or a per-call spam of
warnings.

The activation warning is the library's main user-facing channel for
"you're missing the dependency that would make placement work." It
therefore does real legwork: detect the desktop/session, check which
optional deps are actually importable *in this interpreter*, name the
right pip extra, and print a command using sys.executable so a user in
a venv gets that venv's pip — not a system Python whose ~/.local
packages wouldn't help the venv anyway (the exact trap that wastes an
afternoon).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shlex
import sys

from ._base import NativeHandle

log = logging.getLogger(__name__)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _is_kde() -> bool:
    xdg = os.environ.get("XDG_CURRENT_DESKTOP", "")
    desktop = os.environ.get("DESKTOP_SESSION", "")
    blob = f"{xdg}:{desktop}".lower()
    return "kde" in blob or "plasma" in blob


def _install_cmd(extra: str) -> str:
    """A copy-pasteable install command targeting THIS interpreter.

    Using sys.executable -m pip guarantees the install lands in the same
    environment that's running wbb (venv, conda, system) — the #1 cause
    of "I installed it and it still doesn't work" is pip going to a
    different interpreter / user site than the one importing wbb.
    """
    return f"{shlex.quote(sys.executable)} -m pip install 'wbb[{extra}]'"


class NoPlacement:
    name = "none"

    def activate(self, handle: NativeHandle) -> bool:
        log.warning(
            "Placement: no backend available (subsystem=%r). always_on_top "
            "/ set_position() will have no effect. %s",
            handle.subsystem,
            self._diagnose(handle),
        )
        return True

    def _diagnose(self, handle: NativeHandle) -> str:
        have_pydbus = _module_available("pydbus")
        have_gi = _module_available("gi")  # PyGObject imports as `gi`
        have_xlib = _module_available("Xlib")  # python-xlib imports as `Xlib`
        kde = _is_kde()

        # KDE (either session type) -> KWin backend, needs pydbus + PyGObject.
        if kde or handle.subsystem == "wayland":
            # On Wayland, KWin is the ONLY option regardless of desktop string.
            missing = []
            if not have_pydbus:
                missing.append("pydbus")
            if not have_gi:
                missing.append("PyGObject")
            if missing:
                return (
                    f"This looks like {'KDE Plasma' if kde else 'a Wayland session'}; "
                    f"placement uses the KWin D-Bus backend, which needs "
                    f"{' + '.join(missing)} (missing here). Install with:\n"
                    f"    {_install_cmd('kde')}\n"
                    f"(or `{sys.executable} -m pip install {' '.join(missing)}` "
                    f"directly). Then make sure KWin scripting is enabled. "
                    + (
                        "On Wayland this is the only mechanism — python-xlib "
                        "will not help, there's no X11 window to act on."
                        if handle.subsystem == "wayland"
                        else ""
                    )
                )
            # Deps present but backend still didn't activate.
            return (
                "pydbus and PyGObject are installed but the KWin backend did "
                "not activate — check that KWin scripting is enabled and that "
                "org.kde.KWin is reachable on the session bus "
                "(`qdbus org.kde.KWin` should respond)."
            )

        # Non-KDE X11 -> EWMH backend, needs python-xlib.
        if handle.subsystem == "x11":
            if not have_xlib:
                return (
                    "This is a non-KDE X11 session; placement uses the "
                    "EWMH backend, which needs python-xlib (missing here). "
                    f"Install with:\n    {_install_cmd('x11')}\n"
                    f"(or `{sys.executable} -m pip install python-xlib`)."
                )
            return (
                "python-xlib is installed but the EWMH backend did not "
                "activate — the window manager may not support "
                "_NET_MOVERESIZE_WINDOW, or no X11 display was reachable."
            )

        # Unknown subsystem.
        return (
            "Unknown windowing subsystem. For KDE Plasma install the 'kde' "
            f"extra ({_install_cmd('kde')}); for other X11 desktops install "
            f"the 'x11' extra ({_install_cmd('x11')})."
        )

    def set_above(self, above: bool) -> None:
        pass

    def set_position(self, x: int, y: int, width: int, height: int) -> None:
        pass

    def supports_position(self) -> bool:
        return False

    def set_click_through(self, enabled: bool) -> bool:
        log.warning(
            "click_through requested but no placement backend is active "
            "(see the earlier 'no backend available' warning); ignoring."
        )
        return False
