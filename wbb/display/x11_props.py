"""
x11_props.py — minimal, dependency-optional X11 property writes.

Everything here is best-effort: no python-xlib, no X11 subsystem, no
reachable display, or a server that rejects the request all return
False. Nothing in this module raises.

Only used for properties that must be on the window *before* the
window manager starts managing it (see window_type.py) and for the
one-shot visual query used by alpha.py. Nothing here is on the frame
hot path.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator, Optional, Sequence

log = logging.getLogger(__name__)


@contextmanager
def x11_connection(display_name: Optional[str] = None) -> Iterator[Optional[object]]:
    """Yield a python-xlib Display, or None if one cannot be opened.

    `display_name` overrides $DISPLAY. That parameter exists for nested
    X servers (gamescope runs one); it is not needed for the ordinary
    XWayland/X11 case.
    """
    try:
        from Xlib import display as xdisplay  # noqa: PLC0415
    except ImportError:
        log.debug("x11_props: python-xlib not installed")
        yield None
        return

    disp = None
    try:
        disp = xdisplay.Display(display_name)
    except Exception:
        log.debug("x11_props: could not open X display %r", display_name)
        yield None
        return

    try:
        yield disp
    finally:
        try:
            disp.close()
        except Exception:
            pass


def set_atom_list_property(
    window_id: int,
    prop_name: str,
    atom_names: Sequence[str],
    *,
    display_name: Optional[str] = None,
) -> bool:
    """Replace `prop_name` on `window_id` with a 32-bit ATOM list.

    Returns True only if the change was flushed and the server round-trip
    (`sync()`) completed without an error.
    """
    if not atom_names:
        return False
    with x11_connection(display_name) as disp:
        if disp is None:
            return False
        try:
            from Xlib import Xatom  # noqa: PLC0415

            win = disp.create_resource_object("window", window_id)
            prop = disp.intern_atom(prop_name)
            atoms = [disp.intern_atom(n) for n in atom_names]
            win.change_property(prop, Xatom.ATOM, 32, atoms)
            disp.sync()
            return True
        except Exception:
            log.debug("x11_props: failed to set %s on 0x%x", prop_name, window_id, exc_info=True)
            return False


def set_cardinal_property(
    window_id: int,
    prop_name: str,
    values: Sequence[int],
    *,
    display_name: Optional[str] = None,
) -> bool:
    """Replace `prop_name` on `window_id` with a 32-bit CARDINAL list."""
    with x11_connection(display_name) as disp:
        if disp is None:
            return False
        try:
            from Xlib import Xatom  # noqa: PLC0415

            win = disp.create_resource_object("window", window_id)
            prop = disp.intern_atom(prop_name)
            win.change_property(prop, Xatom.CARDINAL, 32, list(values))
            disp.sync()
            return True
        except Exception:
            log.debug("x11_props: failed to set %s on 0x%x", prop_name, window_id, exc_info=True)
            return False


def window_depth(window_id: int, *, display_name: Optional[str] = None) -> Optional[int]:
    """Bit depth of an existing X window, or None if it cannot be read.

    Used by alpha.py to *verify* that the window really got a 32-bit
    ARGB visual, instead of assuming the SDL hint was honoured.
    """
    with x11_connection(display_name) as disp:
        if disp is None:
            return None
        try:
            win = disp.create_resource_object("window", window_id)
            return int(win.get_geometry().depth)
        except Exception:
            return None
