"""
wbb.display — SDL2-backed window client for a FrameBuffer.

Import layering
---------------
PySDL2 is an optional extra. The submodules that do *not* need it —
`geometry`, `window_type`, `overlay_diag`, `x11_props` — are imported
eagerly so they can be used (and tested) on a machine with no SDL and no
display at all. `DisplayClient`, `DisplayBounds` and `list_displays`
pull in `sdl2` and are resolved lazily via PEP 562, raising an
ImportError that names the extra at the point of use rather than at
`import wbb`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .geometry import Anchor, anchored_origin
from .overlay_diag import OverlayIssue, diagnose_overlay
from .window_type import WindowType

if TYPE_CHECKING:
    from ._window import (
        DisplayBounds,
        centered_position,
        list_displays,
        preinit_alpha,
    )
    from .client import DisplayClient, WindowPosition

_LAZY = {
    "DisplayClient": ".client",
    "WindowPosition": ".client",
    "DisplayBounds": "._window",
    "list_displays": "._window",
    "centered_position": "._window",
    "preinit_alpha": "._window",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        mod = importlib.import_module(module, __name__)
    except ImportError as exc:
        raise ImportError(
            f"wbb.display.{name} requires the optional display extra: "
            f"pip install 'wbb[display]'  (underlying error: {exc})"
        ) from exc
    return getattr(mod, name)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "DisplayClient",
    "WindowPosition",
    "DisplayBounds",
    "list_displays",
    "centered_position",
    "preinit_alpha",
    "Anchor",
    "anchored_origin",
    "WindowType",
    "OverlayIssue",
    "diagnose_overlay",
]
