"""
wbb — WebView Buffer Bridge

Renders a live website into an off-screen shared-memory pixel buffer and
exposes that buffer as a composable, scriptable async Python primitive.

Public surface::

    from wbb import BrowserBridge, FrameBuffer, Frame, DisplayClient
    from wbb import filters

Optional-dependency note
------------------------
`DisplayClient` and everything else under `wbb.display` need PySDL2,
which is an *optional* extra (`pip install 'wbb[display]'`). They are
therefore resolved lazily via PEP 562 `__getattr__`: importing `wbb`,
using `BrowserBridge`/`FrameBuffer`/`filters`, and running
`python -m wbb screenshot` all work with the base install. Touching
`wbb.DisplayClient` without PySDL2 raises ImportError at that point,
naming the extra — instead of `import wbb` failing outright, which is
what happened before.
"""

from typing import TYPE_CHECKING, Any

from wbb.browser import BrowserBridge
from wbb.buffer import FrameBuffer
from wbb.frame import Frame
from wbb.pool import BrowserPool
from wbb import filters

if TYPE_CHECKING:  # for type checkers only; no runtime import
    from wbb.display import (
        Anchor,
        DisplayClient,
        WindowType,
        centered_position,
        diagnose_overlay,
        preinit_alpha,
    )

_LAZY_DISPLAY = {
    "DisplayClient",
    "Anchor",
    "WindowType",
    "diagnose_overlay",
    "centered_position",
    "preinit_alpha",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_DISPLAY:
        try:
            import wbb.display as _display
        except ImportError as exc:
            raise ImportError(
                f"wbb.{name} requires the optional display extra: "
                f"pip install 'wbb[display]'  (underlying error: {exc})"
            ) from exc
        return getattr(_display, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__))


__all__ = [
    "BrowserBridge",
    "FrameBuffer",
    "Frame",
    "DisplayClient",
    "filters",
    "BrowserPool",
    "Anchor",
    "WindowType",
    "diagnose_overlay",
    "centered_position",
    "preinit_alpha",
]

__version__ = "0.1.5.3"
