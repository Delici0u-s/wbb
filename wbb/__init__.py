"""
wbb — WebView Buffer Bridge

Renders a live website into an off-screen shared-memory pixel buffer and
exposes that buffer as a composable, scriptable async Python primitive.

Public surface::

    from wbb import BrowserBridge, FrameBuffer, Frame, DisplayClient
    from wbb import filters
"""

from wbb.browser import BrowserBridge
from wbb.buffer import FrameBuffer
from wbb.frame import Frame
from wbb.display import DisplayClient
from wbb import filters

__all__ = [
    "BrowserBridge",
    "FrameBuffer",
    "Frame",
    "DisplayClient",
    "filters",
]

__version__ = "0.1.0"
