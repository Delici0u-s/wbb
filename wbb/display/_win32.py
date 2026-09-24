"""
_win32.py — ctypes bindings for the Win32 pieces wbb needs, and the
per-pixel-alpha presenter built on ``UpdateLayeredWindow``.

Why a second present path exists at all
---------------------------------------
On Windows the SDL2 renderer (Direct3D by default) draws into a normal
top-level window, and DWM composites every normal window *opaque*: the
texture's alpha is written to the back buffer correctly and then thrown
away, so transparent pixels show up black. SDL2 has no per-pixel
transparent window flag (``SDL_WINDOW_TRANSPARENT`` is SDL3-only), and
the GPU-side tricks (DWM blur-behind with an empty region + OpenGL) are
driver-dependent. ``UpdateLayeredWindow`` is the documented, universal
mechanism: the window is marked ``WS_EX_LAYERED`` and every frame is
handed to DWM as a premultiplied BGRA bitmap. It costs one CPU-side
swizzle into a DIB section plus one GDI upload per frame, and it works
on every GPU/driver combination.

Consequences the caller has to know about
-----------------------------------------
* The SDL renderer must NOT be used on a window driven by this
  presenter — once ``UpdateLayeredWindow`` has been called, Windows
  ignores WM_PAINT/swap-chain output for that window. ``SDLWindow``
  therefore creates no renderer at all in this mode.
* The bitmap covers the *whole* window rectangle, including any
  non-client frame, so alpha windows are forced borderless (see
  ``SDLWindow``).
* Layered windows hit-test per pixel: pixels with alpha 0 do not receive
  mouse input and clicks pass through them to whatever is below. On X11
  a fully transparent pixel still eats clicks unless click_through is
  set. This is a real platform difference, not a bug here.
* Frames must be premultiplied. ``DisplayClient._prepare_alpha`` already
  does that whenever ``alpha_active`` is True, which it is for this
  presenter.

Everything here is import-safe on non-Windows: the DLLs are bound on
first use, and ``available()`` reports False off Windows.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

# --- Win32 constants ----------------------------------------------------
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_NOACTIVATE = 0x08000000

LWA_ALPHA = 0x00000002
ULW_ALPHA = 0x00000002
AC_SRC_OVER = 0x00
AC_SRC_ALPHA = 0x01

HWND_TOPMOST = -1
HWND_NOTOPMOST = -2

SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020
SWP_NOOWNERZORDER = 0x0200

BI_RGB = 0
DIB_RGB_COLORS = 0


# --- Structures ----------------------------------------------------------
class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [
        ("BlendOp", ctypes.c_ubyte),
        ("BlendFlags", ctypes.c_ubyte),
        ("SourceConstantAlpha", ctypes.c_ubyte),
        ("AlphaFormat", ctypes.c_ubyte),
    ]


# --- Lazy DLL binding ----------------------------------------------------
class _Api:
    """Holds bound Win32 functions with explicit argtypes/restype.

    Explicit restype matters on 64-bit Python: ctypes' default return
    type is a C int, which silently truncates HWND/HDC/HBITMAP handles
    to 32 bits — the kind of bug that works on one machine and crashes
    on the next.
    """

    def __init__(self) -> None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)  # type: ignore[attr-defined]
        gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)  # type: ignore[attr-defined]

        # SetWindowLongPtrW only exists in the 64-bit user32; on 32-bit
        # Python the pointer-sized variant is a macro for SetWindowLongW.
        get_long = getattr(user32, "GetWindowLongPtrW", None) or user32.GetWindowLongW
        set_long = getattr(user32, "SetWindowLongPtrW", None) or user32.SetWindowLongW
        get_long.argtypes = [wintypes.HWND, ctypes.c_int]
        get_long.restype = ctypes.c_ssize_t
        set_long.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
        set_long.restype = ctypes.c_ssize_t
        self.GetWindowLongPtr = get_long
        self.SetWindowLongPtr = set_long

        self.SetWindowPos = user32.SetWindowPos
        self.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        self.SetWindowPos.restype = wintypes.BOOL

        self.GetWindowRect = user32.GetWindowRect
        self.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.GetWindowRect.restype = wintypes.BOOL

        self.ClientToScreen = user32.ClientToScreen
        self.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        self.ClientToScreen.restype = wintypes.BOOL

        self.IsWindow = user32.IsWindow
        self.IsWindow.argtypes = [wintypes.HWND]
        self.IsWindow.restype = wintypes.BOOL

        self.SetLayeredWindowAttributes = user32.SetLayeredWindowAttributes
        self.SetLayeredWindowAttributes.argtypes = [
            wintypes.HWND,
            wintypes.COLORREF,
            ctypes.c_ubyte,
            wintypes.DWORD,
        ]
        self.SetLayeredWindowAttributes.restype = wintypes.BOOL

        self.UpdateLayeredWindow = user32.UpdateLayeredWindow
        self.UpdateLayeredWindow.argtypes = [
            wintypes.HWND,
            wintypes.HDC,
            ctypes.POINTER(wintypes.POINT),
            ctypes.POINTER(wintypes.SIZE),
            wintypes.HDC,
            ctypes.POINTER(wintypes.POINT),
            wintypes.COLORREF,
            ctypes.POINTER(BLENDFUNCTION),
            wintypes.DWORD,
        ]
        self.UpdateLayeredWindow.restype = wintypes.BOOL

        self.CreateCompatibleDC = gdi32.CreateCompatibleDC
        self.CreateCompatibleDC.argtypes = [wintypes.HDC]
        self.CreateCompatibleDC.restype = wintypes.HDC

        self.DeleteDC = gdi32.DeleteDC
        self.DeleteDC.argtypes = [wintypes.HDC]
        self.DeleteDC.restype = wintypes.BOOL

        self.CreateDIBSection = gdi32.CreateDIBSection
        self.CreateDIBSection.argtypes = [
            wintypes.HDC,
            ctypes.POINTER(BITMAPINFO),
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p),
            wintypes.HANDLE,
            wintypes.DWORD,
        ]
        self.CreateDIBSection.restype = wintypes.HBITMAP

        self.SelectObject = gdi32.SelectObject
        self.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
        self.SelectObject.restype = wintypes.HGDIOBJ

        self.DeleteObject = gdi32.DeleteObject
        self.DeleteObject.argtypes = [wintypes.HGDIOBJ]
        self.DeleteObject.restype = wintypes.BOOL

        self.GdiFlush = gdi32.GdiFlush
        self.GdiFlush.argtypes = []
        self.GdiFlush.restype = wintypes.BOOL


_API: Optional[_Api] = None


def api() -> _Api:
    """Bound Win32 functions. Raises OSError off Windows."""
    global _API
    if _API is None:
        if not IS_WINDOWS:
            raise OSError("Win32 API is only available on Windows")
        _API = _Api()
    return _API


def available() -> bool:
    if not IS_WINDOWS:
        return False
    try:
        api()
        return True
    except Exception:
        log.debug("win32: binding user32/gdi32 failed", exc_info=True)
        return False


def last_error() -> int:
    return int(ctypes.get_last_error())  # type: ignore[attr-defined]


# --- Extended-style helpers ---------------------------------------------
def get_exstyle(hwnd: int) -> int:
    return int(api().GetWindowLongPtr(hwnd, GWL_EXSTYLE))


def set_exstyle(hwnd: int, style: int) -> None:
    a = api()
    a.SetWindowLongPtr(hwnd, GWL_EXSTYLE, style)
    # Style bits are cached by the window manager until the frame is
    # recalculated; SWP_FRAMECHANGED forces that without moving,
    # resizing, restacking or activating the window.
    a.SetWindowPos(
        hwnd,
        None,
        0,
        0,
        0,
        0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED,
    )


# --- RGBA -> BGRA --------------------------------------------------------
def rgba_to_bgra(src: np.ndarray, dst: np.ndarray) -> None:
    """Swizzle H×W×4 RGBA into an H×W×4 BGRA view, no allocation.

    Four strided channel copies. Benchmarked against a uint32
    mask/shift/or variant (same result, same zero-allocation property):
    ~3.1 ms vs ~3.7 ms per 1920×1080 frame, so the simpler one wins.
    """
    dst[..., 0] = src[..., 2]
    dst[..., 1] = src[..., 1]
    dst[..., 2] = src[..., 0]
    dst[..., 3] = src[..., 3]


# --- Presenter -----------------------------------------------------------
class LayeredPresenter:
    """Presents premultiplied RGBA frames through UpdateLayeredWindow.

    Owns one memory DC and one top-down 32-bpp DIB section. The DIB is
    grow-only, mirroring SDLWindow's texture: a window that resizes
    every frame reallocates only when the frame exceeds the largest
    size seen so far. The frame is written into the top-left w×h of the
    DIB and only that sub-rectangle is handed to DWM.
    """

    def __init__(self, hwnd: int) -> None:
        self._api = api()
        self._hwnd = hwnd
        self._memdc: Any = None
        self._bitmap: Any = None
        self._old_bitmap: Any = None
        self._pixels: Optional[np.ndarray] = None  # (cap_h, cap_w, 4) uint8 view of the DIB
        self._capacity = (0, 0)
        self._size = (0, 0)
        self._blend = BLENDFUNCTION(AC_SRC_OVER, 0, 255, AC_SRC_ALPHA)
        self._src_origin = wintypes.POINT(0, 0)
        self._failure_logged = False

        self._memdc = self._api.CreateCompatibleDC(None)
        if not self._memdc:
            raise OSError(f"CreateCompatibleDC failed (error {last_error()})")

        style = get_exstyle(hwnd)
        if not style & WS_EX_LAYERED:
            set_exstyle(hwnd, style | WS_EX_LAYERED)

    # ------------------------------------------------------------------
    def _ensure_capacity(self, w: int, h: int) -> None:
        cap_w, cap_h = self._capacity
        if self._pixels is not None and w <= cap_w and h <= cap_h:
            return
        cap_w, cap_h = max(w, cap_w), max(h, cap_h)

        bmi = BITMAPINFO()
        hdr = bmi.bmiHeader
        hdr.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        hdr.biWidth = cap_w
        hdr.biHeight = -cap_h  # negative = top-down rows, same as numpy
        hdr.biPlanes = 1
        hdr.biBitCount = 32
        hdr.biCompression = BI_RGB

        bits = ctypes.c_void_p()
        bitmap = self._api.CreateDIBSection(
            self._memdc, ctypes.byref(bmi), DIB_RGB_COLORS, ctypes.byref(bits), None, 0
        )
        if not bitmap or not bits.value:
            raise OSError(f"CreateDIBSection({cap_w}x{cap_h}) failed (error {last_error()})")

        previous = self._api.SelectObject(self._memdc, bitmap)
        if self._bitmap is not None:
            # `previous` is our old DIB; free it. The DC's original
            # 1x1 stock bitmap is kept in _old_bitmap for close().
            self._api.DeleteObject(self._bitmap)
        else:
            self._old_bitmap = previous
        self._bitmap = bitmap

        # 32-bpp DIB rows are DWORD-aligned by construction, so the
        # stride is exactly cap_w * 4 and a plain reshape is exact.
        buf = (ctypes.c_uint8 * (cap_w * cap_h * 4)).from_address(bits.value)
        self._pixels = np.ctypeslib.as_array(buf).reshape(cap_h, cap_w, 4)
        self._capacity = (cap_w, cap_h)

    # ------------------------------------------------------------------
    def present(self, arr: np.ndarray) -> bool:
        """Upload one premultiplied H×W×4 RGBA frame. Returns True if the
        window's size changed."""
        h, w = int(arr.shape[0]), int(arr.shape[1])
        self._ensure_capacity(w, h)
        assert self._pixels is not None

        # GDI may still be reading the DIB from the previous
        # UpdateLayeredWindow; GdiFlush is the documented barrier before
        # touching DIB-section memory directly.
        self._api.GdiFlush()
        rgba_to_bgra(arr, self._pixels[:h, :w])

        resized = (w, h) != self._size
        self._size = (w, h)
        self._update(w, h)
        return resized

    def present_again(self) -> None:
        """Re-issue the last frame without touching the pixels.

        DWM retains a layered window's bitmap, so an expose does not
        actually need this; it exists so SDLWindow.present_again keeps
        the same contract on both paths.
        """
        w, h = self._size
        if w and h:
            self._update(w, h)

    def _update(self, w: int, h: int) -> None:
        size = wintypes.SIZE(w, h)
        ok = self._api.UpdateLayeredWindow(
            self._hwnd,
            None,  # screen DC
            None,  # keep the current position; placement owns that
            ctypes.byref(size),
            self._memdc,
            ctypes.byref(self._src_origin),
            0,
            ctypes.byref(self._blend),
            ULW_ALPHA,
        )
        if not ok and not self._failure_logged:
            # Logged once: this runs every frame, and a persistent
            # failure (e.g. someone called SetLayeredWindowAttributes on
            # this window, which disables UpdateLayeredWindow for it)
            # would otherwise flood the log.
            self._failure_logged = True
            log.warning(
                "UpdateLayeredWindow failed (error %d); frames will not be "
                "shown. Further failures are not logged.",
                last_error(),
            )

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    def close(self) -> None:
        if self._memdc:
            if self._old_bitmap is not None:
                self._api.SelectObject(self._memdc, self._old_bitmap)
            self._api.DeleteDC(self._memdc)
            self._memdc = None
        if self._bitmap is not None:
            self._api.DeleteObject(self._bitmap)
            self._bitmap = None
        self._pixels = None
