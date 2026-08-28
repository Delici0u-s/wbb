"""
alpha.py — per-pixel window alpha for the SDL2 window.

Three things have to line up, and they happen at three different times:

1. Before `SDL_CreateWindow`: the window needs a visual/config that has
   an alpha channel. On X11 (including XWayland) that means a depth-32
   TrueColor visual, forced via the `SDL_VIDEO_X11_VISUALID` hint,
   because SDL2 otherwise picks the screen's default 24-bit visual. On
   native Wayland the surface is already ARGB, but the GL config still
   has to be asked for alpha, hence `SDL_GL_ALPHA_SIZE = 8` — which is
   also harmless on X11 and is set unconditionally.
2. After `SDL_CreateWindow`: verify. `verify()` reads the window's
   actual depth off the X server rather than trusting the hint.
3. Every frame: the texture must be blitted with blending *off* so the
   frame's alpha is written to the target instead of being composited
   against it, and the clear colour must have alpha 0 for the same
   reason. `configure_renderer()` sets both once; neither is per-frame
   work.

Nothing here raises. Without python-xlib, step 1's X11 half is skipped
and `verify()` returns None ("unknown"); the caller decides whether to
proceed with an opaque window or fail.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from .. import _compat
from .x11_props import window_depth, x11_connection

log = logging.getLogger(__name__)

#: X11 visual class constant (Xlib.X.TrueColor). Inlined so this module
#: does not import Xlib just to read one integer.
_TRUE_COLOR = 4


@dataclass(frozen=True, slots=True)
class AlphaPreparation:
    """What `prepare()` managed to set up, for logging and for `verify()`."""

    requested: bool
    x11_visual_id: Optional[int] = None
    gl_alpha_requested: bool = False

    def describe(self) -> str:
        if not self.requested:
            return "alpha not requested"
        bits = []
        if self.x11_visual_id is not None:
            bits.append(f"X11 visual 0x{self.x11_visual_id:x} (depth 32)")
        else:
            bits.append("no ARGB X11 visual found/forced")
        bits.append("GL alpha requested" if self.gl_alpha_requested else "GL alpha not requested")
        return "; ".join(bits)


def find_argb_visual_id(display_name: Optional[str] = None) -> Optional[int]:
    """Return a depth-32 TrueColor visual id on the default screen, or None."""
    with x11_connection(display_name) as disp:
        if disp is None:
            return None
        try:
            screen = disp.screen()
            for depth in screen.allowed_depths:
                if depth.depth != 32:
                    continue
                for visual in depth.visuals:
                    if visual.visual_class == _TRUE_COLOR:
                        return int(visual.visual_id)
        except Exception:
            log.debug("alpha: visual enumeration failed", exc_info=True)
        return None


def prepare(
    requested: bool,
    *,
    display_name: Optional[str] = None,
    live_windows: int = 0,
) -> AlphaPreparation:
    """Set the pre-creation SDL hints/attributes. Call before SDL_CreateWindow.

    SDL2's X11 backend resolves the screen visual once, in
    `X11_VideoInit` — i.e. inside `SDL_Init(SDL_INIT_VIDEO)` — so the
    `SDL_VIDEO_X11_VISUALID` hint has to be set before that. Anything
    that touches SDL first defeats it, including innocuous-looking calls
    like `list_displays()` (and therefore `centered_position()`), which
    initialise video to query monitor bounds.

    If video is already up but no windows exist yet, the hint is set and
    the video subsystem is bounced so `X11_VideoInit` runs again and
    reads it. With windows already open that is not safe, so it warns
    instead.
    """
    if not requested:
        return AlphaPreparation(requested=False)

    import sdl2  # noqa: PLC0415

    already_up = bool(sdl2.SDL_WasInit(sdl2.SDL_INIT_VIDEO))

    gl_ok = False
    try:
        gl_ok = sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_ALPHA_SIZE, 8) == 0
    except Exception:
        log.debug("alpha: SDL_GL_SetAttribute(ALPHA_SIZE) failed", exc_info=True)

    vid = find_argb_visual_id(display_name)
    if vid is not None:
        value = f"0x{vid:x}"
        # SDL2's X11 backend reads this with SDL_getenv() in
        # get_visualinfo(), NOT with SDL_GetHint() — so SDL_SetHint alone
        # is silently ignored, which is exactly what we saw: the visual
        # was found and forced and the window still came out 24-bit. Set
        # the environment variable, which os.environ writes through to
        # putenv() so the C-level getenv() sees it. The SetHint call
        # stays for SDL builds that do read it as a hint.
        os.environ["SDL_VIDEO_X11_VISUALID"] = value
        sdl2.SDL_SetHint(b"SDL_VIDEO_X11_VISUALID", value.encode())
        # 0x-prefixed parses identically under strtol base 0 and base 16,
        # so this is safe whichever SDL uses.
    else:
        log.debug(
            "alpha: no depth-32 TrueColor visual available (or python-xlib "
            "missing). On X11 the window will be opaque."
        )

    if already_up and vid is not None:
        if live_windows == 0:
            log.debug(
                "alpha: SDL video was already initialised; restarting the video "
                "subsystem so the forced visual is picked up"
            )
            sdl2.SDL_QuitSubSystem(sdl2.SDL_INIT_VIDEO)
            if sdl2.SDL_InitSubSystem(sdl2.SDL_INIT_VIDEO) != 0:
                log.warning(
                    "alpha: could not restart SDL video (%s); the window will "
                    "be opaque",
                    sdl2.SDL_GetError().decode(errors="replace"),
                )
        else:
            log.warning(
                "alpha: SDL video is already initialised and %d window(s) are "
                "open, so the forced visual cannot be applied and this window "
                "will be opaque. Create the alpha window before any other "
                "SDLWindow, or call wbb.display.preinit_alpha() first.",
                live_windows,
            )

    prep = AlphaPreparation(requested=True, x11_visual_id=vid, gl_alpha_requested=gl_ok)
    log.debug("alpha: %s", prep.describe())
    return prep


def clear_hints() -> None:
    """No-op kept for compatibility; the visual selection is not cleared.

    Both the hint and the environment variable are process-global, but
    SDL resolves the visual once at video-init time and reuses it for
    every window afterwards, so clearing them after the first window has
    no effect on that window and would only confuse a later
    `preinit_alpha()`. A 32-bit visual is harmless for opaque windows:
    decoded frames carry alpha 255 and the texture blend mode is NONE,
    so they are written fully opaque.
    """
    return


def verify(handle: object, *, display_name: Optional[str] = None) -> Optional[bool]:
    """Did the created window actually get a 32-bit visual?

    True/False on X11, None when it cannot be determined (native
    Wayland, or python-xlib missing). None is not a failure: on Wayland
    the surface is ARGB regardless.
    """
    if getattr(handle, "subsystem", "") != "x11":
        return None
    window_id = getattr(handle, "x11_window", None)
    if not window_id:
        return None
    depth = window_depth(int(window_id), display_name=display_name)
    if depth is None:
        return None
    return depth == 32


def configure_renderer(renderer: object, alpha: bool) -> None:
    """Set the draw colour and its blend mode once, at renderer creation.

    With alpha on, the clear colour is (0, 0, 0, 0) and the draw blend
    mode is NONE, so `SDL_RenderClear` *writes* alpha 0 rather than
    blending a transparent black over whatever was there. With alpha
    off, the clear is opaque black — which is also a change from the
    previous behaviour, where the clear colour was left at SDL's
    default and any part of the window not covered by the texture (a
    resize the WM refused, say) cleared to that default rather than to
    something deliberate.
    """
    if _compat.LEGACY_RENDER_CONFIG:
        log.debug("alpha: WBB_LEGACY_RENDER_CONFIG set; leaving renderer state alone")
        return
    import sdl2  # noqa: PLC0415

    sdl2.SDL_SetRenderDrawBlendMode(renderer, sdl2.SDL_BLENDMODE_NONE)
    sdl2.SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0 if alpha else 255)


def configure_texture(texture: object) -> None:
    """Blit the frame with blending off so its alpha is written, not composited.

    `SDL_CreateTexture` already defaults to `SDL_BLENDMODE_NONE`, so
    this is an assertion rather than a fix — but it is load-bearing for
    transparency and cheap to state explicitly rather than inherit.
    """
    if _compat.LEGACY_RENDER_CONFIG:
        return
    import sdl2  # noqa: PLC0415

    sdl2.SDL_SetTextureBlendMode(texture, sdl2.SDL_BLENDMODE_NONE)
