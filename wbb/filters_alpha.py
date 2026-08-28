"""
filters_alpha.py — filters that write the alpha channel.

These do nothing visible on their own. Alpha only reaches the screen
when the window has a 32-bit visual, i.e. `DisplayClient(alpha=True)`
and `is_alpha_active()` returning True. On an opaque window the alpha
is uploaded and discarded, which looks exactly like the filter not
working.

Everything here answers one of three questions:

    make the whole window see-through      -> opacity(0.5)
    make a region see-through              -> alpha_rect(x, y, w, h, inside=0)
    make everything except a region see-   -> alpha_rect(x, y, w, h)
    through                                   (outside=0 is the default)

`inside` and `outside` are alpha values in 0-255, or None to leave that
side of the boundary untouched. That one convention covers cut-outs,
keep-onlys, and partial fades without a separate function for each.

Cost and allocation
-------------------
Every filter is O(H*W) and allocates one output frame per call, because
the frame it is handed is a read-only view into shared memory. Masks are
built once per (height, width, geometry) and cached, so the per-frame
work is a copy plus one gather — no trigonometry, no float temporaries.
Chaining N of these costs N copies; prefer one `alpha_regions([...])`
over several `alpha_rect` calls.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

import numpy as np

Filter = Callable[[np.ndarray], np.ndarray]
Rect = tuple[int, int, int, int]

__all__ = [
    "premultiply",
    "opacity",
    "alpha_mask",
    "alpha_rect",
    "alpha_regions",
    "alpha_rounded_rect",
    "alpha_ellipse",
    "alpha_color_key",
    "alpha_gradient",
]


def premultiply(frame: np.ndarray) -> np.ndarray:
    """Convert straight (unassociated) alpha to premultiplied alpha.

    X11 composites 32-bit ARGB windows with XRender's PictOpOver, and
    KWin's OpenGL backend uses `glBlendFunc(GL_ONE,
    GL_ONE_MINUS_SRC_ALPHA)`. Both assume **premultiplied** alpha: the
    colour channels must already be scaled by the alpha. A pixel of
    white with alpha 0 — `(255, 255, 255, 0)`, which is what you get by
    zeroing the alpha channel of a white background — is not "invisible"
    to a premultiplied compositor. It is "add full-intensity white with
    zero coverage", and it comes out looking opaque.

    That is why an alpha filter over a white page background produces a
    solid white rectangle, while a transparent PNG from Chrome — whose
    empty areas are `(0, 0, 0, 0)` — looks perfectly transparent: black
    happens to be the one colour that is identical in both conventions.

    `DisplayClient(alpha=True)` applies this automatically before upload
    (see `premultiply=` there), so most callers never need this
    function. It is here for anyone driving `SDLWindow` directly.

    Cost: one pass over H*W*3. The common case where alpha is only ever
    0 or 255 is detected and handled with a boolean write instead of a
    multiply.
    """
    out = frame.copy()  # never mutate the caller's array
    _premultiply_inplace(out)
    return out


def _premultiply_inplace(out: np.ndarray) -> None:
    """Premultiply `out` in place. `out` must be a writeable uint8 RGBA array.

    Measured at 1280x720: ~8 ms, or ~0.5 ms for a typical 420x200
    overlay. Fully opaque frames short-circuit on a min() reduction, so
    enabling alpha and then not using it costs almost nothing. The
    uint16 workspace is cached for the last frame shape, so this is
    allocation-free once the size settles.
    """
    alpha = out[..., 3]
    if int(alpha.min()) == 255:
        return  # nothing to scale

    work = _PM_WORKSPACE.get(out.shape)
    if work is None:
        work = np.empty(out.shape, dtype=np.uint16)
        _PM_WORKSPACE.clear()  # only ever keep the current shape
        _PM_WORKSPACE[out.shape] = work

    keep = alpha.copy()
    # Scale all four channels, then put alpha back. Doing it over the
    # whole array keeps the multiply on contiguous memory, which is
    # roughly three times faster than a strided [..., :3] view.
    np.multiply(out, out[..., 3:4], out=work, dtype=np.uint16)
    work += 127
    work //= 255
    np.copyto(out, work, casting="unsafe")
    out[..., 3] = keep


#: Workspace for _premultiply_inplace, keyed by frame shape. Holds one
#: entry: the shape most recently seen.
_PM_WORKSPACE: dict[tuple[int, ...], np.ndarray] = {}


def _to_alpha(value: float) -> int:
    """Accept 0.0-1.0 or 0-255 and return 0-255."""
    if isinstance(value, float) and 0.0 <= value <= 1.0:
        return int(round(value * 255))
    return int(np.clip(int(value), 0, 255))


def _apply_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    inside: Optional[int],
    outside: Optional[int],
) -> np.ndarray:
    """Write `inside`/`outside` into the alpha channel where `mask` is True/False."""
    out = frame.copy()
    alpha = out[..., 3]
    if inside is not None and outside is not None:
        alpha[...] = np.where(mask, inside, outside)
    elif inside is not None:
        alpha[mask] = inside
    elif outside is not None:
        alpha[~mask] = outside
    return out


def opacity(level: float) -> Filter:
    """Scale the whole frame's alpha.

    `level` is 0.0-1.0 or 0-255. `opacity(0.5)` makes the entire window
    half see-through; `opacity(0)` makes it fully transparent, which is
    a useful way to check that the window really has an alpha visual.

    Multiplies the existing alpha rather than replacing it, so this
    composes with the region filters below in either order.
    """
    scale = _to_alpha(level) / 255.0

    def _filter(frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        # uint8 * float -> float temporary of H*W, not H*W*4.
        out[..., 3] = (out[..., 3] * scale).astype(np.uint8)
        return out

    _filter.__name__ = f"opacity({level})"
    return _filter


def alpha_mask(
    mask: np.ndarray,
    *,
    inside: Optional[int] = 255,
    outside: Optional[int] = 0,
) -> Filter:
    """Use an explicit boolean mask. True means "inside".

    The mask must match the frame's height and width. Use this when the
    shape you want is easier to compute than to describe — a logo, a
    threshold on the frame's own content, anything.
    """
    mask = np.asarray(mask, dtype=bool)
    inside_a = None if inside is None else _to_alpha(inside)
    outside_a = None if outside is None else _to_alpha(outside)

    def _filter(frame: np.ndarray) -> np.ndarray:
        if mask.shape != frame.shape[:2]:
            raise ValueError(
                f"alpha_mask: mask is {mask.shape}, frame is {frame.shape[:2]}"
            )
        return _apply_mask(frame, mask, inside_a, outside_a)

    _filter.__name__ = "alpha_mask"
    return _filter


def alpha_regions(
    rects: Sequence[Rect],
    *,
    inside: Optional[int] = 255,
    outside: Optional[int] = 0,
) -> Filter:
    """Alpha for the union of several `(x, y, width, height)` rectangles.

    Defaults keep the rectangles opaque and make everything else
    transparent. Swap to `inside=0, outside=None` to punch holes instead
    and leave the rest of the frame as it was.

    Pairs well with `BrowserBridge.element_bounds(selector)`: resolve the
    elements you care about, hand the rectangles straight in.
    """
    boxes = [tuple(int(v) for v in r) for r in rects]
    inside_a = None if inside is None else _to_alpha(inside)
    outside_a = None if outside is None else _to_alpha(outside)
    cache: dict[tuple[int, int], np.ndarray] = {}

    def _filter(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        mask = cache.get((h, w))
        if mask is None:
            mask = np.zeros((h, w), dtype=bool)
            for x, y, rw, rh in boxes:
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(w, x + rw), min(h, y + rh)
                if x1 > x0 and y1 > y0:
                    mask[y0:y1, x0:x1] = True
            cache[(h, w)] = mask
        return _apply_mask(frame, mask, inside_a, outside_a)

    _filter.__name__ = f"alpha_regions({len(boxes)})"
    return _filter


def alpha_rect(
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    inside: Optional[int] = 255,
    outside: Optional[int] = 0,
) -> Filter:
    """Alpha for a single rectangle. See `alpha_regions`."""
    return alpha_regions([(x, y, width, height)], inside=inside, outside=outside)


def _rounded_mask(h: int, w: int, rect: Rect, radius: int) -> np.ndarray:
    x, y, rw, rh = rect
    radius = max(0, min(radius, rw // 2, rh // 2))
    yy, xx = np.ogrid[:h, :w]
    left = xx - x
    right = (x + rw - 1) - xx
    top = yy - y
    bottom = (y + rh - 1) - yy
    inside_rect = (left >= 0) & (right >= 0) & (top >= 0) & (bottom >= 0)
    if radius == 0:
        return inside_rect
    inx = np.clip(np.minimum(left, right), 0, radius)
    iny = np.clip(np.minimum(top, bottom), 0, radius)
    in_corner = (inx < radius) & (iny < radius)
    corner_ok = np.hypot(radius - inx, radius - iny) <= radius
    return inside_rect & np.where(in_corner, corner_ok, True)


def alpha_rounded_rect(
    *,
    rect: Optional[Rect] = None,
    inset: int = 0,
    radius: int = 0,
    inside: Optional[int] = 255,
    outside: Optional[int] = 0,
) -> Filter:
    """Alpha for a rounded rectangle.

    Give either an explicit `rect=(x, y, width, height)` or an `inset`,
    which is the margin left outside the rectangle on all four sides.
    `inset` is the common case for an overlay: it matches a CSS `margin`
    on the page, so the page's card and the window's opaque area line up.
    """
    inside_a = None if inside is None else _to_alpha(inside)
    outside_a = None if outside is None else _to_alpha(outside)
    cache: dict[tuple[int, int], np.ndarray] = {}

    def _filter(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        mask = cache.get((h, w))
        if mask is None:
            r = rect if rect is not None else (
                inset, inset, max(1, w - 2 * inset), max(1, h - 2 * inset)
            )
            mask = _rounded_mask(h, w, r, radius)
            cache[(h, w)] = mask
        return _apply_mask(frame, mask, inside_a, outside_a)

    _filter.__name__ = "alpha_rounded_rect"
    return _filter


def alpha_ellipse(
    *,
    rect: Optional[Rect] = None,
    inset: int = 0,
    inside: Optional[int] = 255,
    outside: Optional[int] = 0,
) -> Filter:
    """Alpha for an ellipse inscribed in a rectangle (or in the whole frame)."""
    inside_a = None if inside is None else _to_alpha(inside)
    outside_a = None if outside is None else _to_alpha(outside)
    cache: dict[tuple[int, int], np.ndarray] = {}

    def _filter(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        mask = cache.get((h, w))
        if mask is None:
            x, y, rw, rh = rect if rect is not None else (
                inset, inset, max(1, w - 2 * inset), max(1, h - 2 * inset)
            )
            cx, cy = x + rw / 2, y + rh / 2
            ax, by = max(rw / 2, 1e-6), max(rh / 2, 1e-6)
            yy, xx = np.ogrid[:h, :w]
            mask = ((xx - cx) / ax) ** 2 + ((yy - cy) / by) ** 2 <= 1.0
            cache[(h, w)] = mask
        return _apply_mask(frame, mask, inside_a, outside_a)

    _filter.__name__ = "alpha_ellipse"
    return _filter


def alpha_color_key(
    color: Iterable[int],
    *,
    tolerance: int = 12,
    alpha: int = 0,
) -> Filter:
    """Make pixels close to `color` transparent (chroma key).

    Useful when the shape you want to knock out is defined by the page
    rather than by coordinates — a page with `background: transparent`
    comes through a JPEG screencast as flat white, and
    `alpha_color_key((255, 255, 255))` removes it.

    Caveats worth knowing before relying on it: JPEG's chroma
    subsampling smears colour across edges, so keyed borders fringe. Use
    a generous `tolerance`, pick a key colour nothing else in the page
    uses, or switch to `screencast_format="png"` where the page's real
    alpha is available and none of this is necessary.

    Per frame: one int16 subtraction over H*W*3 plus a reduction. That is
    the most expensive filter here; budget a few milliseconds at 720p.
    """
    key = np.asarray(list(color)[:3], dtype=np.int16)
    tol = int(tolerance)
    alpha_v = _to_alpha(alpha)

    def _filter(frame: np.ndarray) -> np.ndarray:
        diff = np.abs(frame[..., :3].astype(np.int16) - key)
        hit = diff.max(axis=-1) <= tol
        out = frame.copy()
        out[..., 3][hit] = alpha_v
        return out

    _filter.__name__ = "alpha_color_key"
    return _filter


def alpha_gradient(
    *,
    direction: str = "down",
    start: float = 1.0,
    end: float = 0.0,
) -> Filter:
    """Fade alpha linearly across the frame.

    `direction` is "down", "up", "left" or "right"; `start` and `end` are
    0.0-1.0 multipliers applied to the existing alpha, so this composes
    with the region filters.
    """
    if direction not in ("down", "up", "left", "right"):
        raise ValueError("direction must be 'down', 'up', 'left' or 'right'")
    cache: dict[tuple[int, int], np.ndarray] = {}

    def _filter(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        ramp = cache.get((h, w))
        if ramp is None:
            n = h if direction in ("down", "up") else w
            line = np.linspace(start, end, n, dtype=np.float32)
            if direction in ("up", "right"):
                line = line[::-1]
            ramp = line[:, None] if direction in ("down", "up") else line[None, :]
            cache[(h, w)] = ramp
        out = frame.copy()
        out[..., 3] = np.clip(out[..., 3] * ramp, 0, 255).astype(np.uint8)
        return out

    _filter.__name__ = f"alpha_gradient({direction})"
    return _filter
