"""
filters — composable image transformations.

Every function has signature ``(np.ndarray) -> np.ndarray`` where the
array is H×W×4 uint8 RGBA. Filters are pure (they return new arrays or
views; they do not modify their input), stateless, and may be composed
freely.

User-defined filters are identical in form — any callable matching the
signature plugs into a DisplayClient filter pipeline or a manual pipeline
without modification.

Built-ins
---------
crop        — sub-region slice (zero-copy view)
scale       — resize to new (width, height)
grayscale   — luminance conversion, alpha preserved
blur        — simple box blur
flip        — horizontal or vertical flip (or both)
colorize    — tint RGBA by a multiplier per channel
brightness  — additive brightness shift
contrast    — contrast stretch around the midpoint
compose     — pipeline combinator: compose(f, g)(x) == g(f(x))
chain       — compose an ordered list of filters into one
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Filter = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# LUT-fused per-channel color filters
# ---------------------------------------------------------------------------
#
# colorize / brightness / contrast (and the RGB-scaling part of any
# per-channel point operation) are all functions applied independently
# to each pixel value in each channel. That makes them expressible as a
# 256-entry uint8 lookup table per channel — a (256, 4) array — applied
# in one pass with ``lut[frame]`` fancy indexing.
#
# Two wins over the old astype(float32) -> math -> clip -> astype(uint8)
# form:
#   1. One uint8 gather instead of several full-frame float32 temporaries
#      (a 1280x720x4 float32 temp is 14.7 MB; the old colorize+contrast
#      chain allocated a handful of them per frame). The LUT itself is
#      4 KB, built once at filter-construction time, not per frame.
#   2. Composable for free: applying LUT B after LUT A is just
#      ``B_table[A_table]`` — gather the two tables into one at
#      chain-build time, so an N-deep color chain costs exactly one
#      gather per frame regardless of N.
#
# A LUT filter is marked with ._wbb_lut (the (256,4) table) so chain()
# and compose() can detect adjacent LUT filters and fuse their tables.


class _LutFilter:
    """A per-channel point filter backed by a (256, 4) uint8 table.

    Calling it applies ``table[frame]``. Two ``_LutFilter``s compose by
    composing their tables, so a chain of them collapses to a single
    gather per frame.
    """

    __slots__ = ("_table",)

    def __init__(self, table: np.ndarray) -> None:
        # table: (256, 4) uint8 — table[v, c] is the output for input
        # value v in channel c.
        self._table = table

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        # frame is H x W x 4 uint8. Index each channel's column of the
        # table by that channel's values. take_along_axis keeps it one
        # vectorised gather with no Python-level channel loop.
        # self._table[frame] would broadcast the whole (256,4) table per
        # element; instead gather per-channel:
        f = frame
        out = np.empty_like(f)
        t = self._table
        out[..., 0] = t[f[..., 0], 0]
        out[..., 1] = t[f[..., 1], 1]
        out[..., 2] = t[f[..., 2], 2]
        out[..., 3] = t[f[..., 3], 3]
        return out

    def then(self, other: "_LutFilter") -> "_LutFilter":
        """Return a single LUT equivalent to applying self then other."""
        # other applied after self: for each channel, compose the maps.
        # other._table[self._table] gathers self's outputs through
        # other's table, per channel.
        s = self._table
        o = other._table
        fused = np.empty((256, 4), dtype=np.uint8)
        for c in range(4):
            fused[:, c] = o[s[:, c], c]
        return _LutFilter(fused)


def _identity_table() -> np.ndarray:
    base = np.arange(256, dtype=np.uint8)
    return np.repeat(base[:, None], 4, axis=1)  # (256, 4)


# ---------------------------------------------------------------------------
# Spatial transforms
# ---------------------------------------------------------------------------


def crop(x: int, y: int, width: int, height: int) -> Filter:
    """
    Return a filter that slices the sub-region ``[y:y+h, x:x+w]``.

    The result is a zero-copy view — no pixel data is copied.
    """

    def _crop(frame: np.ndarray) -> np.ndarray:
        return frame[y : y + height, x : x + width]

    return _crop


def scale(width: int, height: int) -> Filter:
    """Resize to *(width, height)* using Pillow's LANCZOS filter."""

    def _scale(frame: np.ndarray) -> np.ndarray:
        from PIL import Image  # noqa: PLC0415

        img = Image.fromarray(frame, mode="RGBA")
        img = img.resize((width, height), Image.LANCZOS)  # type: ignore[attr-defined]
        return np.asarray(img, dtype=np.uint8)

    return _scale


def flip(*, horizontal: bool = False, vertical: bool = False) -> Filter:
    """Flip horizontally and/or vertically (zero-copy via numpy slicing)."""

    def _flip(frame: np.ndarray) -> np.ndarray:
        if horizontal:
            frame = frame[:, ::-1]
        if vertical:
            frame = frame[::-1, :]
        return frame

    return _flip


# ---------------------------------------------------------------------------
# Color transforms
# ---------------------------------------------------------------------------


def grayscale() -> Filter:
    """
    Convert RGB channels to luminance (ITU-R BT.601) while preserving
    the alpha channel.

    Not a per-channel LUT (it mixes channels), so it stays a real
    per-frame compute and does not fuse with the LUT filters around it.
    """
    _coeffs = np.array([0.299, 0.587, 0.114, 0.0], dtype=np.float32)

    def _gray(frame: np.ndarray) -> np.ndarray:
        lum = (frame.astype(np.float32) * _coeffs).sum(axis=2, keepdims=True)
        lum = lum.clip(0, 255).astype(np.uint8)
        out = np.empty_like(frame)
        out[..., :3] = lum
        out[..., 3] = frame[..., 3]
        return out

    return _gray


def colorize(r: float = 1.0, g: float = 1.0, b: float = 1.0, a: float = 1.0) -> Filter:
    """
    Multiply each channel by the given factor (clipped to [0, 255]).

    Implemented as a per-channel LUT, so it composes with brightness/
    contrast into a single lookup (see module-level ``_LutFilter``).

    Example — a warm red tint::

        filters.colorize(r=1.3, g=0.8, b=0.7)
    """
    vals = np.arange(256, dtype=np.float32)
    table = np.empty((256, 4), dtype=np.uint8)
    for c, m in enumerate((r, g, b, a)):
        table[:, c] = (vals * m).clip(0, 255).astype(np.uint8)
    return _LutFilter(table)


def brightness(delta: int) -> Filter:
    """
    Shift pixel values by *delta* (positive = brighter, negative = darker).

    Alpha is unchanged. Implemented as a per-channel LUT, fusable with
    colorize/contrast.
    """
    vals = np.arange(256, dtype=np.int16)
    table = np.empty((256, 4), dtype=np.uint8)
    shifted = (vals + delta).clip(0, 255).astype(np.uint8)
    table[:, 0] = shifted
    table[:, 1] = shifted
    table[:, 2] = shifted
    table[:, 3] = np.arange(256, dtype=np.uint8)  # alpha untouched
    return _LutFilter(table)


def contrast(factor: float) -> Filter:
    """
    Scale contrast around the midpoint (128) by *factor*.

    Values < 1 reduce contrast; values > 1 increase it. Per-channel LUT,
    fusable with colorize/brightness.
    """
    vals = np.arange(256, dtype=np.float32)
    scaled = ((vals - 128) * factor + 128).clip(0, 255).astype(np.uint8)
    table = np.empty((256, 4), dtype=np.uint8)
    table[:, 0] = scaled
    table[:, 1] = scaled
    table[:, 2] = scaled
    table[:, 3] = np.arange(256, dtype=np.uint8)  # alpha untouched
    return _LutFilter(table)


# ---------------------------------------------------------------------------
# Spatial blur
# ---------------------------------------------------------------------------


def blur(radius: int = 2) -> Filter:
    """
    Box blur with the given *radius* (in pixels).

    Uses a separable 1-D convolution for efficiency. Requires scipy if
    ``radius > 1``; falls back to a numpy roll-average for radius == 1.
    """

    def _blur(frame: np.ndarray) -> np.ndarray:
        try:
            from scipy.ndimage import uniform_filter  # noqa: PLC0415

            out = np.asarray(frame.copy())
            size = radius * 2 + 1
            out[..., :3] = uniform_filter(frame[..., :3], size=[size, size, 1])
            return out
        except ImportError:
            # minimal fallback — single-pixel shift average
            out = frame.astype(np.float32)
            shifted = np.roll(out, 1, axis=0) + np.roll(out, -1, axis=0)
            shifted += np.roll(out, 1, axis=1) + np.roll(out, -1, axis=1)
            return np.asarray(((out + shifted) / 5).clip(0, 255).astype(np.uint8))

    return _blur


# ---------------------------------------------------------------------------
# Pipeline combinators
# ---------------------------------------------------------------------------


def compose(first: Filter, second: Filter) -> Filter:
    """Return a filter that applies *first* then *second*.

    If both are per-channel LUT filters (colorize/brightness/contrast),
    they are fused into a single LUT at build time, so the composed
    filter costs one gather per frame instead of two.
    """
    if isinstance(first, _LutFilter) and isinstance(second, _LutFilter):
        return first.then(second)

    def _composed(frame: np.ndarray) -> np.ndarray:
        return second(first(frame))

    return _composed


def chain(*filters: Filter) -> Filter:
    """
    Compose an ordered sequence of filters into a single filter.

    ``chain(f, g, h)(x)`` is equivalent to ``h(g(f(x)))``.

    Runs of adjacent per-channel LUT filters (colorize/brightness/
    contrast) are fused into a single LUT, so e.g.
    ``chain(colorize(...), contrast(...), brightness(...))`` collapses to
    one gather per frame. A non-LUT filter (crop/scale/blur/grayscale/
    flip/custom) breaks the run; LUT fusion resumes after it.
    """
    if not filters:
        return lambda f: f

    # Coalesce maximal runs of _LutFilter into single fused LUTs first.
    coalesced: list[Filter] = []
    for f in filters:
        if isinstance(f, _LutFilter) and coalesced and isinstance(coalesced[-1], _LutFilter):
            coalesced[-1] = coalesced[-1].then(f)
        else:
            coalesced.append(f)

    result = coalesced[0]
    for f in coalesced[1:]:
        result = compose(result, f)
    return result


# ---------------------------------------------------------------------------
# Convenience identity
# ---------------------------------------------------------------------------


def identity() -> Filter:
    """Pass-through: returns the frame unchanged."""
    return lambda frame: frame
