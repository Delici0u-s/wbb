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

    Example — a warm red tint::

        filters.colorize(r=1.3, g=0.8, b=0.7)
    """
    _muls = np.array([r, g, b, a], dtype=np.float32)

    def _colorize(frame: np.ndarray) -> np.ndarray:
        return (frame.astype(np.float32) * _muls).clip(0, 255).astype(np.uint8)

    return _colorize


def brightness(delta: int) -> Filter:
    """
    Shift pixel values by *delta* (positive = brighter, negative = darker).

    Alpha is unchanged. *delta* is clipped to keep values in [0, 255].
    """

    def _bright(frame: np.ndarray) -> np.ndarray:
        out = np.asarray(frame.copy())
        rgb = out[..., :3].astype(np.int16)
        out[..., :3] = (rgb + delta).clip(0, 255).astype(np.uint8)
        return out

    return _bright


def contrast(factor: float) -> Filter:
    """
    Scale contrast around the midpoint (128) by *factor*.

    Values < 1 reduce contrast; values > 1 increase it.
    """

    def _contrast(frame: np.ndarray) -> np.ndarray:
        out = np.asarray(frame.copy())
        rgb = out[..., :3].astype(np.float32)
        out[..., :3] = ((rgb - 128) * factor + 128).clip(0, 255).astype(np.uint8)
        return out

    return _contrast


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
    """Return a filter that applies *first* then *second*."""

    def _composed(frame: np.ndarray) -> np.ndarray:
        return second(first(frame))

    return _composed


def chain(*filters: Filter) -> Filter:
    """
    Compose an ordered sequence of filters into a single filter.

    ``chain(f, g, h)(x)`` is equivalent to ``h(g(f(x)))``.
    """
    if not filters:
        return lambda f: f

    result = filters[0]
    for f in filters[1:]:
        result = compose(result, f)
    return result


# ---------------------------------------------------------------------------
# Convenience identity
# ---------------------------------------------------------------------------


def identity() -> Filter:
    """Pass-through: returns the frame unchanged."""
    return lambda frame: frame
