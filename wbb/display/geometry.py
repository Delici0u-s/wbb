"""
geometry.py — pure geometry helpers for a window that resizes itself.

No SDL, no X11, no I/O: importable and unit-testable anywhere. All
functions are O(1) and allocate nothing but the returned tuple.
"""

from __future__ import annotations

from enum import Enum


class Anchor(str, Enum):
    """Which point of the window stays put when the frame changes shape.

    `push_frame` resizes the window by calling `SDL_SetWindowSize`, which
    keeps the *top-left* corner fixed — the window grows down and to the
    right. For an overlay pinned to, say, the bottom-right of a screen,
    that is the wrong corner: growing the window walks it off the edge.
    `DisplayClient(anchor=...)` recomputes the origin so the chosen point
    is the one that does not move, and sends the new origin and size to
    the placement backend as one geometry change.
    """

    TOP_LEFT = "top-left"
    TOP_RIGHT = "top-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_RIGHT = "bottom-right"
    CENTER = "center"


def anchored_origin(
    anchor: Anchor,
    x: int,
    y: int,
    old_w: int,
    old_h: int,
    new_w: int,
    new_h: int,
) -> tuple[int, int]:
    """New top-left origin that keeps `anchor` fixed while resizing.

    `(x, y, old_w, old_h)` is the window's current geometry;
    `(new_w, new_h)` the size it is about to become. Returns the top-left
    the window should move to.
    """
    dw = new_w - old_w
    dh = new_h - old_h

    if anchor is Anchor.TOP_LEFT:
        return (x, y)
    if anchor is Anchor.TOP_RIGHT:
        return (x - dw, y)
    if anchor is Anchor.BOTTOM_LEFT:
        return (x, y - dh)
    if anchor is Anchor.BOTTOM_RIGHT:
        return (x - dw, y - dh)
    if anchor is Anchor.CENTER:
        return (x - dw // 2, y - dh // 2)
    return (x, y)
