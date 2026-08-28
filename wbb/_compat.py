"""
_compat.py — per-change kill switches, for bisecting a regression.

Every behavioural change made in the 0.1.5 review is individually
disableable with an environment variable, so a regression can be
bisected without editing or reverting code:

    WBB_LEGACY_RENDER_CONFIG=1   don't touch the renderer draw colour /
                                 blend modes; use SDL's defaults
    WBB_LEGACY_STABLE_FRAME=1    filters read the shared-memory view
                                 directly again (no copy_latest())
    WBB_LEGACY_EXPOSE=1          ignore SDL_WINDOWEVENT again (no
                                 repaint on expose, no close handling)
    WBB_LEGACY_ACK=1             ack screencast frames before decoding
                                 (the pipelined, reorder-prone order)
    WBB_LEGACY_PARK=1            park in next_frame() for 1.0s again
    WBB_LEGACY_PREMULTIPLY=1     upload straight (unassociated) alpha
                                 again, as if DisplayClient(premultiply=
                                 False) had been passed. Exists so an
                                 example that does not expose the
                                 constructor flag can still be A/B'd
                                 without editing it.

Read once at import. These exist to answer "which change did it", not
as a supported configuration surface — every one of them turns a fix
back off.
"""

from __future__ import annotations

import os


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


LEGACY_RENDER_CONFIG = _flag("WBB_LEGACY_RENDER_CONFIG")
LEGACY_STABLE_FRAME = _flag("WBB_LEGACY_STABLE_FRAME")
LEGACY_EXPOSE = _flag("WBB_LEGACY_EXPOSE")
LEGACY_ACK = _flag("WBB_LEGACY_ACK")
LEGACY_PARK = _flag("WBB_LEGACY_PARK")
LEGACY_PREMULTIPLY = _flag("WBB_LEGACY_PREMULTIPLY")


def active() -> list[str]:
    """Names of the legacy switches currently on, for logging."""
    return [
        n
        for n, v in (
            ("WBB_LEGACY_RENDER_CONFIG", LEGACY_RENDER_CONFIG),
            ("WBB_LEGACY_STABLE_FRAME", LEGACY_STABLE_FRAME),
            ("WBB_LEGACY_EXPOSE", LEGACY_EXPOSE),
            ("WBB_LEGACY_ACK", LEGACY_ACK),
            ("WBB_LEGACY_PARK", LEGACY_PARK),
            ("WBB_LEGACY_PREMULTIPLY", LEGACY_PREMULTIPLY),
        )
        if v
    ]
