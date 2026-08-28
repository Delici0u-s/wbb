"""
_pages.py — small helper shared by the 10-13 examples.

`data_url()` exists because a raw `data:text/html,<...>` string is a
trap: an unescaped `#` starts the URL fragment, so a page written as

    data:text/html,<body style='background:#101418'>hello</body>

is silently truncated to `<body style='background:` and renders blank.
Spaces and `%` have the same problem. Percent-encode the document and
the whole class of "why is my overlay black" disappears.
"""

from __future__ import annotations

from urllib.parse import quote


def data_url(html: str) -> str:
    """Percent-encoded data: URL for an HTML fragment."""
    return "data:text/html;charset=utf-8," + quote(html, safe="")
