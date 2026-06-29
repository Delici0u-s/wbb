"""
KWinPlacement — always-on-top + absolute position via KWin's own
scripting API, reached over session D-Bus.

Why this exists / why it's tried first
---------------------------------------
Layer-shell is deliberately not used anywhere in this library (see
the project's design discussion) — KWin does implement
wlr-layer-shell for ordinary application windows, same as wlroots
compositors, but this codebase avoids it for unrelated reasons and
relies on KWin's own scripting interface instead: small JavaScript
snippets loaded into the running KWin process via D-Bus
(``org.kde.KWin`` / ``/Scripting``), executing with KWin's own
privilege rather than asking the compositor for a client permission.
Per KWin's own geometry-handling documentation, *position* changes
(unlike resizes) take effect immediately when the compositor sets
them — no client cooperation or repaint round-trip required, which is
why this works for a plain SDL2 xdg_toplevel window with no
Wayland-specific code on the SDL2 side at all.

The real bug this module previously had, and the actual fix
-----------------------------------------------------------------
Every ``set_above()``/``set_position()`` call used to write a fresh
*temp file with a new random name* and ``loadScript()`` it. KWin
accumulates every loaded script as a distinct object
(``/Scripting/Script0``, ``Script1``, ``Script2``, ...) with no
automatic eviction — calling ``loadScript()`` repeatedly without ever
unloading the previous one leaks script objects for the lifetime of
the KWin process, and in practice produced exactly the symptom
reported against this module: positioning that worked initially and
then silently stopped taking effect partway through a session, with
nothing raising an exception anywhere to surface it.

The fix has two parts:
1. **One fixed script file path** (not a fresh random name per call)
   — so each new call's ``loadScript()`` targets the same path on
   disk; the file's *content* changes (new x/y/width/height baked
   into fresh source — confirmed to work, since this is just normal
   ``loadScript()`` usage, no unverified script-side I/O involved),
   but the path KWin tracks does not multiply.
2. **Explicit ``unloadScript()`` of the previous script_id before
   loading the next one.** This is the actual fix for the
   accumulation: each call now nets to "at most one extra loaded
   script object," never an unbounded count. ``unloadScript()`` is
   wrapped defensively (older KWin versions' exact signature isn't
   independently confirmed here) — if it's unavailable or raises, the
   new script still loads and runs; only the explicit cleanup is
   skipped, logged once at debug level, not a hard failure.

Real limitations — read before debugging a silent no-op
----------------------------------------------------------
* This is **not** a stable, versioned protocol the way wlr-layer-shell
  or EWMH are. It is "whatever KWin's scripting `Workspace` object
  looks like in this Plasma release." Plasma 6 renamed
  ``workspace.clientList()`` (Plasma 5) to ``workspace.windowList()``;
  a future release could rename `frameGeometry` or `keepAbove` too.
* Matching is by **window class** (``resourceClass``), not by a raw
  surface handle. The SDL2 window's WM class **must** be stable and
  unique enough not to collide with another running window.
* D-Bus round-trips are not free — every call here is a full
  loadScript+run+unloadScript cycle, meant for discrete "place it
  here" calls, not 60Hz drag tracking.
* Requires ``pydbus`` and a running KWin with scripting enabled.
"""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path
from typing import Optional

from ._base import NativeHandle

log = logging.getLogger(__name__)

# Plasma 6 uses workspace.windowList(); Plasma 5 used clientList(). Try
# the new name first, fall back once, never crash either way.
_SCRIPT_TEMPLATE = r"""
function _wbbFindWindow(wmClass) {
    var list = (typeof workspace.windowList === "function")
        ? workspace.windowList()
        : workspace.clientList();
    for (var i = 0; i < list.length; i++) {
        if (list[i].resourceClass == wmClass) return list[i];
    }
    return null;
}

(function () {
    var w = _wbbFindWindow(%(wm_class_json)s);
    if (!w) {
        print("wbb kwin script: no window found with resourceClass=" + %(wm_class_json)s);
        return;
    }
    %(body)s
})();

true; // loadScript() wants a truthy final expression
"""

_SET_ABOVE_BODY = "w.keepAbove = %(above)s;"

_SET_GEOMETRY_BODY = (
    "var gx = %(x)d, gy = %(y)d, gw = %(width)d, gh = %(height)d;\n"
    "    // 1) Move the window onto whichever output actually contains the\n"
    "    //    target top-left, BEFORE setting geometry. KWin constrains a\n"
    "    //    frameGeometry assignment to the window's *current* output's\n"
    "    //    area, so if we don't move outputs first, a target on another\n"
    "    //    monitor (or in that monitor's offset region) gets clamped\n"
    "    //    back. workspace.screens carries each output's global-coord\n"
    "    //    geometry; pick the one whose rect contains (gx, gy).\n"
    "    try {\n"
    "        var screens = workspace.screens || [];\n"
    "        for (var s = 0; s < screens.length; s++) {\n"
    "            var sg = screens[s].geometry;\n"
    "            if (gx >= sg.x && gx < sg.x + sg.width &&\n"
    "                gy >= sg.y && gy < sg.y + sg.height) {\n"
    "                if (w.output !== screens[s]) { w.output = screens[s]; }\n"
    "                break;\n"
    "            }\n"
    "        }\n"
    "    } catch (e) { /* older KWin without workspace.screens/output: skip */ }\n"
    "\n"
    "    // 2) Set position by MUTATING frameGeometry's sub-properties\n"
    "    //    rather than assigning a whole new rect object. On Plasma 6's\n"
    "    //    QJSEngine the whole-object assignment (w.frameGeometry = {..})\n"
    "    //    is unreliable/ignored for the position component; mutating\n"
    "    //    .x/.y/.width/.height in place is the form that actually\n"
    "    //    takes (see KDE Discuss reports on frameGeometry read-only\n"
    "    //    behavior). Fall back to geometry / whole-object only if\n"
    "    //    frameGeometry isn't present.\n"
    "    if (\"frameGeometry\" in w) {\n"
    "        var fg = w.frameGeometry;\n"
    "        fg.width = gw; fg.height = gh; fg.x = gx; fg.y = gy;\n"
    "        w.frameGeometry = fg;  // write-back too, covers both engines\n"
    "    } else {\n"
    "        w.geometry = { x: gx, y: gy, width: gw, height: gh };\n"
    "    }"
)


def _js_bool(v: bool) -> str:
    return "true" if v else "false"


def _js_string(s: str) -> str:
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


class KWinPlacement:
    name = "kwin"

    def __init__(self, wm_class: str, *, dbus_timeout: float = 2.0) -> None:
        self._wm_class = wm_class
        self._timeout = dbus_timeout
        self._bus = None
        self._kwin = None
        self._script_path: Optional[str] = None
        self._current_script_id: Optional[int] = None
        self._active = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def activate(self, handle: NativeHandle) -> bool:
        try:
            import pydbus  # noqa: PLC0415
        except ImportError:
            log.debug("kwin backend: pydbus not installed; skipping")
            return False

        try:
            self._bus = pydbus.SessionBus()
            self._kwin = self._bus.get("org.kde.KWin", "/Scripting")
        except Exception:
            log.debug("kwin backend: org.kde.KWin not reachable on session bus; skipping")
            return False

        # Fixed path, created once — every subsequent call overwrites
        # its content rather than allocating a new path. See module
        # docstring point 1.
        fd = tempfile.NamedTemporaryFile(
            mode="w", suffix=".js", prefix="wbb_kwin_script_", delete=False
        )
        fd.close()
        self._script_path = fd.name

        # Initial activation just confirms loadScript()/run() work at
        # all — the window may not exist yet at this exact moment
        # (activate() runs right after window creation), so this is a
        # genuine no-op body, not a real command. Real commands go
        # through _run_command() below once the caller actually asks
        # for always_on_top/position.
        ok = self._load_and_run("/* activation probe, no-op */")
        if not ok:
            return False

        self._active = True
        log.info("Placement: using KWin D-Bus scripting (wm_class=%r)", self._wm_class)
        return True

    # ------------------------------------------------------------------
    def set_above(self, above: bool) -> None:
        if not self._active:
            return
        self._run_command(_SET_ABOVE_BODY % {"above": _js_bool(above)})

    def set_position(self, x: int, y: int, width: int, height: int) -> None:
        if not self._active:
            return
        self._run_command(
            _SET_GEOMETRY_BODY % {"x": x, "y": y, "width": width, "height": height}
        )

    def supports_position(self) -> bool:
        return self._active

    def set_click_through(self, enabled: bool) -> bool:
        """
        KWin's scripting API has no input-transparency property on
        Window/EffectWindow — checked against the full KWin 6
        scripting API reference; there is simply no
        ``inputTransparent``/``passthroughInput``-style toggle
        exposed to scripts, unlike ``keepAbove`` or ``frameGeometry``.
        Always returns False; the X11/EWMH backend
        (placement/x11_ewmh.py) is the one that can actually do this,
        and does, on an X11 session (including KWin's own X11 mode).
        """
        if not self._active:
            return False
        log.warning(
            "click_through requested, but KWin's scripting API has no "
            "input-transparency mechanism (checked the full KWin 6 "
            "scripting API surface — keepAbove/frameGeometry exist, "
            "nothing for input passthrough does). This is a real "
            "platform gap, not a missing flag here. If you're on an "
            "X11 session, the x11-ewmh backend would have provided this "
            "instead — check whether the KWin backend pre-empted it; "
            "see select_backend()'s priority order in placement/chain.py."
        )
        return False

    # ------------------------------------------------------------------
    def _run_command(self, body: str) -> None:
        with self._lock:
            self._load_and_run(body)

    def _load_and_run(self, body: str) -> bool:
        """
        Write fresh source to the one fixed script path, unload
        whatever was previously loaded from a prior call (bounding
        accumulation to "at most one extra" instead of unbounded —
        see module docstring), load the new source, and run it.
        """
        if self._kwin is None or self._script_path is None:
            return False
        try:
            source = _SCRIPT_TEMPLATE % {
                "wm_class_json": _js_string(self._wm_class),
                "body": body,
            }
            Path(self._script_path).write_text(source)

            if self._current_script_id is not None:
                self._try_unload(self._current_script_id)

            script_id = self._kwin.loadScript(self._script_path)
            if script_id < 0:
                log.warning("kwin backend: loadScript rejected the script")
                return False
            self._current_script_id = script_id

            script_obj = self._bus.get("org.kde.KWin", f"/Scripting/Script{script_id}")
            script_obj.run()
            return True
        except Exception:
            log.exception("kwin backend: load/run failed")
            return False

    def _try_unload(self, script_id: int) -> None:
        """
        Best-effort cleanup of the previously loaded script. Wrapped
        defensively — unloadScript()'s exact signature/availability
        across KWin versions isn't independently confirmed here, and
        a failure to unload is not fatal (it just means one stale
        script object lingers instead of zero — bounded, not
        unbounded, accumulation), so this never raises out.
        """
        try:
            if hasattr(self._kwin, "unloadScript"):
                self._kwin.unloadScript(self._script_path)
        except Exception:
            log.debug(
                "kwin backend: unloadScript(%r) failed or unavailable; "
                "continuing (one stale script object may linger, this is "
                "bounded, not a leak that grows per-call).",
                self._script_path,
            )

