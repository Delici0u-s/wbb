#!/usr/bin/env python3
"""
bench_hotpath.py — before/after microbenchmarks for the wbb hot path.

Measures the four stages that were actually changed, each as an
old-implementation-vs-new-implementation pair on identical input, so you
can see the delta directly rather than trusting a single absolute number.

Stages
------
1. JPEG decode -> RGBA
   OLD: TurboJPEG decode to BGR, then allocate a separate RGBA array and
        copy three channels + fill alpha (or Pillow .convert('RGBA')).
   NEW: TurboJPEG decode straight to RGBA in one pass (or Pillow
        fallback, unchanged).
   Without libjpeg-turbo installed the bench runs the Pillow fallback for
   both sides — the decode delta then reflects only the removed
   BGR->RGBA channel shuffle that the turbo path used to do, which the
   Pillow path never did, so the two sides look identical. Install
   PyTurboJPEG to see the real decode win (see note printed at runtime).

2. Color filter chain (colorize -> contrast -> brightness)
   OLD: each filter does astype(float32) -> math -> clip -> astype(uint8),
        composed naively (one full pass + temporaries per filter).
   NEW: per-channel LUTs fused at chain-build time into a single 256x4
        table, applied as one uint8 gather per frame.

3. FrameBuffer write + reader wake throughput
   OLD: next_frame() parks its blocking Condition wait on asyncio's
        SHARED default executor (where decode also runs).
   NEW: a dedicated per-buffer wait pool, so parked readers never steal
        decode threads. The bench runs N concurrent readers alongside a
        synthetic "decode" load saturating the default pool, and reports
        end-to-end frames delivered — the contended case is where the
        two diverge.

4. SDL present-side copy (the ascontiguousarray guard)
   OLD: np.ascontiguousarray(arr) unconditionally — re-materialises the
        whole frame even when it's already contiguous (the common case).
   NEW: copy only when not already C-contiguous.
   (The SDL_LockTexture vs SDL_UpdateTexture change can't be measured
   without a live GPU/window, so this stage isolates just the CPU-side
   copy that was removable in pure Python.)

Usage
-----
    python benchmarks/bench_hotpath.py
    python benchmarks/bench_hotpath.py --width 1280 --height 720 --iters 200

All stages default to a 1280x720 RGBA frame (the README's reference
viewport). Times are reported as median per-frame plus frames/sec.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import importlib.util
import io
import statistics
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

# Optional libjpeg-turbo
try:
    from turbojpeg import TJPF_RGBA, TurboJPEG

    _turbo = TurboJPEG()
except Exception:  # ImportError or libturbojpeg-not-found
    _turbo = None
    TJPF_RGBA = 7  # value is correct even if unused


# ---------------------------------------------------------------------------
# Module loading that bypasses wbb/__init__.py (which imports sdl2)
# ---------------------------------------------------------------------------
def _load(name: str, relpath: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(ROOT / relpath))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Stand up a minimal 'wbb' package namespace so buffer.py's
# `from wbb.frame import Frame` / `from wbb._shm import ShmSegment` resolve
# without triggering the real __init__.
_wbb = types.ModuleType("wbb")
_wbb.__path__ = [str(ROOT / "wbb")]
sys.modules["wbb"] = _wbb
_shm = _load("wbb._shm", "wbb/_shm.py")
_frame = _load("wbb.frame", "wbb/frame.py")
filters = _load("wbb.filters", "wbb/filters.py")
buffer_mod = _load("wbb.buffer", "wbb/buffer.py")
FrameBuffer = buffer_mod.FrameBuffer


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
def _bench(fn, iters: int, warmup: int = 5) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    med = statistics.median(samples)
    fps = 1.0 / med if med > 0 else float("inf")
    return med, fps


def _report(label: str, old_med: float, new_med: float) -> None:
    old_fps = 1.0 / old_med if old_med else float("inf")
    new_fps = 1.0 / new_med if new_med else float("inf")
    speedup = old_med / new_med if new_med else float("inf")
    print(f"  OLD: {old_med * 1e3:8.3f} ms/frame  ({old_fps:7.1f} fps)")
    print(f"  NEW: {new_med * 1e3:8.3f} ms/frame  ({new_fps:7.1f} fps)")
    print(f"  -> {speedup:.2f}x  ({(1 - new_med / old_med) * 100:+.1f}% time)")


# ===========================================================================
# Stage 1: decode
# ===========================================================================
def _make_jpeg(width: int, height: int) -> bytes:
    rng = np.random.default_rng(1)
    # A gradient + noise image — compresses like a real screenshot, not a
    # flat color that decodes unrealistically fast.
    base = np.linspace(0, 255, width, dtype=np.uint8)
    img = np.tile(base, (height, 1))
    rgb = np.stack([img, np.roll(img, 50, 1), np.roll(img, 100, 1)], axis=2)
    rgb = (rgb.astype(np.int16) + rng.integers(-20, 20, rgb.shape)).clip(0, 255).astype(np.uint8)
    b = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(b, "JPEG", quality=80)
    return b.getvalue()


def _decode_old(raw: bytes, width: int, height: int) -> np.ndarray:
    if _turbo is not None:
        bgr = _turbo.decode(raw)  # default TJPF_BGR
        if bgr.shape[:2] == (height, width):
            rgba = np.empty((bgr.shape[0], bgr.shape[1], 4), dtype=np.uint8)
            rgba[..., 0] = bgr[..., 2]
            rgba[..., 1] = bgr[..., 1]
            rgba[..., 2] = bgr[..., 0]
            rgba[..., 3] = 255
            return rgba
        bgr = np.asarray(
            Image.fromarray(bgr[..., ::-1]).resize((width, height), Image.Resampling.LANCZOS)
        )
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[..., :3] = bgr
        rgba[..., 3] = 255
        return rgba
    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def _decode_new(raw: bytes, width: int, height: int) -> np.ndarray:
    if _turbo is not None:
        rgba = _turbo.decode(raw, pixel_format=TJPF_RGBA)
        if rgba.shape[:2] == (height, width):
            return rgba
        return np.asarray(
            Image.fromarray(rgba, mode="RGBA").resize((width, height), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )
    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def stage_decode(width: int, height: int, iters: int) -> None:
    print("\n[1] JPEG decode -> RGBA")
    raw = _make_jpeg(width, height)
    # correctness: both produce the same pixels
    a, b = _decode_old(raw, width, height), _decode_new(raw, width, height)
    assert a.shape == b.shape == (height, width, 4)
    assert (a[..., 3] == 255).all() and (b[..., 3] == 255).all()
    if _turbo is not None and not np.array_equal(a, b):
        print("  WARN: old/new decode differ (unexpected)")
    old = _bench(lambda: _decode_old(raw, width, height), iters)
    new = _bench(lambda: _decode_new(raw, width, height), iters)
    if _turbo is None:
        print("  (PyTurboJPEG not installed — both sides run the identical")
        print("   Pillow fallback; install wbb[fast-jpeg] to measure the real")
        print("   decode-to-RGBA win. Shown delta here is ~noise.)")
    _report("decode", old[0], new[0])


# ===========================================================================
# Stage 2: color filter chain
# ===========================================================================
def _old_colorize(r, g, b, a=1.0):
    muls = np.array([r, g, b, a], dtype=np.float32)
    return lambda fr: (fr.astype(np.float32) * muls).clip(0, 255).astype(np.uint8)


def _old_brightness(delta):
    def f(fr):
        out = np.asarray(fr.copy())
        rgb = out[..., :3].astype(np.int16)
        out[..., :3] = (rgb + delta).clip(0, 255).astype(np.uint8)
        return out

    return f


def _old_contrast(factor):
    def f(fr):
        out = np.asarray(fr.copy())
        rgb = out[..., :3].astype(np.float32)
        out[..., :3] = ((rgb - 128) * factor + 128).clip(0, 255).astype(np.uint8)
        return out

    return f


def stage_filters(width: int, height: int, iters: int) -> None:
    print("\n[2] Color filter chain: colorize -> contrast -> brightness")
    rng = np.random.default_rng(2)
    frame = rng.integers(0, 256, (height, width, 4), dtype=np.uint8)

    oc, oct_, ob = _old_colorize(0.95, 0.95, 1.1), _old_contrast(1.15), _old_brightness(12)

    def old_chain(fr):
        return ob(oct_(oc(fr)))

    new_chain = filters.chain(
        filters.colorize(0.95, 0.95, 1.1),
        filters.contrast(1.15),
        filters.brightness(12),
    )
    # correctness
    assert np.array_equal(old_chain(frame), new_chain(frame)), "filter chain mismatch!"
    assert isinstance(new_chain, filters._LutFilter), "chain did not fuse to one LUT"

    old = _bench(lambda: old_chain(frame), iters)
    new = _bench(lambda: new_chain(frame), iters)
    _report("filters", old[0], new[0])


# ===========================================================================
# Stage 3: buffer write + reader wake under decode-pool contention
# ===========================================================================
def _busy_default_pool(stop: threading.Event):
    """Occupy asyncio's default-executor workers with synthetic 'decode'
    work, the way real JPEG decodes would. Each worker does short bursts
    of blocking sleep (standing in for time spent in a C decode call that
    holds the worker but not the GIL), so a reader that *also* parks on
    this same pool has to queue behind them.

    Bounded occupancy (sleeps, not an infinite CPU spin) so this produces
    proportional contention on any core count instead of a hard deadlock
    on a 1-core box.
    """
    loop = asyncio.get_event_loop()
    default = loop._default_executor or ThreadPoolExecutor()  # type: ignore[attr-defined]
    n_workers = default._max_workers  # type: ignore[attr-defined]

    def occupy():
        while not stop.is_set():
            time.sleep(0.003)  # ~a decode's worth of worker-held time

    return [loop.run_in_executor(None, occupy) for _ in range(n_workers)]


async def _run_buffer_variant(use_default_pool: bool, width, height, n_readers, n_frames):
    """Drive n_readers consumers against one buffer while a writer pushes
    n_frames. Returns total frames delivered and wall time.

    use_default_pool=True monkeypatches next_frame to park on the shared
    default executor (the OLD behavior); False uses the buffer's own
    dedicated wait pool (NEW).
    """
    buf = FrameBuffer(f"bench_buf_{int(use_default_pool)}_{time.monotonic_ns()}", width, height)

    if use_default_pool:
        # Re-create OLD behavior: park on the shared default executor.
        async def old_next_frame(timeout=5.0):
            loop = asyncio.get_running_loop()

            def _wait(seen):
                with buf._cv:
                    return buf._cv.wait_for(lambda: buf._generation != seen, timeout=timeout)

            seen = buf._generation
            await loop.run_in_executor(None, _wait, seen)
            return buf.read()

        buf.next_frame = old_next_frame  # type: ignore

    # Saturate the default pool with synthetic decode load in BOTH cases —
    # this is the contention the dedicated pool is meant to dodge.
    stop = threading.Event()
    hog_futs = _busy_default_pool(stop)

    delivered = [0] * n_readers
    frame_payload = np.zeros((height, width, 4), np.uint8)

    async def reader(i):
        last = -1
        # read until we've seen n_frames distinct ids or time out hard
        deadline = time.monotonic() + 10
        while delivered[i] < n_frames and time.monotonic() < deadline:
            fr = await buf.next_frame(timeout=1.0)
            if fr.frame_id != last:
                last = fr.frame_id
                delivered[i] += 1

    async def writer():
        for k in range(n_frames):
            frame_payload[0, 0, 0] = k % 256
            buf.write(frame_payload)
            await asyncio.sleep(0.002)  # ~500fps cap, leaves room to observe wake latency

    t0 = time.monotonic()
    readers = [asyncio.create_task(reader(i)) for i in range(n_readers)]
    w = asyncio.create_task(writer())
    await w
    # give readers a moment to drain
    await asyncio.sleep(0.2)
    for r in readers:
        r.cancel()
    stop.set()
    for f in hog_futs:
        f.cancel()
    elapsed = time.monotonic() - t0

    total = sum(delivered)
    buf.close()
    buf.unlink()
    return total, elapsed


async def stage_buffer(width, height, iters) -> None:
    print("\n[3] Buffer write + reader wake under decode-pool contention")
    n_readers = 4
    n_frames = 300
    # smaller frame keeps this stage fast; wake latency is size-independent
    w, h = 64, 48
    old_total, old_t = await _run_buffer_variant(True, w, h, n_readers, n_frames)
    new_total, new_t = await _run_buffer_variant(False, w, h, n_readers, n_frames)
    print(f"  {n_readers} readers, {n_frames} frames each, default pool saturated")
    print(f"  OLD (park on shared default pool): {old_total:5d} frames in {old_t:.2f}s "
          f"({old_total / old_t:.0f}/s)")
    print(f"  NEW (dedicated wait pool):         {new_total:5d} frames in {new_t:.2f}s "
          f"({new_total / new_t:.0f}/s)")
    if old_total:
        print(f"  -> {new_total / max(old_total, 1):.2f}x frames delivered under contention")


# ===========================================================================
# Stage 4: present-side contiguity guard
# ===========================================================================
def stage_present(width, height, iters) -> None:
    print("\n[4] SDL present path")
    rng = np.random.default_rng(4)
    contiguous = np.ascontiguousarray(rng.integers(0, 256, (height, width, 4), dtype=np.uint8))

    # Honesty check: on modern numpy, np.ascontiguousarray() on an
    # already-C-contiguous, matching-dtype array returns the SAME object
    # without copying. So the OLD push_frame's unconditional
    # ascontiguousarray was NOT actually copying in the common case —
    # meaning the explicit dtype/contiguity guard we added is a
    # correctness/clarity change, not a measurable CPU win on its own.
    probe = np.ascontiguousarray(contiguous, dtype=np.uint8)
    no_copy = probe is contiguous
    print(f"  ascontiguousarray on contiguous input copies? {'no' if no_copy else 'YES'}")
    print("  -> The contiguity-guard change is therefore ~0 CPU delta in the")
    print("     common (already-contiguous) case; numpy already no-op'd it.")
    print("  The real present-side win is SDL_LockTexture (write straight")
    print("  into GPU staging memory) replacing SDL_UpdateTexture's extra")
    print("  driver-side copy — that needs a live GPU/window to measure and")
    print("  is not benchmarked here.")

    # For reference, show the cost that IS paid when a zero-copy filter
    # (crop/flip) hands push_frame a non-contiguous view — both old and
    # new must copy it; the guard doesn't change that, it's shown only so
    # the number isn't mysterious.
    view = contiguous[:, ::-1]  # non-contiguous (horizontal flip)
    assert not view.flags["C_CONTIGUOUS"]
    cost = _bench(lambda: np.ascontiguousarray(view, dtype=np.uint8), max(iters, 30))
    print(f"  (ref: forced copy of a non-contiguous flip view: "
          f"{cost[0] * 1e3:.3f} ms — unchanged old vs new)")


# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    print("=" * 66)
    print(f"wbb hot-path benchmark — {args.width}x{args.height}, {args.iters} iters")
    print(f"libjpeg-turbo: {'present' if _turbo is not None else 'NOT installed (Pillow fallback)'}")
    print(f"numpy: {np.__version__}")
    print("=" * 66)

    stage_decode(args.width, args.height, args.iters)
    stage_filters(args.width, args.height, args.iters)
    asyncio.run(stage_buffer(args.width, args.height, args.iters))
    stage_present(args.width, args.height, args.iters)

    print("\n" + "=" * 66)
    print("Done. Stages 1-2 & 4 are CPU/copy deltas; stage 3 is a")
    print("contention/throughput delta (the dedicated wait pool only helps")
    print("when the default pool is actually saturated by concurrent decode).")
    print("=" * 66)


if __name__ == "__main__":
    main()
