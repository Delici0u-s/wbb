"""
05_record_to_ffmpeg.py — Record frames to video via ffmpeg pipe.
"""

import asyncio
import subprocess
import sys
import time

from wbb import BrowserBridge, FrameBuffer

URL = "https://www.clocktab.com/"
OUTPUT = "recording.mp4"
WIDTH, HEIGHT = 1280, 720
FPS = 30
DURATION_SECONDS = 10


async def main() -> None:
    buf = FrameBuffer("ex05", WIDTH, HEIGHT)

    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            f"{WIDTH}x{HEIGHT}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-vf",
            "scale=in_range=full:out_range=full",
            "-c:v",
            "libopenh264",
            "-pix_fmt",
            "yuv420p",
            "-color_range",
            "pc",
            "-g",
            str(FPS),
            "-movflags",
            "+faststart",
            OUTPUT,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    print(f"Recording {DURATION_SECONDS}s of {URL} -> {OUTPUT}")

    frame_count = 0
    load_event = asyncio.Event()

    try:
        async with BrowserBridge(buf, width=WIDTH, height=HEIGHT) as br:
            br.on("load", lambda: load_event.set())
            await br.navigate(URL)
            await asyncio.wait_for(load_event.wait(), timeout=15)
            await asyncio.sleep(1.5)  # let JS-rendered content settle

            start = time.monotonic()
            next_slot = 0.0
            frame_interval = 1.0 / FPS

            while (elapsed := time.monotonic() - start) < DURATION_SECONDS:
                frame = buf.read()
                payload = frame.data.tobytes()
                del frame

                # Real-time pacing: write as many 1/FPS slots as have
                # elapsed, re-using the latest frame if Chrome hasn't
                # pushed a new one yet (e.g. during its 1fps idle throttle)
                while next_slot <= elapsed:
                    ffmpeg.stdin.write(payload)
                    frame_count += 1
                    next_slot += frame_interval

                del payload
                await asyncio.sleep(0.01)
    finally:
        if ffmpeg.stdin and not ffmpeg.stdin.closed:
            ffmpeg.stdin.close()
        _, stderr_bytes = ffmpeg.communicate(timeout=15)
        buf.close()
        buf.unlink()

    if ffmpeg.returncode != 0:
        print(f"ffmpeg exited with code {ffmpeg.returncode}", file=sys.stderr)
        print(stderr_bytes.decode(errors="replace")[-3000:], file=sys.stderr)
        sys.exit(1)

    print(f"Saved {frame_count} frames -> {OUTPUT}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
