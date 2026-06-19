"""
05_record_to_ffmpeg.py — Record frames to video via ffmpeg pipe.

Demonstrates:
  - Fully headless operation (no display window)
  - Piping raw RGBA frames to an ffmpeg subprocess for video encoding
  - Arbitrary numpy/cv2-style processing in the recording path
  - Clean shutdown on KeyboardInterrupt

Requires ffmpeg in PATH. No display window is opened.
"""

import asyncio
import subprocess
from wbb import BrowserBridge, FrameBuffer

URL = "https://example.com"
OUTPUT = "recording.mp4"
WIDTH, HEIGHT = 1280, 720
FPS = 30
DURATION_SECONDS = 10


async def main() -> None:
    buf = FrameBuffer("ex05", WIDTH, HEIGHT)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgba",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS),
        "-i", "-",           # read from stdin
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        OUTPUT,
    ]

    ffmpeg = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print(f"Recording {DURATION_SECONDS}s of {URL} → {OUTPUT}")
    frame_count = 0
    target_frames = DURATION_SECONDS * FPS

    async with BrowserBridge(buf, width=WIDTH, height=HEIGHT) as br:
        await br.navigate(URL)
        await br.wait_for_load()

        async for frame in buf:
            if frame_count >= target_frames:
                break

            # Raw RGBA bytes straight to ffmpeg stdin — no intermediate copy
            assert ffmpeg.stdin
            ffmpeg.stdin.write(frame.data.tobytes())
            frame_count += 1

            if frame_count % FPS == 0:
                elapsed = frame_count // FPS
                print(f"  {elapsed}/{DURATION_SECONDS}s", end="\r", flush=True)

    assert ffmpeg.stdin
    ffmpeg.stdin.close()
    ffmpeg.wait()
    buf.close()
    buf.unlink()
    print(f"\nSaved {frame_count} frames → {OUTPUT}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
