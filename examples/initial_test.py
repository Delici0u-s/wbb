import asyncio
from wbb import BrowserBridge, FrameBuffer


async def main():
    buf = FrameBuffer("my_buf", 1280, 720)
    async with BrowserBridge(buf) as br:
        await br.navigate("https://gohugo.io/")

        frame = await buf.next_frame()

        frame.save("screenshot.png")
    buf.close()
    buf.unlink()


asyncio.run(main())
