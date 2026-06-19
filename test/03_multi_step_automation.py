"""
03_multi_step_automation.py — Multi-step page interaction automation.

Demonstrates:
  - navigate → wait → type → click → eval → screenshot pipeline
  - JS evaluation to extract page state
  - conditional logic in user space
  - saving screenshots at specific steps
"""

import asyncio
from wbb import BrowserBridge, FrameBuffer

WIDTH, HEIGHT = 1280, 720


async def main() -> None:
    buf = FrameBuffer("ex03", WIDTH, HEIGHT)

    async with BrowserBridge(buf, width=WIDTH, height=HEIGHT, enable_input=True) as br:
        # Step 1: Navigate to a search engine
        print("Step 1: Navigate")
        await br.navigate("https://www.wikipedia.org/")

        frame = await buf.next_frame()
        frame.save("step1_loaded.png")
        print(f"  → Saved step1_loaded.png ({frame.width}×{frame.height})")

        # Step 2: Focus the search box and type a query
        print("Step 2: Type query")
        await br.click(600, 500)
        await asyncio.sleep(0.3)
        await br.type("Pythagorean th")
        for i in "eorem":
            await br.key(i)

        await asyncio.sleep(0.2)
        frame.save("step2_results.png")
        print("  → Saved step2_results.png")
        # This works as return
        await br.key("\r")

        # This doesnt work
        # await br.key("Enter", kind="keyDown")
        # await asyncio.sleep(0.1)
        # await br.key("Enter", kind="keyUp")

        # This again works
        # manual form submit since key does not work
        # await br.eval("document.querySelector('input[name=q]')?.closest('form')?.submit()")

        # Step 3: Wait for results
        print("Step 3: Wait for results")
        await asyncio.sleep(1.0)  # yes i know, bad programming

        frame = buf.read()

        frame.save("step3_results.png")
        print("  → Saved step3_results.png")

        await br.click(316, 310)
        # risky version
        # await asyncio.sleep(1.0)  # yes i know, bad programming
        # frame = buf.read()
        # better version
        await asyncio.sleep(10.0)  # imagine long loading here
        frame = await buf.next_frame(
            2
        )  # i intorduced/added/implemented the imeout, so long waiting can be done no problem
        frame.save("step4_results.png")
        print("  → Saved step4_results.png")

        del frame  # release the zero-copy view before the buffer closes

    buf.close()
    buf.unlink()


if __name__ == "__main__":
    asyncio.run(main())
