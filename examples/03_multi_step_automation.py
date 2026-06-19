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
        await br.navigate("https://duckduckgo.com")
        await br.wait_for_load()

        frame = buf.read()
        frame.save("step1_loaded.png")
        print(f"  → Saved step1_loaded.png ({frame.width}×{frame.height})")

        # Step 2: Focus the search box and type a query
        print("Step 2: Type query")
        await br.click(640, 300)
        await asyncio.sleep(0.3)
        await br.type("WebView Buffer Bridge Python")
        await asyncio.sleep(0.2)
        await br.key("Enter")

        # Step 3: Wait for results
        print("Step 3: Wait for results")
        await br.wait_for_load(timeout=10)
        await asyncio.sleep(1.0)

        frame = buf.read()
        frame.save("step3_results.png")
        print("  → Saved step3_results.png")

        # Step 4: Extract result count via JS
        count_text = await br.eval(
            "document.querySelector('.result__title') ? "
            "document.querySelectorAll('.result__title').length.toString() : '0'"
        )
        print(f"Step 4: Results on page: {count_text}")

        # Step 5: Click the first result
        print("Step 5: Click first result")
        await br.eval("document.querySelector('.result__title a')?.click()")
        await br.wait_for_load(timeout=15)
        await asyncio.sleep(2.0)

        frame = buf.read()
        frame.save("step5_destination.png")
        print("  → Saved step5_destination.png")
        print("Done.")
        del frame  # release the zero-copy view before the buffer closes

    buf.close()
    buf.unlink()


if __name__ == "__main__":
    asyncio.run(main())
