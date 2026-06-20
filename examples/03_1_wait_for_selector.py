"""
03_1_wait_for_selector.py — Wait for a DOM element before interacting.

Demonstrates:
  - wait_for_selector()
  - click_element()
  - replacing arbitrary sleep() calls with DOM-based waiting
"""

import asyncio

from wbb import BrowserBridge, FrameBuffer

WIDTH, HEIGHT = 1280, 720


async def main() -> None:
    buf = FrameBuffer("ex03_1", WIDTH, HEIGHT)

    async with BrowserBridge(buf, width=WIDTH, height=HEIGHT, enable_input=True) as br:
        print("Navigate to Wikipedia")
        await br.navigate("https://www.wikipedia.org/")

        print("Waiting for search button icon...")
        found = await br.wait_for_selector(
            'i[data-jsl10n="search-input-button"]',
        )

        if not found:
            print("Search button never appeared!")
            return

        print("Found search button")

        frame = await buf.next_frame(2)
        frame.save("wait_for_selector_loaded.png")
        print("  → Saved wait_for_selector_loaded.png")

        print("Clicking search input...")
        clicked = await br.click_element("button.pure-button.pure-button-primary-progressive")
        # also possible
        # clicked = await br.click_element('button[type="submit"]')

        print(f"  → clicked: {clicked}")

        await asyncio.sleep(0.5)

        frame = await buf.next_frame(2)
        frame.save("wait_for_selector_clicked.png")
        print("  → Saved wait_for_selector_clicked.png")

        del frame

    buf.close()
    buf.unlink()


if __name__ == "__main__":
    asyncio.run(main())
