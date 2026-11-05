#!/usr/bin/env python3
"""Render packaging/icons/pickframe.svg to PNG, ICO (Windows) and ICNS (macOS).

Needs Playwright with Chromium (``pip install playwright && playwright install chromium``).
Set ``PLAYWRIGHT_CHROMIUM`` to use a specific Chromium binary.
"""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ICONS = ROOT / "packaging" / "icons"
SVG = ICONS / "pickframe.svg"


def render(size: int) -> Image.Image:
    svg = SVG.read_text(encoding="utf-8")
    html = f"<html><body style='margin:0;background:transparent'>{svg.replace('<svg ', f'<svg style=\"width:{size}px;height:{size}px;display:block\" ', 1)}</body></html>"
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("PLAYWRIGHT_CHROMIUM") or None)
        page = browser.new_page(viewport={"width": size, "height": size})
        page.set_content(html)
        png = page.screenshot(omit_background=True, clip={"x": 0, "y": 0, "width": size, "height": size})
        browser.close()
    return Image.open(BytesIO(png)).convert("RGBA")


def main() -> None:
    master = render(1024)
    master.save(ICONS / "pickframe-1024.png")
    master.resize((512, 512), Image.Resampling.LANCZOS).save(ICONS / "pickframe-512.png")
    master.resize((256, 256), Image.Resampling.LANCZOS).save(ROOT / "src" / "pickframe" / "gui" / "icon.png")
    master.save(ICONS / "pickframe.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    master.save(ICONS / "pickframe.icns")
    print("icons written to", ICONS)


if __name__ == "__main__":
    main()
