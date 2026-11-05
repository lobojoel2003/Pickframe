#!/usr/bin/env python3
"""Regenerate the images used in README.md.

    pip install -e ".[dev]" playwright
    python tools/make_screenshots.py            # writes docs/images/*

Steps: build the demo shoot (tools/make_demo.py), analyse it, then capture
the desktop app (headless Qt), the HTML report (Chromium via Playwright),
the command-line output (Rich SVG export) and the banner.
Set PLAYWRIGHT_CHROMIUM to use a specific Chromium binary.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
ICON = ROOT / "packaging" / "icons" / "pickframe.svg"


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_png(png: bytes, path: Path, width: int | None = None) -> None:
    from PIL import Image

    image = Image.open(BytesIO(png)).convert("RGB")
    if width and image.width > width:
        image = image.resize(
            (width, round(image.height * width / image.width)), Image.Resampling.LANCZOS
        )
    image.save(path, optimize=True)
    print(f"wrote {path} ({image.width}x{image.height}, {path.stat().st_size // 1024} KB)")


def browser(playwright):
    return playwright.chromium.launch(executable_path=os.environ.get("PLAYWRIGHT_CHROMIUM") or None)


def app_screenshot(result, path: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_SCALE_FACTOR", "2")
    from PySide6.QtCore import QBuffer, QIODevice, QSettings, QTimer

    from pickframe.gui.app import MainWindow, create_app

    app = create_app([sys.argv[0]])
    with tempfile.TemporaryDirectory() as tmp:
        window = MainWindow(QSettings(str(Path(tmp) / "settings.ini"), QSettings.Format.IniFormat))
        window.resize(1360, 860)
        window.folder = Path(result.shots[0].primary.parent)
        window.setWindowTitle("Pickframe — wedding-demo")
        window.show()
        window.set_result(result)
        window.groups.setCurrentRow(1)  # the portrait burst
        blink = next(s.id for s in result.shots if result.decisions[s.id].flags == ["eyes-closed"])
        row = window.model.row_of(blink)
        window.grid.setCurrentIndex(window.model.index(row))
        captured: dict[str, bytes] = {}

        def grab() -> None:
            app.processEvents()
            buffer = QBuffer()
            buffer.open(QIODevice.OpenModeFlag.WriteOnly)
            window.grab().save(buffer, "PNG")
            captured["png"] = bytes(buffer.data())
            window.unsaved = False
            window.close()
            app.quit()

        QTimer.singleShot(2500, grab)
        app.exec()
    save_png(captured["png"], path, width=2000)


def report_screenshot(result, path: Path) -> None:
    from playwright.sync_api import sync_playwright

    from pickframe.report import write_html

    with tempfile.TemporaryDirectory() as tmp:
        html = Path(tmp) / "report.html"
        write_html(result, html)
        with sync_playwright() as p:
            b = browser(p)
            page = b.new_page(viewport={"width": 1240, "height": 900}, device_scale_factor=2)
            page.goto(html.as_uri())
            page.wait_for_timeout(500)
            png = page.screenshot(full_page=False)
            b.close()
    save_png(png, path, width=2000)


def cli_svg(result, folder_name: str, path: Path) -> None:
    from rich.console import Console

    from pickframe.cli import _print_list, _print_summary

    console = Console(
        record=True, width=96, force_terminal=True, color_system="truecolor", highlight=False
    )
    console.print(f"[bold green]$[/] pickframe cull {folder_name} --list")
    console.print()
    _print_summary(result, console)
    console.print()
    _print_list(result, console)
    console.print()
    console.print(
        f"Wrote XMP for [bold]{sum(len(s.files) for s in result.shots)}[/] file(s); 0 already up to date."
    )
    console.save_svg(str(path), title="pickframe")
    print(f"wrote {path}")


def banner(path: Path) -> None:
    from playwright.sync_api import sync_playwright

    svg = ICON.read_text(encoding="utf-8").replace("<svg ", '<svg class="icon" ', 1)
    html = f"""<html><head><style>
      body {{ margin:0; }}
      .b {{ width:1280px; height:360px; display:flex; align-items:center; gap:52px; padding:0 96px;
            box-sizing:border-box; background:radial-gradient(1200px 500px at 20% 0%, #232836 0%, #0f1115 60%);
            font-family: Inter, "Inter Display", "Segoe UI", Helvetica, Arial, sans-serif; color:#e8eaf0; }}
      .icon {{ width:190px; height:190px; flex:none; filter: drop-shadow(0 18px 40px rgba(255,181,71,.25)); }}
      h1 {{ font-size:92px; margin:0; letter-spacing:-2px; font-weight:800; }}
      h1 span {{ color:#ffb547; }}
      p {{ font-size:30px; margin:10px 0 22px; color:#c3c9d4; }}
      .chips {{ display:flex; gap:12px; }}
      .chip {{ font-size:18px; padding:7px 16px; border-radius:999px; border:1px solid #343a48; color:#9aa3b2; }}
    </style></head><body><div class="b">{svg}<div>
      <h1>Pick<span>frame</span></h1>
      <p>Free, local AI photo culling</p>
      <div class="chips"><span class="chip">Near-duplicate groups</span><span class="chip">Blur &amp; closed eyes</span>
      <span class="chip">XMP star ratings</span><span class="chip">100% offline</span></div>
    </div></div></body></html>"""
    with sync_playwright() as p:
        b = browser(p)
        page = b.new_page(viewport={"width": 1280, "height": 360}, device_scale_factor=2)
        page.set_content(html)
        page.wait_for_timeout(300)
        png = page.screenshot(clip={"x": 0, "y": 0, "width": 1280, "height": 360})
        b.close()
    save_png(png, path)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    make_demo = load_tool("make_demo")
    from pickframe.config import Settings
    from pickframe.pipeline import Analyzer

    with tempfile.TemporaryDirectory() as tmp:
        shoot = Path(tmp) / "wedding-demo"
        make_demo.main([str(shoot), "--cache", str(ROOT / ".cache" / "demo-sources")])
        result = Analyzer(Settings(embedder="classic"), use_cache=False).run([shoot])
        banner(OUT / "banner.png")
        cli_svg(result, shoot.name, OUT / "cli.svg")
        report_screenshot(result, OUT / "report.png")
        app_screenshot(result, OUT / "app.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
