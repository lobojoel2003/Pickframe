# PyInstaller spec for Pickframe: one folder with the desktop app (windowed)
# and the command-line tool (console), sharing the same libraries.
#
#   pyinstaller packaging/pyinstaller/pickframe.spec --noconfirm
#
# Environment:
#   PICKFRAME_VERSION_FILE  Windows version resource (tools/stamp_build.py)
#   PICKFRAME_BUILD         build number for the macOS bundle
import os
import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH).resolve().parents[1]
ICONS = ROOT / "packaging" / "icons"
VERSION = re.search(r'__version__ = "([^"]+)"', (ROOT / "src/pickframe/__init__.py").read_text()).group(1)
BUILD = os.environ.get("PICKFRAME_BUILD", "0")
VERSION_FILE = os.environ.get("PICKFRAME_VERSION_FILE") or None

datas = collect_data_files("pickframe", includes=["models/*.onnx", "models/*.md", "gui/*.png"])
datas += collect_data_files("certifi")
binaries = collect_dynamic_libs("onnxruntime")
hiddenimports = ["pickframe.gui.app", "certifi"]
for optional in ("rawpy", "pillow_heif"):
    try:
        __import__(optional)
        hiddenimports.append(optional)
        binaries += collect_dynamic_libs(optional)
    except ImportError:
        pass

excludes = [
    "tkinter", "matplotlib", "IPython", "pytest", "scipy", "pandas",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtQuick",
    "PySide6.QtQml", "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization",
    "PySide6.QtPdf", "PySide6.QtBluetooth", "PySide6.QtSql", "PySide6.QtTest",
]

def analysis(script):
    return Analysis(
        [str(ROOT / "packaging" / "pyinstaller" / script)],
        pathex=[str(ROOT / "src")],
        binaries=binaries,
        datas=datas,
        hiddenimports=hiddenimports,
        excludes=excludes,
        noarchive=False,
    )

gui_a = analysis("gui_entry.py")
cli_a = analysis("cli_entry.py")

# Qt ships many modules Pickframe never loads (QML/Quick, PDF, virtual
# keyboard, GTK theme); dropping them saves ~40 MB per platform.
UNUSED = re.compile(
    r"Qt6?(Quick|Qml|QmlMeta|QmlModels|QmlWorkerScript|Pdf|VirtualKeyboard|VirtualKeyboardQml)(\.|\b|_)"
    r"|libqtvirtualkeyboard|qtvirtualkeyboard|libqpdf|qpdf\.dll|libqgtk3|/translations/(?!qtbase_)",
    re.IGNORECASE,
)


def prune(toc):
    return [entry for entry in toc if not UNUSED.search(entry[0].replace("\\", "/"))]


for a in (gui_a, cli_a):
    a.binaries = prune(a.binaries)
    a.datas = prune(a.datas)

icon = None
if sys.platform == "win32":
    icon = str(ICONS / "pickframe.ico")
elif sys.platform == "darwin":
    icon = str(ICONS / "pickframe.icns")

gui_exe = EXE(
    PYZ(gui_a.pure),
    gui_a.scripts,
    [],
    exclude_binaries=True,
    name="Pickframe",
    console=False,
    icon=icon,
    version=VERSION_FILE,
    upx=False,
)
cli_exe = EXE(
    PYZ(cli_a.pure),
    cli_a.scripts,
    [],
    exclude_binaries=True,
    name="pickframe-cli",
    console=True,
    icon=icon,
    version=VERSION_FILE,
    upx=False,
)
coll = COLLECT(
    gui_exe,
    gui_a.binaries,
    gui_a.datas,
    cli_exe,
    cli_a.binaries,
    cli_a.datas,
    name="Pickframe",
    upx=False,
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Pickframe.app",
        icon=icon,
        bundle_identifier="io.github.lobojoel2003.pickframe",
        version=VERSION,
        info_plist={
            "CFBundleName": "Pickframe",
            "CFBundleDisplayName": "Pickframe",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": BUILD,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            "NSHumanReadableCopyright": "GPL-3.0-or-later",
            "LSApplicationCategoryType": "public.app-category.photography",
        },
    )
