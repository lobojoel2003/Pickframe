#!/usr/bin/env bash
# Package dist/Pickframe (PyInstaller output) as an AppImage.
#   packaging/linux/build_appimage.sh dist/Pickframe Pickframe-1.0.0-linux-x86_64.AppImage
set -euo pipefail

SRC="${1:?usage: build_appimage.sh <pyinstaller-dir> <output.AppImage>}"
OUT="${2:?usage: build_appimage.sh <pyinstaller-dir> <output.AppImage>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WORK="$(mktemp -d)"
APPDIR="$WORK/Pickframe.AppDir"

mkdir -p "$APPDIR/usr/lib" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/512x512/apps"
cp -a "$SRC" "$APPDIR/usr/lib/pickframe"
cp "$HERE/pickframe.desktop" "$APPDIR/pickframe.desktop"
cp "$HERE/pickframe.desktop" "$APPDIR/usr/share/applications/pickframe.desktop"
cp "$ROOT/packaging/icons/pickframe-512.png" "$APPDIR/pickframe.png"
cp "$ROOT/packaging/icons/pickframe-512.png" "$APPDIR/usr/share/icons/hicolor/512x512/apps/pickframe.png"
ln -s pickframe.png "$APPDIR/.DirIcon"

cat > "$APPDIR/AppRun" <<'RUN'
#!/bin/sh
# "Pickframe.AppImage cli ..." runs the command-line tool; anything else opens the app.
HERE="$(dirname "$(readlink -f "$0")")"
if [ "${1:-}" = "cli" ]; then
  shift
  exec "$HERE/usr/lib/pickframe/pickframe-cli" "$@"
fi
exec "$HERE/usr/lib/pickframe/Pickframe" "$@"
RUN
chmod +x "$APPDIR/AppRun"

TOOL="${APPIMAGETOOL:-$WORK/appimagetool}"
if [ ! -x "$TOOL" ]; then
  curl -fsSL -o "$TOOL" \
    "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"
  chmod +x "$TOOL"
fi
ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 "$TOOL" --no-appstream "$APPDIR" "$OUT"
rm -rf "$WORK"
echo "built $OUT"
