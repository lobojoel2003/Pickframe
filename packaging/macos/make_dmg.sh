#!/usr/bin/env bash
# Wrap dist/Pickframe.app in a compressed disk image with an Applications link.
#   packaging/macos/make_dmg.sh dist/Pickframe.app Pickframe-1.0.0-macos-arm64.dmg
set -euo pipefail

APP="${1:?usage: make_dmg.sh <Pickframe.app> <output.dmg>}"
OUT="${2:?usage: make_dmg.sh <Pickframe.app> <output.dmg>}"
STAGE="$(mktemp -d)"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
# Ad-hoc signature so Apple Silicon runs the bundle; it is not notarised.
codesign --force --deep --sign - "$STAGE/Pickframe.app" || true
hdiutil create -volname "Pickframe" -srcfolder "$STAGE" -ov -format UDZO "$OUT"
rm -rf "$STAGE"
echo "built $OUT"
