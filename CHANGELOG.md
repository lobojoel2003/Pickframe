# Changelog

All notable changes to Pickframe are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Every release
artifact also carries a build number (the release workflow's run number),
shown by `pickframe --version` and in the app's About box.

## [Unreleased]

## [1.0.0] - 2026-11-05

First public release.

### Added

- Desktop app (Windows, macOS, Linux) with a thumbnail grid, scene groups,
  a detail panel with face boxes and quality meters, keyboard rating
  (`1`-`5`, `0`, `X`, `U`), drag and drop, and an HTML/CSV/JSON report export.
- `pickframe` command-line tool: `cull`, `reset`, `info`, `models`, `cache`
  and `gui` commands, with dry runs and per-photo listings.
- Near-duplicate grouping of bursts and repeated poses using capture time and
  image similarity: a built-in classic descriptor, or the optional CLIP
  ViT-B/32 image encoder (`pickframe models download clip`).
- Blur detection that understands shallow depth of field (sharpest regions
  and eye focus count), plus burst-relative checks for camera shake and missed
  focus.
- Face analysis with bundled ONNX models (YuNet detector and MediaPipe face
  mesh + blendshapes): closed eyes, eye focus, smiles and head turn.
- Exposure checks for clipped highlights and crushed shadows.
- XMP output that Lightroom, darktable and digiKam read: star ratings,
  rejects (`xmp:Rating -1`), Lightroom 13.2+ pick/reject flags, digiKam pick
  labels and optional colour labels. Existing sidecars are merged, ratings you
  set by hand are kept, and `pickframe reset` undoes everything.
- RAW + JPEG pairs are treated as one shot; raw files are analysed through
  their embedded previews (rawpy optional), HEIC/AVIF through pillow-heif.
- Analysis cache, so re-running with different thresholds is instant.
- Release pipeline with version checks, build numbers, Windows installer,
  macOS disk image, Linux AppImage, Python wheel and SHA-256 checksums.

[Unreleased]: https://github.com/lobojoel2003/Pickframe/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/lobojoel2003/Pickframe/releases/tag/v1.0.0
