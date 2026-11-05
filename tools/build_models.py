#!/usr/bin/env python3
"""Rebuild the ONNX models bundled in ``src/pickframe/models`` from their sources.

1. YuNet face detector (MIT) from the OpenCV model zoo. Its input is made
   dynamic so it can run at any resolution on ONNX Runtime.
2. MediaPipe Face Landmarker (Apache-2.0): face mesh + blendshapes, converted
   from TFLite with ``tools/convert_mediapipe.py``.

Usage::

    pip install onnx tflite ai-edge-litert   # ai-edge-litert only for verification
    python tools/build_models.py
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
import urllib.request
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert_mediapipe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "pickframe" / "models"

SOURCES = {
    "yunet": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "face_landmarker": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
}


def fetch(url: str, sha256: str | None, dest: Path) -> Path:
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if sha256 and digest != sha256:
        raise SystemExit(f"checksum mismatch for {url}: {digest}")
    print(f"  sha256 {digest}")
    dest.write_bytes(data)
    return dest


def make_yunet_dynamic(src: Path, dst: Path) -> None:
    model = onnx.load(str(src))
    dims = model.graph.input[0].type.tensor_type.shape.dim
    dims[2].dim_param = "height"
    dims[3].dim_param = "width"
    for output in model.graph.output:  # e.g. cls_8 is [1, H/8 * W/8, 1]
        out_dims = output.type.tensor_type.shape.dim
        out_dims[1].dim_param = f"anchors_{output.name.split('_')[-1]}"
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        yunet = fetch(*SOURCES["yunet"], tmpdir / "yunet.onnx")
        make_yunet_dynamic(yunet, OUT / "face_detection_yunet.onnx")
        task = fetch(*SOURCES["face_landmarker"], tmpdir / "face_landmarker.task")
        convert_mediapipe.main([str(task), str(OUT)])
    for path in sorted(OUT.glob("*.onnx")):
        print(f"{path.name}: {hashlib.sha256(path.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
