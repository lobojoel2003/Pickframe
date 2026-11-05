from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helpers import crop_shift
from pickframe import models
from pickframe.analysis.embed import ClassicEmbedder, ClipEmbedder, create_embedder


def test_classic_unit_norm_and_deterministic(astronaut_rgb):
    embedder = ClassicEmbedder()
    a = embedder.embed(astronaut_rgb)
    b = embedder.embed(astronaut_rgb.copy())
    assert a.dtype == np.float32
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)
    assert np.array_equal(a, b)


def test_classic_similarity(astronaut_rgb, chelsea_rgb):
    embedder = ClassicEmbedder()
    base = embedder.embed(crop_shift(astronaut_rgb))
    for variant in (
        crop_shift(astronaut_rgb, 0.02, 0.0),
        crop_shift(astronaut_rgb, 0.0, -0.02, 1.05),
        np.clip(crop_shift(astronaut_rgb).astype(np.float32) * 0.8, 0, 255).astype(np.uint8),
    ):
        assert float(base @ embedder.embed(variant)) > embedder.default_similarity
    assert float(base @ embedder.embed(chelsea_rgb)) < embedder.default_similarity


def test_classic_handles_flat_image():
    vec = ClassicEmbedder().embed(np.zeros((64, 64, 3), np.uint8))
    assert np.all(np.isfinite(vec))


def _fake_clip(path: Path, with_projection: bool = True) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    weights = rng.standard_normal((3, 512)).astype(np.float32)
    nodes = [
        helper.make_node("ReduceMean", ["pixel_values"], ["pooled"], axes=[2, 3], keepdims=0),
        helper.make_node("MatMul", ["pooled", "w"], ["image_embeds"]),
        helper.make_node("Unsqueeze", ["image_embeds", "axis"], ["last_hidden_state"]),
    ]
    outputs = [
        helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["n", 1, 512]),
    ]
    outputs.append(
        helper.make_tensor_value_info(
            "image_embeds" if with_projection else "pooled_output", TensorProto.FLOAT, ["n", 512]
        )
    )
    if not with_projection:
        nodes.append(helper.make_node("Identity", ["image_embeds"], ["pooled_output"]))
    graph = helper.make_graph(
        nodes,
        "fake-clip",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["n", 3, 224, 224])],
        outputs,
        [
            numpy_helper.from_array(weights, "w"),
            numpy_helper.from_array(np.array([1], np.int64), "axis"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def test_clip_missing_model():
    assert not models.is_installed("clip")
    with pytest.raises(models.ModelError):
        ClipEmbedder()
    with pytest.raises(models.ModelError):
        create_embedder("clip")
    assert create_embedder("auto").name == "classic"
    assert create_embedder("classic").name == "classic"
    with pytest.raises(ValueError, match="unknown embedder"):
        create_embedder("bogus")


def test_fake_clip_model(astronaut_rgb, chelsea_rgb):
    _fake_clip(models.remote_path("clip"))
    embedder = create_embedder("auto")
    assert isinstance(embedder, ClipEmbedder)
    assert embedder._output == "image_embeds"
    vec = embedder.embed(astronaut_rgb)
    assert vec.shape == (512,)
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)
    assert float(vec @ embedder.embed(astronaut_rgb)) == pytest.approx(1.0, abs=1e-5)
    # wide and tall inputs are centre-cropped to 224 x 224
    assert embedder.preprocess(np.zeros((100, 300, 3), np.uint8)).shape == (1, 3, 224, 224)
    assert embedder.preprocess(np.zeros((900, 300, 3), np.uint8)).shape == (1, 3, 224, 224)


def test_fake_clip_without_named_projection():
    _fake_clip(models.remote_path("clip"), with_projection=False)
    embedder = ClipEmbedder()
    assert embedder._output == "pooled_output"  # the 2-D output wins over the 3-D one


def test_auto_falls_back_when_clip_is_broken():
    path = models.remote_path("clip")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not an onnx model")
    assert create_embedder("auto").name == "classic"
