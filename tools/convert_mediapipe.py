#!/usr/bin/env python3
"""Convert the MediaPipe Face Landmarker models to ONNX.

Pickframe ships two models from Google's MediaPipe Face Landmarker
(Apache-2.0) converted to ONNX so they run on ONNX Runtime:

* ``face_landmarks.onnx``  - 478-point face mesh (input 1x256x256x3, RGB in [0, 1])
* ``face_blendshapes.onnx`` - 52 ARKit-style blendshapes (eye blink, smile, ...)

The converter below supports exactly the TFLite operators these two models
use. It keeps TFLite's NHWC layout and wraps convolutions / pooling in
transposes; ONNX Runtime folds those away when it optimises the graph.

Usage::

    python tools/convert_mediapipe.py face_landmarker.task src/pickframe/models

The ``.task`` bundle can be downloaded from
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

If ``ai-edge-litert`` is installed, the converted models are checked against
the original TFLite models on random and structured inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
import tflite
from onnx import TensorProto, helper, numpy_helper
from tflite.ActivationFunctionType import ActivationFunctionType as Act
from tflite.BuiltinOperator import BuiltinOperator as Op
from tflite.Padding import Padding
from tflite.TensorType import TensorType

OPSET = 17

_OP_NAMES = {v: k for k, v in vars(Op).items() if not k.startswith("_")}
_DTYPES = {
    TensorType.FLOAT32: np.float32,
    TensorType.FLOAT16: np.float16,
    TensorType.INT32: np.int32,
    TensorType.INT64: np.int64,
    TensorType.UINT8: np.uint8,
    TensorType.INT8: np.int8,
}


class Converter:
    """Translate one TFLite subgraph into an ONNX graph."""

    def __init__(self, buf: bytes, input_names: list[str], output_names: list[str]):
        self.model = tflite.Model.GetRootAsModel(buf, 0)
        self.graph = self.model.Subgraphs(0)
        self.nodes: list[onnx.NodeProto] = []
        self.consts: dict[int, np.ndarray] = {}
        self.initializers: dict[str, np.ndarray] = {}
        self.casted: set[str] = set()
        self.counter = 0
        self.input_names = input_names
        self.output_names = output_names
        self.renames: dict[int, str] = {}
        for i in range(self.graph.InputsLength()):
            self.renames[self.graph.Inputs(i)] = input_names[i]
        for i in range(self.graph.OutputsLength()):
            self.renames[self.graph.Outputs(i)] = output_names[i]
        self._load_constants()

    # -- helpers ---------------------------------------------------------
    def _load_constants(self) -> None:
        for idx in range(self.graph.TensorsLength()):
            t = self.graph.Tensors(idx)
            data = self.model.Buffers(t.Buffer()).DataAsNumpy()
            if isinstance(data, np.ndarray) and data.size:
                arr = np.frombuffer(data.tobytes(), dtype=_DTYPES[t.Type()])
                shape = [int(v) for v in t.ShapeAsNumpy()] if t.ShapeLength() else []
                self.consts[idx] = arr.reshape(shape).copy()

    def name(self, idx: int) -> str:
        if idx in self.renames:
            return self.renames[idx]
        if idx in self.consts:
            n = f"c{idx}"
            if n not in self.initializers and n not in self.casted:
                self.casted.add(n)
                return self.weight(self.consts[idx], n)
            return n
        return f"t{idx}"

    def weight(self, arr: np.ndarray, name: str | None = None) -> str:
        """Store a float tensor as float16 and cast it back to float32 in-graph.

        Halves the file size; ONNX Runtime constant-folds the casts on load.
        """
        if name is None:
            self.counter += 1
            name = f"w_{self.counter}"
        if arr.dtype.kind != "f" or arr.size < 64:
            self.initializers[name] = arr.astype(np.float32) if arr.dtype.kind == "f" else arr
            return name
        self.initializers[name + "_fp16"] = arr.astype(np.float16)
        self.emit("Cast", [name + "_fp16"], [name], to=TensorProto.FLOAT)
        return name

    def const(self, arr: np.ndarray, hint: str = "k") -> str:
        self.counter += 1
        n = f"{hint}_{self.counter}"
        self.initializers[n] = arr
        return n

    def tmp(self, hint: str = "x") -> str:
        self.counter += 1
        return f"{hint}_{self.counter}"

    def shape(self, idx: int) -> list[int]:
        t = self.graph.Tensors(idx)
        return [int(v) for v in t.ShapeAsNumpy()] if t.ShapeLength() else []

    def emit(self, op_type: str, inputs: list[str], outputs: list[str], **attrs) -> None:
        self.nodes.append(helper.make_node(op_type, inputs, outputs, **attrs))

    def activation(self, x: str, out: str, act: int) -> None:
        if act == Act.NONE:
            self.emit("Identity", [x], [out])
        elif act == Act.RELU:
            self.emit("Relu", [x], [out])
        elif act == Act.RELU6:
            self.emit("Clip", [x, self.const(np.float32(0)), self.const(np.float32(6))], [out])
        elif act == Act.RELU_N1_TO_1:
            self.emit("Clip", [x, self.const(np.float32(-1)), self.const(np.float32(1))], [out])
        elif act == Act.TANH:
            self.emit("Tanh", [x], [out])
        else:
            raise NotImplementedError(f"fused activation {act}")

    @staticmethod
    def same_pads(in_hw: list[int], k: list[int], s: list[int], d: list[int]) -> list[int]:
        begins, ends = [], []
        for size, kernel, stride, dil in zip(in_hw, k, s, d):
            out = -(-size // stride)
            total = max((out - 1) * stride + (kernel - 1) * dil + 1 - size, 0)
            begins.append(total // 2)
            ends.append(total - total // 2)
        return begins + ends

    def nchw(self, x: str) -> str:
        y = self.tmp("nchw")
        self.emit("Transpose", [x], [y], perm=[0, 3, 1, 2])
        return y

    def nhwc(self, x: str, out: str) -> None:
        self.emit("Transpose", [x], [out], perm=[0, 2, 3, 1])

    # -- operators -------------------------------------------------------
    def convert(self) -> onnx.ModelProto:
        for i in range(self.graph.OperatorsLength()):
            op = self.graph.Operators(i)
            oc = self.model.OperatorCodes(op.OpcodeIndex())
            code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
            ins = [op.Inputs(j) for j in range(op.InputsLength())]
            outs = [op.Outputs(j) for j in range(op.OutputsLength())]
            handler = getattr(self, f"op_{_OP_NAMES[code].lower()}", None)
            if handler is None:
                raise NotImplementedError(f"TFLite operator {_OP_NAMES[code]}")
            handler(op, ins, outs)

        def value_info(idx: int, name: str):
            return helper.make_tensor_value_info(name, TensorProto.FLOAT, self.shape(idx))

        inputs = [
            value_info(self.graph.Inputs(i), self.input_names[i])
            for i in range(self.graph.InputsLength())
        ]
        outputs = [
            value_info(self.graph.Outputs(i), self.output_names[i])
            for i in range(self.graph.OutputsLength())
        ]
        inits = [numpy_helper.from_array(v, k) for k, v in self.initializers.items()]
        graph = helper.make_graph(self.nodes, "mediapipe", inputs, outputs, inits)
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", OPSET)],
            producer_name="pickframe-convert-mediapipe",
        )
        model.ir_version = 8
        onnx.checker.check_model(model)
        return model

    def op_dequantize(self, op, ins, outs) -> None:
        if ins[0] in self.consts:
            self.consts[outs[0]] = self.consts[ins[0]].astype(np.float32)
        else:
            self.emit("Cast", [self.name(ins[0])], [self.name(outs[0])], to=TensorProto.FLOAT)

    def _conv(self, op, ins, outs, depthwise: bool) -> None:
        opts_cls = tflite.DepthwiseConv2DOptions if depthwise else tflite.Conv2DOptions
        o = opts_cls()
        t = op.BuiltinOptions()
        o.Init(t.Bytes, t.Pos)
        w = self.consts[ins[1]].astype(np.float32)
        in_shape = self.shape(ins[0])
        if depthwise:
            w = w.transpose(3, 0, 1, 2)  # [1,kh,kw,C*M] -> [C*M,1,kh,kw]
            group = in_shape[3]
        else:
            w = w.transpose(0, 3, 1, 2)  # [O,kh,kw,I] -> [O,I,kh,kw]
            group = 1
        k = [w.shape[2], w.shape[3]]
        s = [o.StrideH(), o.StrideW()]
        d = [o.DilationHFactor(), o.DilationWFactor()]
        pads = self.same_pads(in_shape[1:3], k, s, d) if o.Padding() == Padding.SAME else [0, 0, 0, 0]
        conv_in = [self.nchw(self.name(ins[0])), self.weight(w)]
        if len(ins) > 2 and ins[2] >= 0:
            conv_in.append(self.weight(self.consts[ins[2]]))
        y = self.tmp("conv")
        self.emit("Conv", conv_in, [y], kernel_shape=k, strides=s, dilations=d, pads=pads, group=group)
        a = self.tmp("act")
        self.activation(y, a, o.FusedActivationFunction())
        self.nhwc(a, self.name(outs[0]))

    def op_conv_2d(self, op, ins, outs) -> None:
        self._conv(op, ins, outs, depthwise=False)

    def op_depthwise_conv_2d(self, op, ins, outs) -> None:
        self._conv(op, ins, outs, depthwise=True)

    def op_max_pool_2d(self, op, ins, outs) -> None:
        o = tflite.Pool2DOptions()
        t = op.BuiltinOptions()
        o.Init(t.Bytes, t.Pos)
        k = [o.FilterHeight(), o.FilterWidth()]
        s = [o.StrideH(), o.StrideW()]
        in_shape = self.shape(ins[0])
        pads = self.same_pads(in_shape[1:3], k, s, [1, 1]) if o.Padding() == Padding.SAME else [0, 0, 0, 0]
        y = self.tmp("pool")
        self.emit("MaxPool", [self.nchw(self.name(ins[0]))], [y], kernel_shape=k, strides=s, pads=pads)
        a = self.tmp("act")
        self.activation(y, a, o.FusedActivationFunction())
        self.nhwc(a, self.name(outs[0]))

    def op_prelu(self, op, ins, outs) -> None:
        self.emit("PRelu", [self.name(ins[0]), self.name(ins[1])], [self.name(outs[0])])

    def _binary(self, op, ins, outs, onnx_op: str, opts_cls) -> None:
        act = Act.NONE
        t = op.BuiltinOptions()
        if t is not None and opts_cls is not None:
            o = opts_cls()
            o.Init(t.Bytes, t.Pos)
            act = o.FusedActivationFunction()
        y = self.tmp(onnx_op.lower())
        self.emit(onnx_op, [self.name(ins[0]), self.name(ins[1])], [y])
        self.activation(y, self.name(outs[0]), act)

    def op_add(self, op, ins, outs) -> None:
        self._binary(op, ins, outs, "Add", tflite.AddOptions)

    def op_sub(self, op, ins, outs) -> None:
        self._binary(op, ins, outs, "Sub", tflite.SubOptions)

    def op_mul(self, op, ins, outs) -> None:
        self._binary(op, ins, outs, "Mul", tflite.MulOptions)

    def op_div(self, op, ins, outs) -> None:
        self._binary(op, ins, outs, "Div", tflite.DivOptions)

    def op_squared_difference(self, op, ins, outs) -> None:
        d = self.tmp("diff")
        self.emit("Sub", [self.name(ins[0]), self.name(ins[1])], [d])
        self.emit("Mul", [d, d], [self.name(outs[0])])

    def op_pad(self, op, ins, outs) -> None:
        p = self.consts[ins[1]].astype(np.int64)
        pads = np.concatenate([p[:, 0], p[:, 1]])
        self.emit("Pad", [self.name(ins[0]), self.const(pads, "pads")], [self.name(outs[0])])

    def op_logistic(self, op, ins, outs) -> None:
        self.emit("Sigmoid", [self.name(ins[0])], [self.name(outs[0])])

    def op_sqrt(self, op, ins, outs) -> None:
        self.emit("Sqrt", [self.name(ins[0])], [self.name(outs[0])])

    def op_rsqrt(self, op, ins, outs) -> None:
        s = self.tmp("sqrt")
        self.emit("Sqrt", [self.name(ins[0])], [s])
        self.emit("Reciprocal", [s], [self.name(outs[0])])

    def op_neg(self, op, ins, outs) -> None:
        self.emit("Neg", [self.name(ins[0])], [self.name(outs[0])])

    def op_reshape(self, op, ins, outs) -> None:
        shape = np.array(self.shape(outs[0]), dtype=np.int64)
        self.emit("Reshape", [self.name(ins[0]), self.const(shape, "shape")], [self.name(outs[0])])

    def op_transpose(self, op, ins, outs) -> None:
        perm = [int(v) for v in self.consts[ins[1]]]
        self.emit("Transpose", [self.name(ins[0])], [self.name(outs[0])], perm=perm)

    def op_concatenation(self, op, ins, outs) -> None:
        o = tflite.ConcatenationOptions()
        t = op.BuiltinOptions()
        o.Init(t.Bytes, t.Pos)
        y = self.tmp("concat")
        self.emit("Concat", [self.name(i) for i in ins], [y], axis=o.Axis())
        self.activation(y, self.name(outs[0]), o.FusedActivationFunction())

    def _reduce(self, op, ins, outs, onnx_op: str) -> None:
        o = tflite.ReducerOptions()
        t = op.BuiltinOptions()
        o.Init(t.Bytes, t.Pos)
        axes = [int(v) for v in np.atleast_1d(self.consts[ins[1]])]
        keep = int(o.KeepDims())
        if onnx_op == "ReduceMean":  # axes is an attribute in opset 17
            self.emit(onnx_op, [self.name(ins[0])], [self.name(outs[0])], axes=axes, keepdims=keep)
        else:
            ax = self.const(np.array(axes, dtype=np.int64), "axes")
            self.emit(onnx_op, [self.name(ins[0]), ax], [self.name(outs[0])], keepdims=keep)

    def op_mean(self, op, ins, outs) -> None:
        self._reduce(op, ins, outs, "ReduceMean")

    def op_sum(self, op, ins, outs) -> None:
        self._reduce(op, ins, outs, "ReduceSum")

    def op_strided_slice(self, op, ins, outs) -> None:
        o = tflite.StridedSliceOptions()
        t = op.BuiltinOptions()
        o.Init(t.Bytes, t.Pos)
        if o.EllipsisMask() or o.NewAxisMask():
            raise NotImplementedError("strided slice with ellipsis/new-axis masks")
        begin = [int(v) for v in self.consts[ins[1]]]
        end = [int(v) for v in self.consts[ins[2]]]
        step = [int(v) for v in self.consts[ins[3]]]
        big = np.iinfo(np.int64).max
        starts, ends, shrink = [], [], []
        for axis in range(len(begin)):
            b, e, st = begin[axis], end[axis], step[axis]
            if o.BeginMask() >> axis & 1:
                b = 0 if st > 0 else big
            if o.EndMask() >> axis & 1:
                e = big if st > 0 else -big
            if o.ShrinkAxisMask() >> axis & 1:
                e = b + 1 if b != -1 else big
                shrink.append(axis)
            starts.append(b)
            ends.append(e)
        y = self.tmp("slice") if shrink else self.name(outs[0])
        self.emit(
            "Slice",
            [
                self.name(ins[0]),
                self.const(np.array(starts, np.int64), "starts"),
                self.const(np.array(ends, np.int64), "ends"),
                self.const(np.arange(len(begin), dtype=np.int64), "axes"),
                self.const(np.array(step, np.int64), "steps"),
            ],
            [y],
        )
        if shrink:
            self.emit("Squeeze", [y, self.const(np.array(shrink, np.int64), "sq")], [self.name(outs[0])])


def convert(tflite_bytes: bytes, inputs: list[str], outputs: list[str]) -> onnx.ModelProto:
    return Converter(tflite_bytes, inputs, outputs).convert()


def verify(tflite_bytes: bytes, onnx_model: onnx.ModelProto, feeds: list[np.ndarray]) -> float:
    """Return the max abs difference between TFLite and ONNX outputs."""
    import onnxruntime as ort
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    sess = ort.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
    worst = 0.0
    for feed in feeds:
        interp.set_tensor(interp.get_input_details()[0]["index"], feed)
        interp.invoke()
        ref = [interp.get_tensor(d["index"]) for d in interp.get_output_details()]
        got = sess.run(None, {sess.get_inputs()[0].name: feed})
        for r, g in zip(ref, got):
            worst = max(worst, float(np.max(np.abs(r.reshape(g.shape) - g))))
    return worst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", type=Path, help="face_landmarker.task bundle")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)

    with zipfile.ZipFile(args.task) as z:
        landmarks = z.read("face_landmarks_detector.tflite")
        blendshapes = z.read("face_blendshapes.tflite")

    jobs = [
        ("face_landmarks.onnx", landmarks, ["image"], ["landmarks", "presence", "aux"],
         lambda rng: rng.random((1, 256, 256, 3), dtype=np.float32)),
        ("face_blendshapes.onnx", blendshapes, ["points"], ["blendshapes"],
         lambda rng: (rng.random((1, 146, 2), dtype=np.float32) * 256).astype(np.float32)),
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for filename, data, ins, outs, make_feed in jobs:
        model = convert(data, ins, outs)
        if not args.no_verify:
            try:
                err = verify(data, model, [make_feed(rng) for _ in range(4)])
            except ImportError:
                print("ai-edge-litert not installed; skipping verification", file=sys.stderr)
            else:
                print(f"{filename}: max |tflite - onnx| = {err:.2e}")
                if err > 1e-3:
                    raise SystemExit(f"{filename}: conversion mismatch ({err})")
        path = args.out_dir / filename
        path.write_bytes(model.SerializeToString())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"wrote {path} ({path.stat().st_size} bytes, sha256 {digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
