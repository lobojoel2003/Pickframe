# Third-party models bundled with Pickframe

Pickframe ships three small ONNX models in this folder. They were produced
by `tools/build_models.py` from the upstream files listed below. Their
licenses are compatible with Pickframe's GPL-3.0-or-later license.

| File | Upstream | License |
| --- | --- | --- |
| `face_detection_yunet.onnx` | [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) `face_detection_yunet_2023mar.onnx` (input dimensions made dynamic) | MIT, Copyright (c) 2020 Shiqi Yu |
| `face_landmarks.onnx` | [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) `face_landmarks_detector.tflite`, converted to ONNX | Apache-2.0, Copyright Google LLC |
| `face_blendshapes.onnx` | [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) `face_blendshapes.tflite`, converted to ONNX | Apache-2.0, Copyright Google LLC |

SHA-256 of the upstream downloads:

- `face_detection_yunet_2023mar.onnx`: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
- `face_landmarker.task` (float16, version 1): `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`

The converted models were checked against the original TFLite models with
the TFLite interpreter (`ai-edge-litert`): the largest output difference is
6.4e-4 pixels for the landmarks and 1.4e-6 for the blendshape scores.

## Optional download

`pickframe models download clip` fetches the CLIP ViT-B/32 image encoder
(OpenAI, MIT license; 8-bit ONNX export by Xenova on Hugging Face) into your
user data folder. It is not bundled.

## MIT License (YuNet)

```
MIT License

Copyright (c) 2020 Shiqi Yu <shiqi.yu@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Apache License 2.0 (MediaPipe models)

The MediaPipe models are licensed under the Apache License, Version 2.0.
You may obtain a copy of the license at
<https://www.apache.org/licenses/LICENSE-2.0>. Unless required by applicable
law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied. The models were modified by converting them
from TFLite to ONNX format; the weights are unchanged.
