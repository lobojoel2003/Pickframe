"""Face detection, eye state, smile and face sharpness.

Pipeline for every photo:

1. **YuNet** finds faces and five keypoints (eyes, nose, mouth corners).
2. Each face is cropped, rotated so the eyes are level, and passed to the
   **MediaPipe face mesh** (478 landmarks + a "this really is a face" score,
   which also removes detector false positives).
3. The **MediaPipe blendshape model** turns the landmarks into expression
   scores. Pickframe uses ``eyeBlinkLeft/Right`` (closed eyes) and
   ``mouthSmileLeft/Right`` (smiles).
4. Focus is measured around the eyes, where it matters most in portraits.

All three models run on ONNX Runtime and ship with Pickframe.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from .. import models
from .quality import laplacian_energy

#: Landmark indices fed to the blendshape model (from MediaPipe's
#: ``face_blendshapes_graph.cc``).
BLENDSHAPE_INPUT = np.array(
    [
        0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58,
        61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107,
        109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157,
        158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197,
        234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293,
        295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334,
        336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384,
        385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466, 468,
        469, 470, 471, 472, 473, 474, 475, 476, 477,
    ],
    dtype=np.int64,
)  # fmt: skip

BLINK_LEFT, BLINK_RIGHT = 9, 10
SMILE_LEFT, SMILE_RIGHT = 44, 45

# Eye outlines in the 478-point mesh (subject's right eye, then left eye).
_RIGHT_EYE = np.array([33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246])
_LEFT_EYE = np.array([362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398])
_NOSE_TIP, _CHEEK_RIGHT, _CHEEK_LEFT = 1, 234, 454

_MESH_SIZE = 256
_CROP_SCALE = 1.5  # same expansion MediaPipe applies to detector boxes
_EYE_BAND_WIDTH = 160  # eye-region width used for the focus measurement

# Eye-band Laplacian energy range mapped to 0..1. Calibrated on sharp faces
# and the same faces with Gaussian blur of sigma 1, 2, 3 and 5 pixels.
_EYE_LOG_LOW = -2.6
_EYE_LOG_HIGH = -1.3

# Eye aspect ratio (lid gap / eye width): open eyes measure ~0.22-0.36,
# closed eyes ~0.04-0.17 on the MediaPipe mesh.
_EAR_OPEN = 0.21
_EAR_CLOSED = 0.10
_EAR_PAIRS = (
    (33, 133, (160, 159, 158), (144, 145, 153)),
    (362, 263, (385, 386, 387), (380, 374, 373)),
)


@dataclass
class Face:
    """One detected face. Coordinates are fractions of the image width/height."""

    box: tuple[float, float, float, float]  # x, y, w, h
    confidence: float  # detector score
    presence: float  # face-mesh confirmation score
    blink: tuple[float, float]  # blendshape eyeBlinkLeft/Right (0 open .. 1 closed)
    eye_ratio: tuple[float, float]  # eye aspect ratio of both eyes
    smile: float  # 0 .. 1
    frontal: float  # 1 = looking at the camera, 0 = profile
    sharpness: float  # focus around the eyes, 0 .. 1
    focus_log: float = 0.0  # log10 Laplacian energy around the eyes (unsaturated, for comparisons)
    eyes: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 0.0), (0.0, 0.0))
    pixels: float = 0.0  # face width in working-image pixels (small faces are less reliable)
    area: float = field(init=False)

    def __post_init__(self) -> None:
        self.area = self.box[2] * self.box[3]

    @property
    def eyes_closed_probability(self) -> float:
        """0 = clearly open .. 1 = clearly closed.

        Combines eyelid geometry from the mesh (eye aspect ratio) with the
        blendshape model's blink scores; each alone is noisier.
        """
        ratio = (self.eye_ratio[0] + self.eye_ratio[1]) / 2.0
        from_ratio = (_EAR_OPEN - ratio) / (_EAR_OPEN - _EAR_CLOSED)
        blink = (self.blink[0] + self.blink[1]) / 2.0
        from_blink = (blink - 0.2) / 0.35
        value = 0.6 * min(1.0, max(0.0, from_ratio)) + 0.4 * min(1.0, max(0.0, from_blink))
        return float(value)

    def to_dict(self) -> dict:
        return {
            "box": [round(v, 5) for v in self.box],
            "confidence": round(self.confidence, 4),
            "presence": round(self.presence, 4),
            "blink": [round(v, 4) for v in self.blink],
            "eye_ratio": [round(v, 4) for v in self.eye_ratio],
            "smile": round(self.smile, 4),
            "frontal": round(self.frontal, 4),
            "sharpness": round(self.sharpness, 4),
            "focus_log": round(self.focus_log, 4),
            "eyes": [[round(v, 5) for v in eye] for eye in self.eyes],
            "pixels": round(self.pixels, 1),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Face:
        eyes = data.get("eyes") or [[0, 0], [0, 0]]
        return cls(
            box=tuple(data["box"]),  # type: ignore[arg-type]
            confidence=float(data["confidence"]),
            presence=float(data["presence"]),
            blink=(float(data["blink"][0]), float(data["blink"][1])),
            eye_ratio=(float(data["eye_ratio"][0]), float(data["eye_ratio"][1])),
            smile=float(data["smile"]),
            frontal=float(data["frontal"]),
            sharpness=float(data["sharpness"]),
            focus_log=float(data.get("focus_log", 0.0)),
            eyes=((float(eyes[0][0]), float(eyes[0][1])), (float(eyes[1][0]), float(eyes[1][1]))),
            pixels=float(data.get("pixels", 0.0)),
        )


@dataclass
class _Detection:
    box: np.ndarray  # x, y, w, h in working-image pixels
    score: float
    keypoints: np.ndarray  # 5 x 2: right eye, left eye, nose, mouth right, mouth left


class FaceAnalyzer:
    """Thread-safe face analysis built on three bundled ONNX models."""

    def __init__(
        self,
        detect_size: int = 1280,
        min_confidence: float = 0.6,
        min_presence: float = 0.5,
        max_faces: int = 12,
        threads: int = 0,
    ):
        self.detect_size = detect_size
        self.min_confidence = min_confidence
        self.min_presence = min_presence
        self.max_faces = max_faces
        self._detector = models.session(models.bundled(models.FACE_DETECTOR), threads)
        self._mesh = models.session(models.bundled(models.FACE_LANDMARKS), threads)
        self._blend = models.session(models.bundled(models.FACE_BLENDSHAPES), threads)
        self._detector_outputs = [o.name for o in self._detector.get_outputs()]
        self._lock = threading.Lock()

    # -- detection ---------------------------------------------------------
    def detect(self, rgb: np.ndarray) -> list[_Detection]:
        h, w = rgb.shape[:2]
        scale = min(1.0, self.detect_size / max(h, w))
        image = rgb
        if scale < 1.0:
            image = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        ih, iw = image.shape[:2]
        ph, pw = -(-ih // 32) * 32, -(-iw // 32) * 32
        blob = np.zeros((1, 3, ph, pw), np.float32)
        blob[0, :, :ih, :iw] = image[..., ::-1].transpose(2, 0, 1)  # YuNet expects BGR 0..255
        out = dict(zip(self._detector_outputs, self._detector.run(None, {"input": blob})))

        boxes, scores, keypoints = [], [], []
        for stride in (8, 16, 32):
            cols = pw // stride
            cls = out[f"cls_{stride}"].reshape(-1)
            obj = out[f"obj_{stride}"].reshape(-1)
            score = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
            idx = np.nonzero(score >= self.min_confidence)[0]
            if idx.size == 0:
                continue
            bbox = out[f"bbox_{stride}"].reshape(-1, 4)[idx]
            kps = out[f"kps_{stride}"].reshape(-1, 5, 2)[idx]
            row, col = (idx // cols).astype(np.float32), (idx % cols).astype(np.float32)
            cx = (col + bbox[:, 0]) * stride
            cy = (row + bbox[:, 1]) * stride
            bw = np.exp(bbox[:, 2]) * stride
            bh = np.exp(bbox[:, 3]) * stride
            boxes.append(np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=1))
            scores.append(score[idx])
            keypoints.append(
                np.stack([(kps[..., 0] + col[:, None]) * stride, (kps[..., 1] + row[:, None]) * stride], -1)
            )
        if not boxes:
            return []
        all_boxes = np.concatenate(boxes)
        all_scores = np.concatenate(scores)
        all_kps = np.concatenate(keypoints)
        keep = cv2.dnn.NMSBoxes(all_boxes.tolist(), all_scores.tolist(), self.min_confidence, 0.3)
        keep = np.asarray(keep, dtype=np.int64).reshape(-1)
        detections = [
            _Detection(all_boxes[i] / scale, float(all_scores[i]), all_kps[i] / scale) for i in keep
        ]
        detections.sort(key=lambda d: -d.box[2] * d.box[3])
        return detections

    # -- per-face analysis ---------------------------------------------------
    def _align(self, det: _Detection) -> np.ndarray:
        right_eye, left_eye = det.keypoints[0], det.keypoints[1]
        angle = math.degrees(math.atan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0]))
        cx = det.box[0] + det.box[2] / 2
        cy = det.box[1] + det.box[3] / 2
        size = max(det.box[2], det.box[3]) * _CROP_SCALE
        matrix = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, _MESH_SIZE / size)
        matrix[0, 2] += _MESH_SIZE / 2 - cx
        matrix[1, 2] += _MESH_SIZE / 2 - cy
        return matrix

    def _eye_sharpness(self, gray: np.ndarray, points: np.ndarray) -> tuple[float, float]:
        """Focus score (0..1) and log10 Laplacian energy of the eye region."""
        eyes = points[np.concatenate([_RIGHT_EYE, _LEFT_EYE])]
        x0, y0 = eyes.min(axis=0)
        x1, y1 = eyes.max(axis=0)
        pad_x = (x1 - x0) * 0.15
        pad_y = max((y1 - y0) * 0.6, (x1 - x0) * 0.12)
        h, w = gray.shape
        xa, xb = int(max(0, x0 - pad_x)), int(min(w, x1 + pad_x + 1))
        ya, yb = int(max(0, y0 - pad_y)), int(min(h, y1 + pad_y + 1))
        band = gray[ya:yb, xa:xb]
        if band.shape[0] < 6 or band.shape[1] < 12:
            return 0.0, -6.0
        if band.shape[1] > _EYE_BAND_WIDTH:
            factor = _EYE_BAND_WIDTH / band.shape[1]
            band = cv2.resize(
                band, (_EYE_BAND_WIDTH, max(6, round(band.shape[0] * factor))), interpolation=cv2.INTER_AREA
            )
        energy = laplacian_energy(band, tile=16, top=0.25)
        if energy <= 0:
            return 0.0, -6.0
        log_energy = math.log10(energy)
        value = (log_energy - _EYE_LOG_LOW) / (_EYE_LOG_HIGH - _EYE_LOG_LOW)
        return float(min(1.0, max(0.0, value))), float(log_energy)

    def landmarks(self, rgb: np.ndarray) -> list[np.ndarray]:
        """478 x 2 face-mesh landmarks (working-image pixels) for every confirmed face."""
        points = []
        for det in self.detect(rgb)[: self.max_faces]:
            matrix = self._align(det)
            crop = cv2.warpAffine(
                rgb, matrix, (_MESH_SIZE, _MESH_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            raw, presence_logit, _ = self._mesh.run(None, {"image": (crop.astype(np.float32) / 255.0)[None]})
            if 1.0 / (1.0 + math.exp(-float(np.ravel(presence_logit)[0]))) < self.min_presence:
                continue
            inverse = cv2.invertAffineTransform(matrix)
            points.append(raw.reshape(478, 3)[:, :2] @ inverse[:, :2].T + inverse[:, 2])
        return points

    def analyze(self, rgb: np.ndarray) -> list[Face]:
        h, w = rgb.shape[:2]
        detections = self.detect(rgb)[: self.max_faces]
        if not detections:
            return []
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces: list[Face] = []
        for det in detections:
            if min(det.box[2], det.box[3]) < 12:
                continue
            matrix = self._align(det)
            crop = cv2.warpAffine(
                rgb, matrix, (_MESH_SIZE, _MESH_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            tensor = (crop.astype(np.float32) * (1.0 / 255.0))[None]
            raw_points, presence_logit, _ = self._mesh.run(None, {"image": tensor})
            presence = 1.0 / (1.0 + math.exp(-float(np.ravel(presence_logit)[0])))
            if presence < self.min_presence:
                continue
            crop_points = raw_points.reshape(478, 3)[:, :2]
            inverse = cv2.invertAffineTransform(matrix)
            points = crop_points @ inverse[:, :2].T + inverse[:, 2]
            scores = self._blend.run(
                None, {"points": points[BLENDSHAPE_INPUT][None].astype(np.float32)}
            )[0].reshape(-1)

            # Yaw from the nose position between the cheeks, in the level crop.
            left, right = crop_points[_CHEEK_RIGHT, 0], crop_points[_CHEEK_LEFT, 0]
            ratio = (crop_points[_NOSE_TIP, 0] - left) / max(1e-3, right - left)
            frontal = float(min(1.0, max(0.0, 1.0 - abs(ratio - 0.5) / 0.4)))

            ratios = []
            for corner_a, corner_b, upper, lower in _EAR_PAIRS:
                width = float(np.linalg.norm(crop_points[corner_a] - crop_points[corner_b]))
                gap = np.mean(
                    [np.linalg.norm(crop_points[u] - crop_points[lo]) for u, lo in zip(upper, lower)]
                )
                ratios.append(float(gap / max(width, 1e-3)))

            focus, focus_log = self._eye_sharpness(gray, points)
            right_eye = points[_RIGHT_EYE].mean(axis=0)
            left_eye = points[_LEFT_EYE].mean(axis=0)
            x, y, bw, bh = (float(v) for v in det.box)
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(w), x + bw), min(float(h), y + bh)
            faces.append(
                Face(
                    box=(x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h),
                    confidence=det.score,
                    presence=presence,
                    blink=(float(scores[BLINK_LEFT]), float(scores[BLINK_RIGHT])),
                    eye_ratio=(ratios[1], ratios[0]),
                    smile=float((scores[SMILE_LEFT] + scores[SMILE_RIGHT]) / 2),
                    frontal=frontal,
                    sharpness=focus,
                    focus_log=focus_log,
                    eyes=(
                        (float(left_eye[0] / w), float(left_eye[1] / h)),
                        (float(right_eye[0] / w), float(right_eye[1] / h)),
                    ),
                    pixels=float(bw),
                )
            )
        return faces

