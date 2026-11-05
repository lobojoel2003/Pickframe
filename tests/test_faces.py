from __future__ import annotations

import cv2
import numpy as np
import pytest

from pickframe.analysis.faces import Face, FaceAnalyzer


def test_astronaut_face(face_analyzer: FaceAnalyzer, astronaut_rgb):
    faces = face_analyzer.analyze(astronaut_rgb)
    assert len(faces) == 1
    face = faces[0]
    assert face.eyes_closed_probability < 0.3
    assert face.smile > 0.5
    assert face.frontal > 0.6
    assert face.sharpness > 0.7
    assert face.presence > 0.9
    assert face.confidence > 0.8
    x, y, w, h = face.box
    assert 0.2 < x < 0.5
    assert 0.0 < y < 0.3
    assert 0.1 < w < 0.4
    assert face.area == pytest.approx(w * h)
    # eyes are inside the box, left eye (subject's) appears on the right
    (lx, ly), (rx, _ry) = face.eyes
    assert x < rx < lx < x + w
    assert y < ly < y + h
    assert face.pixels > 50
    assert all(0.15 < r < 0.5 for r in face.eye_ratio)


def test_no_faces_on_cat(face_analyzer: FaceAnalyzer, chelsea_rgb):
    assert face_analyzer.analyze(chelsea_rgb) == []
    assert face_analyzer.landmarks(chelsea_rgb) == []


def test_blank_image(face_analyzer: FaceAnalyzer):
    assert face_analyzer.analyze(np.zeros((100, 100, 3), np.uint8)) == []


def test_closed_eyes_detected(face_analyzer: FaceAnalyzer, astronaut_rgb, make_demo):
    closed = make_demo.close_eyes(astronaut_rgb)
    faces = face_analyzer.analyze(closed)
    assert len(faces) == 1
    assert faces[0].eyes_closed_probability >= 0.55


def test_blurred_face_loses_focus(face_analyzer: FaceAnalyzer, astronaut_rgb):
    sharp = face_analyzer.analyze(astronaut_rgb)[0]
    soft = face_analyzer.analyze(cv2.GaussianBlur(astronaut_rgb, (0, 0), 3))
    assert soft, "a moderately blurred face is still found"
    assert soft[0].focus_log < sharp.focus_log - 0.5
    assert soft[0].sharpness < sharp.sharpness


def test_large_image_is_downscaled_for_detection(astronaut_rgb):
    analyzer = FaceAnalyzer(detect_size=256)
    big = cv2.resize(astronaut_rgb, (1024, 1024))
    faces = analyzer.analyze(big)
    assert len(faces) == 1
    assert 0.2 < faces[0].box[0] < 0.5


def test_landmarks(face_analyzer: FaceAnalyzer, astronaut_rgb):
    points = face_analyzer.landmarks(astronaut_rgb)
    assert len(points) == 1
    assert points[0].shape == (478, 2)
    assert np.all(points[0] > 0)
    assert np.all(points[0] < 512)


def test_face_dict_roundtrip(face_analyzer: FaceAnalyzer, astronaut_rgb):
    face = face_analyzer.analyze(astronaut_rgb)[0]
    again = Face.from_dict(face.to_dict())
    assert again.box == pytest.approx(face.box, abs=1e-4)
    assert again.blink == pytest.approx(face.blink, abs=1e-3)
    assert again.eye_ratio == pytest.approx(face.eye_ratio, abs=1e-3)
    assert again.eyes_closed_probability == pytest.approx(face.eyes_closed_probability, abs=1e-2)
    assert again.focus_log == pytest.approx(face.focus_log, abs=1e-3)
    assert again.pixels == pytest.approx(face.pixels, abs=0.1)


def test_eyes_closed_probability_formula():
    def face(ratio: float, blink: float) -> Face:
        return Face(
            box=(0.1, 0.1, 0.2, 0.2), confidence=0.9, presence=1.0, blink=(blink, blink),
            eye_ratio=(ratio, ratio), smile=0.0, frontal=1.0, sharpness=1.0,
        )  # fmt: skip

    assert face(0.30, 0.05).eyes_closed_probability == 0.0
    assert face(0.05, 0.9).eyes_closed_probability == pytest.approx(1.0)
    middle = face(0.155, 0.375).eyes_closed_probability
    assert 0.4 < middle < 0.6


def test_eye_sharpness_of_tiny_region(face_analyzer: FaceAnalyzer):
    gray = np.zeros((50, 50), np.uint8)
    points = np.full((478, 2), 25.0)
    assert face_analyzer._eye_sharpness(gray, points) == (0.0, -6.0)
