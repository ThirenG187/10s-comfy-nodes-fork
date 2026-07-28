"""Multi-face detection helpers for the QrusheRFork identity reinforcer.

Derived from TenStrip's YuNet/MediaPipe/Haar fallback detector (MIT). Unlike
upstream's single-face helper, this module returns every usable face box.
"""
from __future__ import annotations

import os
import urllib.request
from typing import Iterable

import numpy as np

BBox = tuple[float, float, float, float]

_YUNET_DETECTOR = None
_YUNET_LOAD_TRIED = False
_YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def _log(message: str, debug: bool) -> None:
    if debug:
        print(f"[QrusheRFork FaceDetect] {message}")


def _pad_bbox(box: BBox, padding: float) -> BBox:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    half_w = (x2 - x1) * 0.5 * (1.0 + padding)
    half_h = (y2 - y1) * 0.5 * (1.0 + padding)
    return (
        max(0.0, cx - half_w),
        max(0.0, cy - half_h),
        min(1.0, cx + half_w),
        min(1.0, cy + half_h),
    )


def _area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(a: BBox, b: BBox) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _area(a) + _area(b) - intersection
    return intersection / union if union > 0 else 0.0


def _nms(boxes: Iterable[BBox], threshold: float = 0.45) -> list[BBox]:
    selected: list[BBox] = []
    for box in sorted(boxes, key=_area, reverse=True):
        if all(_iou(box, current) < threshold for current in selected):
            selected.append(box)
    return selected


def _download_yunet_model(target_path: str, debug: bool = False) -> bool:
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        _log(f"downloading YuNet model to {target_path}", debug)
        urllib.request.urlretrieve(_YUNET_MODEL_URL, target_path)
        return True
    except Exception as error:
        _log(f"YuNet download failed: {type(error).__name__}: {error}", debug)
        return False


def _get_yunet_detector(debug: bool = False):
    global _YUNET_DETECTOR, _YUNET_LOAD_TRIED
    if _YUNET_DETECTOR is not None:
        return _YUNET_DETECTOR
    if _YUNET_LOAD_TRIED:
        return None
    _YUNET_LOAD_TRIED = True
    try:
        import cv2

        candidates = [
            os.path.join(os.path.dirname(__file__), "models", "face_detection_yunet_2023mar.onnx"),
            os.path.expanduser("~/.cache/10s_comfy/face_detection_yunet_2023mar.onnx"),
        ]
        model_path = next((path for path in candidates if os.path.exists(path)), None)
        if model_path is None and _download_yunet_model(candidates[-1], debug):
            model_path = candidates[-1]
        if model_path is None:
            return None
        _YUNET_DETECTOR = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), 0.45, 0.3, 5000
        )
        _log(f"YuNet loaded from {model_path}", debug)
        return _YUNET_DETECTOR
    except Exception as error:
        _log(f"YuNet unavailable: {type(error).__name__}: {error}", debug)
        return None


def _detect_yunet(image_np: np.ndarray, debug: bool) -> list[BBox]:
    try:
        import cv2

        height, width = image_np.shape[:2]
        detector = _get_yunet_detector(debug)
        if detector is None:
            return []
        detector.setInputSize((width, height))
        _, faces = detector.detect(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        if faces is None:
            return []
        boxes = []
        for face in faces:
            x, y, w, h = map(float, face[:4])
            if w <= 1 or h <= 1:
                continue
            boxes.append((
                max(0.0, x / width),
                max(0.0, y / height),
                min(1.0, (x + w) / width),
                min(1.0, (y + h) / height),
            ))
        return boxes
    except Exception as error:
        _log(f"YuNet detection failed: {type(error).__name__}: {error}", debug)
        return []


def _detect_mediapipe(image_np: np.ndarray, debug: bool) -> list[BBox]:
    try:
        import mediapipe as mp

        boxes = []
        with mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.45
        ) as detector:
            results = detector.process(image_np)
            for detection in results.detections or []:
                value = detection.location_data.relative_bounding_box
                boxes.append((
                    max(0.0, float(value.xmin)),
                    max(0.0, float(value.ymin)),
                    min(1.0, float(value.xmin + value.width)),
                    min(1.0, float(value.ymin + value.height)),
                ))
        return boxes
    except Exception as error:
        _log(f"MediaPipe unavailable: {type(error).__name__}: {error}", debug)
        return []


def _detect_haar(image_np: np.ndarray, debug: bool) -> list[BBox]:
    try:
        import cv2

        height, width = image_np.shape[:2]
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=4, minSize=(32, 32)
        )
        return [
            (x / width, y / height, (x + w) / width, (y + h) / height)
            for x, y, w, h in faces
        ]
    except Exception as error:
        _log(f"Haar unavailable: {type(error).__name__}: {error}", debug)
        return []


def detect_all_faces(
    image_np: np.ndarray,
    *,
    padding: float = 0.15,
    debug: bool = False,
    max_faces: int = 8,
) -> list[BBox]:
    """Return normalized face boxes, largest-first, with backend fallbacks."""
    if image_np.ndim != 3 or image_np.shape[2] < 3:
        raise ValueError(f"Expected HxWx3 image, got {image_np.shape}")
    image_np = np.ascontiguousarray(image_np[:, :, :3].astype(np.uint8, copy=False))

    boxes = _detect_yunet(image_np, debug)
    backend = "YuNet"
    if not boxes:
        boxes = _detect_mediapipe(image_np, debug)
        backend = "MediaPipe"
    if not boxes:
        boxes = _detect_haar(image_np, debug)
        backend = "Haar"

    boxes = [_pad_bbox(box, padding) for box in boxes if _area(box) > 1e-5]
    boxes = _nms(boxes)[:max_faces]
    _log(f"{backend} returned {len(boxes)} face(s): {boxes}", debug)
    return boxes


def detect_largest_face(
    image_np: np.ndarray,
    *,
    padding: float = 0.15,
    debug: bool = False,
) -> BBox | None:
    boxes = detect_all_faces(image_np, padding=padding, debug=debug)
    return boxes[0] if boxes else None


def sort_faces(boxes: list[BBox], mode: str) -> list[BBox]:
    if mode == "largest_first":
        return sorted(boxes, key=_area, reverse=True)
    return sorted(boxes, key=lambda box: ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5))
