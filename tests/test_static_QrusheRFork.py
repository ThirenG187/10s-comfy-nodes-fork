from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_independent_phase_ranges():
    patch = _load("qrf_patch_test", "ltx_multiface_reference_patch_QrusheRFork.py")
    cos = torch.ones(1, 1, 12, 8)
    sin = torch.zeros_like(cos)
    cos, sin = patch._compose_source_phase_range(cos, sin, 0, 4, 2.0, 1.0)
    cos, sin = patch._compose_source_phase_range(cos, sin, 4, 8, 3.0, 1.0)
    assert torch.allclose(cos[:, :, 8:, :], torch.ones_like(cos[:, :, 8:, :]))
    assert not torch.allclose(cos[:, :, :4, :], cos[:, :, 4:8, :])
    assert torch.count_nonzero(sin[:, :, :8, :]) > 0


def test_face_sorting():
    faces = _load("qrf_face_test", "face_detection_QrusheRFork.py")
    boxes = [(0.7, 0.1, 0.9, 0.3), (0.1, 0.1, 0.2, 0.2), (0.35, 0.1, 0.65, 0.5)]
    left = faces.sort_faces(boxes, "left_to_right")
    largest = faces.sort_faces(boxes, "largest_first")
    assert left[0] == boxes[1]
    assert largest[0] == boxes[2]


if __name__ == "__main__":
    test_independent_phase_ranges()
    test_face_sorting()
    print("QrusheRFork static tests passed")
