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


def test_independent_legacy_phase_ranges():
    patch = _load("qrf_patch_test", "ltx_multiface_reference_patch_QrusheRFork.py")
    cosine = torch.ones(1, 1, 12, 8)
    sine = torch.zeros_like(cosine)
    pe = (cosine, sine, True)
    pe = patch._rotate_reference_block(pe, 4, 4, 2.0)
    pe = patch._rotate_reference_block(pe, 8, 4, 3.0)
    output_cosine, output_sine = pe[0], pe[1]
    assert torch.allclose(output_cosine[:, :, :4, :], torch.ones_like(output_cosine[:, :, :4, :]))
    assert not torch.allclose(output_cosine[:, :, 4:8, :], output_cosine[:, :, 8:12, :])
    assert torch.count_nonzero(output_sine[:, :, 4:, :]) > 0


def test_fused_matrix_phase_range():
    patch = _load("qrf_patch_matrix_test", "ltx_multiface_reference_patch_QrusheRFork.py")
    matrix = torch.eye(2).reshape(1, 1, 1, 1, 2, 2).expand(1, 10, 2, 4, 2, 2).clone()
    output = patch._rotate_reference_block((matrix, False), 3, 4, 2.0)[0]
    assert torch.allclose(output[:, :3], matrix[:, :3])
    assert torch.allclose(output[:, 7:], matrix[:, 7:])
    assert not torch.allclose(output[:, 3:7], matrix[:, 3:7])


def test_tass_layouts():
    patch = _load("qrf_patch_layout_test", "ltx_multiface_reference_patch_QrusheRFork.py")
    reference = torch.tensor([[[0.0, 1.0], [0.0, 2.0], [0.0, 2.0]]])
    target = torch.tensor([[[0.0, 8.0], [0.0, 16.0], [0.0, 16.0]]])
    assert torch.equal(patch._apply_tass_layout(reference, target, "overlap"), reference)
    shifted = patch._apply_tass_layout(reference, target, "st_drc")
    assert torch.all(shifted.amin(dim=2) >= target.amax(dim=2))
    strata = patch._apply_tass_layout(reference, target, "strata", strata_start=20.0)
    assert float(strata[:, 0, :].amin()) == 20.0
    assert torch.equal(strata[:, 1:, :], reference[:, 1:, :])


def test_face_sorting():
    faces = _load("qrf_face_test", "face_detection_QrusheRFork.py")
    boxes = [(0.7, 0.1, 0.9, 0.3), (0.1, 0.1, 0.2, 0.2), (0.35, 0.1, 0.65, 0.5)]
    left = faces.sort_faces(boxes, "left_to_right")
    largest = faces.sort_faces(boxes, "largest_first")
    assert left[0] == boxes[1]
    assert largest[0] == boxes[2]


if __name__ == "__main__":
    test_independent_legacy_phase_ranges()
    test_fused_matrix_phase_range()
    test_tass_layouts()
    test_face_sorting()
    print("QrusheRFork BFS hybrid static tests passed")
