"""Experimental multi-face LTX Best-Face-ID reinforcer for ComfyUI.

This is a namespaced fork of TenStrip's LTXFaceIdentityReinforcer. It retains
reference_image_2 as an alternate view of Subject 1 and adds independent
reference pairs for Subjects 2-4. Each subject is encoded separately, assigned
to a detected target face, spatially gated, and given an independent RoPE
source phase range.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .face_detection_QrusheRFork import BBox, detect_all_faces, detect_largest_face, sort_faces
from .ltx_multiface_reference_patch_QrusheRFork import install_on_model


@dataclass
class _SubjectInput:
    index: int
    primary: torch.Tensor
    secondary: torch.Tensor | None
    strength: float
    face_index: int


def _debug(message: str, enabled: bool) -> None:
    if enabled:
        print(f"[QrusheRFork Reinforcer] {message}")


def _image_to_numpy(image: torch.Tensor):
    return (image[0].detach().cpu().clamp(0.0, 1.0) * 255.0).to(torch.uint8).numpy()


def _latent_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, dict):
        value = value.get("samples")
    if not isinstance(value, torch.Tensor):
        raise TypeError("target_latent must be a LATENT tensor or {'samples': tensor}")
    return value


def _target_dimensions(target_latent: Any, vae_scale: int = 32) -> tuple[int, int, tuple[int, int, int, int, int]]:
    latent = _latent_tensor(target_latent)
    if latent.dim() == 5:
        batch, channels, frames, height, width = latent.shape
    elif latent.dim() == 4:
        batch, channels, height, width = latent.shape
        frames = 1
    else:
        raise ValueError(f"Expected target latent with 4 or 5 dimensions, got {tuple(latent.shape)}")
    return height * vae_scale, width * vae_scale, (batch, channels, frames, height, width)


def _resize_bhwc(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if image.dim() != 4:
        raise ValueError(f"Expected IMAGE [B,H,W,C], got {tuple(image.shape)}")
    if image.shape[1:3] == (height, width):
        return image.clamp(0.0, 1.0)
    value = image.permute(0, 3, 1, 2).contiguous()
    value = F.interpolate(value, size=(height, width), mode="bicubic", align_corners=False)
    return value.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0)


def _auto_face_crop(
    image: torch.Tensor,
    bbox: BBox,
    target_height: int,
    target_width: int,
    zoom_factor: float,
) -> tuple[torch.Tensor, BBox]:
    """Crop around one face while preserving the target aspect ratio."""
    _, source_height, source_width, _ = image.shape
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) * 0.5 * source_width
    center_y = (y1 + y2) * 0.5 * source_height
    face_width = max(2.0, (x2 - x1) * source_width)
    face_height = max(2.0, (y2 - y1) * source_height)

    crop_width = face_width * zoom_factor
    crop_height = face_height * zoom_factor
    target_aspect = target_width / max(target_height, 1)
    if crop_width / max(crop_height, 1e-6) < target_aspect:
        crop_width = crop_height * target_aspect
    else:
        crop_height = crop_width / target_aspect

    crop_width = min(float(source_width), crop_width)
    crop_height = min(float(source_height), crop_height)
    left = min(max(0.0, center_x - crop_width * 0.5), source_width - crop_width)
    top = min(max(0.0, center_y - crop_height * 0.5), source_height - crop_height)
    right = left + crop_width
    bottom = top + crop_height

    ix1, iy1 = int(math.floor(left)), int(math.floor(top))
    ix2, iy2 = int(math.ceil(right)), int(math.ceil(bottom))
    cropped = image[:, iy1:iy2, ix1:ix2, :]
    cropped = _resize_bhwc(cropped, target_height, target_width)

    transformed = (
        max(0.0, ((x1 * source_width) - left) / max(crop_width, 1e-6)),
        max(0.0, ((y1 * source_height) - top) / max(crop_height, 1e-6)),
        min(1.0, ((x2 * source_width) - left) / max(crop_width, 1e-6)),
        min(1.0, ((y2 * source_height) - top) / max(crop_height, 1e-6)),
    )
    return cropped, transformed


def _align_face_to_target(
    image: torch.Tensor,
    source_bbox: BBox,
    target_bbox: BBox,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    """Uniformly scale and place a source face at a target face box."""
    batch, source_height, source_width, channels = image.shape
    sx1, sy1, sx2, sy2 = source_bbox
    tx1, ty1, tx2, ty2 = target_bbox
    source_face_w = max(1.0, (sx2 - sx1) * source_width)
    source_face_h = max(1.0, (sy2 - sy1) * source_height)
    target_face_w = max(1.0, (tx2 - tx1) * target_width)
    target_face_h = max(1.0, (ty2 - ty1) * target_height)
    scale = math.sqrt((target_face_w / source_face_w) * (target_face_h / source_face_h))

    new_width = max(1, int(round(source_width * scale)))
    new_height = max(1, int(round(source_height * scale)))
    scaled = _resize_bhwc(image, new_height, new_width)

    source_center_x = (sx1 + sx2) * 0.5 * new_width
    source_center_y = (sy1 + sy2) * 0.5 * new_height
    target_center_x = (tx1 + tx2) * 0.5 * target_width
    target_center_y = (ty1 + ty2) * 0.5 * target_height
    origin_x = int(round(target_center_x - source_center_x))
    origin_y = int(round(target_center_y - source_center_y))

    # Start from a resized full image so uncovered regions remain meaningful,
    # then paste the correctly aligned source over it.
    canvas = _resize_bhwc(image, target_height, target_width).clone()
    src_x1, src_y1 = max(0, -origin_x), max(0, -origin_y)
    dst_x1, dst_y1 = max(0, origin_x), max(0, origin_y)
    copy_width = min(new_width - src_x1, target_width - dst_x1)
    copy_height = min(new_height - src_y1, target_height - dst_y1)
    if copy_width > 0 and copy_height > 0:
        canvas[:, dst_y1:dst_y1 + copy_height, dst_x1:dst_x1 + copy_width, :] = scaled[
            :, src_y1:src_y1 + copy_height, src_x1:src_x1 + copy_width, :
        ]
    return canvas.clamp(0.0, 1.0)


def _make_face_mask(
    bbox: BBox | None,
    latent_shape: tuple[int, int, int, int, int],
    mode: str,
    dilation: float,
) -> torch.Tensor | None:
    if bbox is None or mode == "off":
        return None
    _, _, _, height, width = latent_shape
    x1, y1, x2, y2 = bbox
    center_x, center_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    half_w = max(1e-5, (x2 - x1) * (0.5 + dilation))
    half_h = max(1e-5, (y2 - y1) * (0.5 + dilation))

    yy = (torch.arange(height, dtype=torch.float32) + 0.5).view(-1, 1) / height
    xx = (torch.arange(width, dtype=torch.float32) + 0.5).view(1, -1) / width
    dx = (xx - center_x).abs() / half_w
    dy = (yy - center_y).abs() / half_h
    distance = torch.maximum(dx, dy)
    if mode == "mask_hard":
        mask = (distance <= 1.0).float()
    else:
        mask = torch.where(
            distance <= 1.0,
            torch.ones_like(distance),
            torch.where(
                distance >= 1.65,
                torch.zeros_like(distance),
                0.5 * (1.0 + torch.cos((distance - 1.0) * math.pi / 0.65)),
            ),
        )
    return mask.view(1, 1, 1, height, width)


def _extract_vae_latent(encoded: Any) -> torch.Tensor:
    if isinstance(encoded, dict):
        for key in ("samples", "latent", "latents"):
            if isinstance(encoded.get(key), torch.Tensor):
                encoded = encoded[key]
                break
    if not isinstance(encoded, torch.Tensor):
        raise TypeError(f"VAE.encode returned unsupported type {type(encoded).__name__}")
    if encoded.dim() == 4:
        encoded = encoded.unsqueeze(2)
    if encoded.dim() != 5:
        raise ValueError(f"Expected encoded latent [B,C,F,H,W], got {tuple(encoded.shape)}")
    return encoded


def _encode_reference(vae, image: torch.Tensor, strength: float) -> torch.Tensor:
    encoded = _extract_vae_latent(vae.encode(image))
    return encoded * float(strength)


def _resolve_assignments(
    boxes: list[BBox],
    subjects: list[_SubjectInput],
    assignment_mode: str,
) -> dict[int, BBox]:
    ordered = sort_faces(boxes, "largest_first" if assignment_mode == "largest_first" else "left_to_right")
    assignments: dict[int, BBox] = {}
    used: set[int] = set()
    for position, subject in enumerate(subjects):
        index = subject.face_index if assignment_mode == "manual" else position
        if index < 0 or index >= len(ordered):
            raise ValueError(
                f"Subject {subject.index} requests target face index {index}, but only {len(ordered)} face(s) were detected."
            )
        if index in used:
            raise ValueError(f"Target face index {index} is assigned to more than one subject.")
        used.add(index)
        assignments[subject.index] = ordered[index]
    return assignments


class LTXFaceIdentityReinforcer_QrusheRFork:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "reference_image": ("IMAGE", {
                    "tooltip": "Subject 1 primary identity reference."
                }),
                "target_latent": ("LATENT",),
            },
            "optional": {
                "target_image": ("IMAGE", {
                    "tooltip": "The i2v first frame/composition image containing all target faces. Required for 2+ subjects."
                }),
                "reference_image_2": ("IMAGE", {
                    "tooltip": "Optional alternate/cropped view of Subject 1 (same person)."
                }),
                "subject_2_reference_image": ("IMAGE",),
                "subject_2_reference_image_2": ("IMAGE", {
                    "tooltip": "Optional alternate view of Subject 2."
                }),
                "subject_3_reference_image": ("IMAGE",),
                "subject_3_reference_image_2": ("IMAGE",),
                "subject_4_reference_image": ("IMAGE",),
                "subject_4_reference_image_2": ("IMAGE",),
                "identity_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "subject_2_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "subject_3_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "subject_4_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "assignment_mode": (["left_to_right", "largest_first", "manual"], {"default": "left_to_right"}),
                "subject_1_face_index": ("INT", {"default": 0, "min": 0, "max": 15}),
                "subject_2_face_index": ("INT", {"default": 1, "min": 0, "max": 15}),
                "subject_3_face_index": ("INT", {"default": 2, "min": 0, "max": 15}),
                "subject_4_face_index": ("INT", {"default": 3, "min": 0, "max": 15}),
                "face_padding": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.05}),
                "auto_face_crop": ("BOOLEAN", {"default": True}),
                "crop_zoom_factor": ("FLOAT", {"default": 2.0, "min": 1.2, "max": 4.0, "step": 0.1}),
                "spatial_gating": (["mask_soft", "mask_hard", "off"], {"default": "mask_soft"}),
                "placement_mode": (["i2v_safe", "t2v_overlap", "prefix"], {"default": "i2v_safe"}),
                "source_id_base": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 16.0, "step": 1.0}),
                "source_id_stride": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1}),
                "phase_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "background_reference_strength": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.25, "step": 0.01}),
                "zero_reference_timesteps": ("BOOLEAN", {"default": False}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "reinforce"
    CATEGORY = "10S Nodes_QrusheRFork/Identity"
    DESCRIPTION = (
        "Experimental multi-face fork of TenStrip's LTX Best-Face-ID reinforcer. "
        "Assigns up to four independent identity references to faces detected in "
        "the supplied target/first-frame image, with one RoPE source range and "
        "spatial mask per subject."
    )

    def reinforce(
        self,
        model,
        vae,
        reference_image,
        target_latent,
        target_image=None,
        reference_image_2=None,
        subject_2_reference_image=None,
        subject_2_reference_image_2=None,
        subject_3_reference_image=None,
        subject_3_reference_image_2=None,
        subject_4_reference_image=None,
        subject_4_reference_image_2=None,
        identity_strength: float = 1.0,
        subject_2_strength: float = 1.0,
        subject_3_strength: float = 1.0,
        subject_4_strength: float = 1.0,
        assignment_mode: str = "left_to_right",
        subject_1_face_index: int = 0,
        subject_2_face_index: int = 1,
        subject_3_face_index: int = 2,
        subject_4_face_index: int = 3,
        face_padding: float = 0.15,
        auto_face_crop: bool = True,
        crop_zoom_factor: float = 2.0,
        spatial_gating: str = "mask_soft",
        placement_mode: str = "i2v_safe",
        source_id_base: float = 2.0,
        source_id_stride: float = 1.0,
        phase_scale: float = 1.0,
        background_reference_strength: float = 0.02,
        zero_reference_timesteps: bool = False,
        debug: bool = False,
    ):
        subjects = [
            _SubjectInput(1, reference_image, reference_image_2, identity_strength, subject_1_face_index)
        ]
        optional_subjects = [
            (2, subject_2_reference_image, subject_2_reference_image_2, subject_2_strength, subject_2_face_index),
            (3, subject_3_reference_image, subject_3_reference_image_2, subject_3_strength, subject_3_face_index),
            (4, subject_4_reference_image, subject_4_reference_image_2, subject_4_strength, subject_4_face_index),
        ]
        subjects.extend(
            _SubjectInput(index, primary, secondary, strength, face_index)
            for index, primary, secondary, strength, face_index in optional_subjects
            if primary is not None
        )

        target_height, target_width, latent_shape = _target_dimensions(target_latent)
        if len(subjects) > 1 and target_image is None:
            raise ValueError(
                "target_image is required when using more than one subject. Wire the same first-frame image used by your i2v conditioning."
            )

        target_faces: list[BBox] = []
        assignments: dict[int, BBox] = {}
        if target_image is not None:
            target_faces = detect_all_faces(
                _image_to_numpy(target_image), padding=face_padding, debug=debug, max_faces=16
            )
            if len(target_faces) < len(subjects):
                raise ValueError(
                    f"Detected {len(target_faces)} target face(s), but {len(subjects)} subject reference(s) are connected."
                )
            assignments = _resolve_assignments(target_faces, subjects, assignment_mode)
            _debug(f"target assignments: {assignments}", debug)

        reference_subjects: list[dict[str, Any]] = []
        for subject in subjects:
            primary_bbox = detect_largest_face(
                _image_to_numpy(subject.primary), padding=face_padding, debug=debug
            )
            if primary_bbox is None and auto_face_crop:
                raise ValueError(f"No face was detected in Subject {subject.index}'s primary reference.")
            target_bbox = assignments.get(subject.index)

            if target_bbox is not None and primary_bbox is not None:
                prepared_primary = _align_face_to_target(
                    subject.primary, primary_bbox, target_bbox, target_height, target_width
                )
            elif auto_face_crop and primary_bbox is not None:
                prepared_primary, target_bbox = _auto_face_crop(
                    subject.primary, primary_bbox, target_height, target_width, crop_zoom_factor
                )
            else:
                prepared_primary = _resize_bhwc(subject.primary, target_height, target_width)

            latent_parts = [_encode_reference(vae, prepared_primary, subject.strength)]
            if subject.secondary is not None:
                secondary_bbox = detect_largest_face(
                    _image_to_numpy(subject.secondary), padding=face_padding, debug=debug
                )
                if target_bbox is not None and secondary_bbox is not None:
                    prepared_secondary = _align_face_to_target(
                        subject.secondary, secondary_bbox, target_bbox, target_height, target_width
                    )
                elif auto_face_crop and secondary_bbox is not None:
                    prepared_secondary, _ = _auto_face_crop(
                        subject.secondary, secondary_bbox, target_height, target_width, crop_zoom_factor
                    )
                else:
                    prepared_secondary = _resize_bhwc(subject.secondary, target_height, target_width)
                latent_parts.append(_encode_reference(vae, prepared_secondary, subject.strength))

            subject_latent = torch.cat(latent_parts, dim=2) if len(latent_parts) > 1 else latent_parts[0]
            mask = _make_face_mask(target_bbox, latent_shape, spatial_gating, face_padding)
            source_id = float(source_id_base + (subject.index - 1) * source_id_stride)
            reference_subjects.append({
                "subject_index": subject.index,
                "latent": subject_latent,
                "source_id": source_id,
                "phase_scale": float(phase_scale),
                "spatial_mask": mask,
                "background_floor": float(background_reference_strength),
            })
            _debug(
                f"subject {subject.index}: latent={tuple(subject_latent.shape)}, "
                f"source_id={source_id}, target_bbox={target_bbox}, strength={subject.strength}",
                debug,
            )

        cloned = model.clone()
        install_on_model(
            cloned,
            zero_reference_timesteps=zero_reference_timesteps,
            verbose=debug,
        )
        transformer_options = cloned.model_options.setdefault("transformer_options", {})
        transformer_options["qrf_reference_subjects"] = reference_subjects
        transformer_options["qrf_reference_position_mode"] = (
            "prefix" if placement_mode == "prefix" else "overlap"
        )
        return (cloned,)


NODE_CLASS_MAPPINGS = {
    "LTXFaceIdentityReinforcer_QrusheRFork": LTXFaceIdentityReinforcer_QrusheRFork,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXFaceIdentityReinforcer_QrusheRFork": "🧑‍🤝‍🧑 LTX Multi-Face Identity Reinforcer _QrusheRFork",
}
