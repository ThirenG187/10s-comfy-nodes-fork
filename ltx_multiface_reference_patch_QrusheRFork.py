"""Namespaced multi-subject reference-token patch for LTX2/LTX-AV.

This module is derived from TenStrip's ltx_reference_enable.py mechanism but
uses qrf_* keys/attributes so the fork can coexist with the upstream 10S pack.
It prepends one independently tagged reference-token range per subject.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

_PATCHES_APPLIED = False
_PATCH_ERROR: Optional[str] = None
_ORIGINAL_PROCESS_INPUT = None
_ORIGINAL_PREPARE_TIMESTEP = None
_ORIGINAL_PREPARE_PE = None
_VERBOSE = False


def _log(message: str) -> None:
    if _VERBOSE:
        print(f"[QrusheRFork MultiFace] {message}")


def _import_comfy():
    import comfy.ldm.lightricks.av_model as av_module
    from comfy.ldm.lightricks.symmetric_patchifier import latent_to_pixel_coords
    return av_module, latent_to_pixel_coords


def _compose_source_phase_range(
    cos_orig: torch.Tensor,
    sin_orig: torch.Tensor,
    start: int,
    end: int,
    source_id: float,
    phase_scale: float,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Best-Face-ID phase composition to one sequence range."""
    if source_id == 0.0 or phase_scale == 0.0 or end <= start:
        return cos_orig, sin_orig
    if cos_orig.dim() != 4 or sin_orig.dim() != 4:
        return cos_orig, sin_orig

    _, _, sequence_length, head_dim = cos_orig.shape
    start = max(0, min(int(start), sequence_length))
    end = max(start, min(int(end), sequence_length))
    if end <= start:
        return cos_orig, sin_orig

    device = cos_orig.device
    dtype = cos_orig.dtype
    pair_count = max(1, head_dim // 2)
    pair_index = torch.arange(pair_count, device=device, dtype=torch.float32)
    rate_per_pair = theta ** (-2.0 * pair_index / max(head_dim, 1))
    rate = rate_per_pair.repeat_interleave(2)[:head_dim]
    if rate.numel() < head_dim:
        rate = torch.cat([rate, rate[-1:].expand(head_dim - rate.numel())])

    angle = source_id * phase_scale * rate
    cos_extra = angle.cos().to(dtype=dtype).view(1, 1, 1, head_dim)
    sin_extra = angle.sin().to(dtype=dtype).view(1, 1, 1, head_dim)

    cos_slice = cos_orig[:, :, start:end, :]
    sin_slice = sin_orig[:, :, start:end, :]
    cos_new = cos_slice * cos_extra - sin_slice * sin_extra
    sin_new = cos_slice * sin_extra + sin_slice * cos_extra

    cos_out = cos_orig.clone()
    sin_out = sin_orig.clone()
    cos_out[:, :, start:end, :] = cos_new
    sin_out[:, :, start:end, :] = sin_new
    return cos_out, sin_out


def _apply_reference_ranges_to_pe(pe: Any, ranges: list[dict[str, Any]], theta: float) -> Any:
    """Apply each subject's source phase to self-attention frequencies only."""
    if not ranges or not isinstance(pe, (list, tuple)) or not pe:
        return pe
    self_attention = pe[0]
    if not isinstance(self_attention, (list, tuple)):
        return pe

    updated = []
    for frequency_tuple in self_attention:
        if not isinstance(frequency_tuple, tuple) or len(frequency_tuple) < 2:
            updated.append(frequency_tuple)
            continue
        cos_value, sin_value = frequency_tuple[0], frequency_tuple[1]
        extras = frequency_tuple[2:]
        if not isinstance(cos_value, torch.Tensor) or cos_value.dim() != 4:
            updated.append(frequency_tuple)
            continue
        for item in ranges:
            cos_value, sin_value = _compose_source_phase_range(
                cos_value,
                sin_value,
                int(item["start"]),
                int(item["end"]),
                float(item["source_id"]),
                float(item["phase_scale"]),
                theta,
            )
        updated.append((cos_value, sin_value, *extras))

    new_self_attention = tuple(updated)
    if isinstance(pe, list):
        return [new_self_attention] + list(pe[1:])
    return (new_self_attention,) + tuple(pe[1:])


def _patched_prepare_positional_embeddings(self, pixel_coords, frame_rate, x_dtype):
    pe = _ORIGINAL_PREPARE_PE(self, pixel_coords, frame_rate, x_dtype)
    ranges = list(getattr(self, "_qrf_pending_reference_ranges", []) or [])
    if ranges:
        theta = float(getattr(self, "positional_embedding_theta", 10000.0) or 10000.0)
        pe = _apply_reference_ranges_to_pe(pe, ranges, theta)
        _log(f"phase-tagged {len(ranges)} subject range(s): {ranges}")
    return pe


def _resize_reference_latent(latent: torch.Tensor, target_shape: Any) -> torch.Tensor:
    if target_shape is None or len(target_shape) < 5:
        return latent
    target_h, target_w = int(target_shape[3]), int(target_shape[4])
    source_h, source_w = int(latent.shape[3]), int(latent.shape[4])
    if (source_h, source_w) == (target_h, target_w):
        return latent
    batch, channels, frames, _, _ = latent.shape
    flattened = latent.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, source_h, source_w)
    flattened = F.interpolate(flattened, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return flattened.reshape(batch, frames, channels, target_h, target_w).permute(0, 2, 1, 3, 4).contiguous()


def _mask_to_token_scale(self, mask: Any, latent: torch.Tensor, token_count: int, device, dtype):
    if not isinstance(mask, torch.Tensor):
        return None
    batch, _, frames, height, width = latent.shape
    value = mask.to(device=device, dtype=torch.float32)
    if value.dim() == 4:
        value = value.unsqueeze(2)
    if value.dim() != 5:
        return None
    if value.shape[1] != 1:
        value = value.mean(dim=1, keepdim=True)
    if value.shape[3:] != (height, width):
        flattened = value.permute(0, 2, 1, 3, 4).reshape(-1, 1, value.shape[3], value.shape[4])
        flattened = F.interpolate(flattened, size=(height, width), mode="bilinear", align_corners=False)
        value = flattened.reshape(value.shape[0], value.shape[2], 1, height, width).permute(0, 2, 1, 3, 4)
    if value.shape[2] == 1 and frames > 1:
        value = value.expand(-1, -1, frames, -1, -1)
    elif value.shape[2] != frames:
        indices = torch.linspace(0, value.shape[2] - 1, frames, device=value.device).round().long()
        value = value.index_select(2, indices)

    try:
        mask_tokens, _ = self.patchifier.patchify(value.expand(batch, -1, -1, -1, -1))
        scale = mask_tokens.mean(dim=-1, keepdim=True)
    except Exception:
        scale = value[:, 0].reshape(value.shape[0], -1, 1)

    if scale.shape[1] != token_count:
        scale = F.interpolate(scale.transpose(1, 2), size=token_count, mode="linear", align_corners=False).transpose(1, 2)
    return scale.to(device=device, dtype=dtype).clamp(0.0, 1.0)


def _read_subjects(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    transformer_options = kwargs.get("transformer_options", {}) or {}
    subjects = kwargs.get("qrf_reference_subjects")
    if subjects is None and isinstance(transformer_options, dict):
        subjects = transformer_options.get("qrf_reference_subjects")
    if not isinstance(subjects, (list, tuple)):
        return []
    return [dict(item) for item in subjects if isinstance(item, dict) and isinstance(item.get("latent"), torch.Tensor)]


def _patched_process_input(self, x, keyframe_idxs, denoise_mask, **kwargs):
    result = _ORIGINAL_PROCESS_INPUT(self, x, keyframe_idxs, denoise_mask, **kwargs)
    tokens_list, coords_list, additional_args = result

    self._qrf_pending_reference_seq_len = 0
    self._qrf_pending_reference_ranges = []
    self._qrf_pending_reference_frames = 0

    subjects = _read_subjects(kwargs)
    if not subjects:
        return result

    transformer_options = kwargs.get("transformer_options", {}) or {}
    position_mode = kwargs.get("qrf_reference_position_mode")
    if position_mode is None and isinstance(transformer_options, dict):
        position_mode = transformer_options.get("qrf_reference_position_mode")
    position_mode = position_mode or "overlap"

    target_tokens = tokens_list[0]
    target_coords = coords_list[0]
    target_shape = additional_args.get("orig_shape")
    _, latent_to_pixel_coords = _import_comfy()

    token_blocks: list[torch.Tensor] = []
    coordinate_blocks: list[torch.Tensor] = []
    reference_ranges: list[dict[str, Any]] = []
    total_reference_frames = 0
    cursor = 0

    for subject_index, subject in enumerate(subjects, start=1):
        latent = subject["latent"]
        if latent.dim() == 4:
            latent = latent.unsqueeze(2)
        if latent.dim() != 5:
            _log(f"subject {subject_index}: ignored invalid latent shape {tuple(latent.shape)}")
            continue
        latent = latent.to(device=target_tokens.device, dtype=target_tokens.dtype)
        latent = _resize_reference_latent(latent, target_shape)

        try:
            reference_tokens, latent_coords = self.patchifier.patchify(latent)
            reference_coords = latent_to_pixel_coords(
                latent_coords=latent_coords,
                scale_factors=self.vae_scale_factors,
                causal_fix=self.causal_temporal_positioning,
            )
            if position_mode in ("prefix", "prefix_continuous"):
                temporal_end = float(reference_coords[:, 0, :, 1].max().item())
                reference_coords = reference_coords.clone()
                reference_coords[:, 0, :, :] -= temporal_end
            reference_tokens = self.patchify_proj(reference_tokens)
        except Exception as error:
            _log(f"subject {subject_index}: patchification failed: {type(error).__name__}: {error}")
            continue

        if reference_tokens.shape[0] != target_tokens.shape[0]:
            if reference_tokens.shape[0] == 1:
                reference_tokens = reference_tokens.expand(target_tokens.shape[0], -1, -1)
                reference_coords = reference_coords.expand(target_tokens.shape[0], -1, -1, -1)
            else:
                _log(f"subject {subject_index}: incompatible batch {reference_tokens.shape[0]} vs {target_tokens.shape[0]}")
                continue

        token_scale = _mask_to_token_scale(
            self,
            subject.get("spatial_mask"),
            latent,
            reference_tokens.shape[1],
            reference_tokens.device,
            reference_tokens.dtype,
        )
        if token_scale is not None:
            if token_scale.shape[0] == 1 and reference_tokens.shape[0] > 1:
                token_scale = token_scale.expand(reference_tokens.shape[0], -1, -1)
            background_floor = float(subject.get("background_floor", 0.0) or 0.0)
            token_scale = background_floor + (1.0 - background_floor) * token_scale
            reference_tokens = reference_tokens * token_scale

        start = cursor
        end = start + int(reference_tokens.shape[1])
        reference_ranges.append({
            "start": start,
            "end": end,
            "source_id": float(subject.get("source_id", subject_index + 1)),
            "phase_scale": float(subject.get("phase_scale", 1.0)),
            "subject_index": int(subject.get("subject_index", subject_index)),
        })
        cursor = end
        total_reference_frames += int(latent.shape[2])
        token_blocks.append(reference_tokens)
        coordinate_blocks.append(reference_coords)

    if not token_blocks:
        return result

    reference_tokens = torch.cat(token_blocks, dim=1)
    reference_coords = torch.cat(coordinate_blocks, dim=2)
    tokens_list[0] = torch.cat([reference_tokens, target_tokens], dim=1)
    coords_list[0] = torch.cat([reference_coords, target_coords], dim=2)

    reference_sequence_length = int(reference_tokens.shape[1])
    target_sequence_length = int(target_tokens.shape[1])
    patches_per_reference_frame = max(1, reference_sequence_length // max(1, total_reference_frames))
    target_frames = max(1, target_sequence_length // patches_per_reference_frame)

    additional_args["qrf_reference_seq_len"] = reference_sequence_length
    additional_args["qrf_reference_frames"] = total_reference_frames
    additional_args["qrf_target_seq_len"] = target_sequence_length
    additional_args["qrf_target_frames"] = target_frames

    self._qrf_pending_reference_seq_len = reference_sequence_length
    self._qrf_pending_reference_ranges = reference_ranges
    self._qrf_pending_reference_frames = total_reference_frames

    _log(
        f"prepended {reference_sequence_length} tokens from {len(reference_ranges)} subject(s); "
        f"ranges={reference_ranges}"
    )
    return tokens_list, coords_list, additional_args


def _extend_prefix_in_tensor(tensor: torch.Tensor, target_size: int, prefix_size: int) -> torch.Tensor:
    if tensor.dim() < 2 or tensor.shape[1] != target_size:
        return tensor
    prefix = tensor[:, 0:1, ...].expand(-1, prefix_size, *tensor.shape[2:])
    return torch.cat([prefix, tensor], dim=1)


def _walk_and_extend_item(
    item: Any,
    target_sequence_length: int,
    reference_sequence_length: int,
    target_frames: int,
    reference_frames: int,
    zero_reference_timesteps: bool,
    depth: int = 0,
):
    if item is None or depth > 5:
        return item, 0
    if isinstance(item, list):
        total = 0
        for index, child in enumerate(item):
            item[index], count = _walk_and_extend_item(
                child,
                target_sequence_length,
                reference_sequence_length,
                target_frames,
                reference_frames,
                zero_reference_timesteps,
                depth + 1,
            )
            total += count
        return item, total
    if isinstance(item, tuple):
        values = []
        total = 0
        for child in item:
            new_child, count = _walk_and_extend_item(
                child,
                target_sequence_length,
                reference_sequence_length,
                target_frames,
                reference_frames,
                zero_reference_timesteps,
                depth + 1,
            )
            values.append(new_child)
            total += count
        return tuple(values), total

    if hasattr(item, "data") and hasattr(item, "num_frames") and hasattr(item, "patches_per_frame"):
        try:
            data = item.data
            number_of_frames = int(item.num_frames)
            patches_per_frame = int(item.patches_per_frame)
            if not isinstance(data, torch.Tensor) or data.dim() < 2:
                return item, 0
            if patches_per_frame == 1 and number_of_frames == 1:
                return item, 0
            if patches_per_frame > 1 and number_of_frames * patches_per_frame == target_sequence_length:
                prefix = data[:, 0:1, :].expand(-1, reference_frames, -1).contiguous()
                if zero_reference_timesteps:
                    prefix = torch.zeros_like(prefix)
                item.data = torch.cat([prefix, data], dim=1).contiguous()
                item.num_frames = number_of_frames + reference_frames
                return item, 1
            if patches_per_frame == 1 and number_of_frames == target_sequence_length:
                prefix = data[:, 0:1, :].expand(-1, reference_sequence_length, -1).contiguous()
                if zero_reference_timesteps:
                    prefix = torch.zeros_like(prefix)
                item.data = torch.cat([prefix, data], dim=1).contiguous()
                item.num_frames = number_of_frames + reference_sequence_length
                return item, 1
        except Exception as error:
            _log(f"could not extend compressed timestep: {type(error).__name__}: {error}")
        return item, 0

    if isinstance(item, torch.Tensor) and item.dim() >= 2:
        if item.shape[1] == target_sequence_length:
            value = _extend_prefix_in_tensor(item, target_sequence_length, reference_sequence_length)
            if zero_reference_timesteps:
                value = value.clone()
                value[:, :reference_sequence_length] = 0.0
            return value, 1
        if item.shape[1] == target_frames:
            value = _extend_prefix_in_tensor(item, target_frames, reference_frames)
            if zero_reference_timesteps:
                value = value.clone()
                value[:, :reference_frames] = 0.0
            return value, 1
    return item, 0


def _patched_prepare_timestep(self, timestep, batch_size, hidden_dtype, **kwargs):
    reference_sequence_length = int(kwargs.get("qrf_reference_seq_len", 0) or 0)
    if reference_sequence_length <= 0:
        return _ORIGINAL_PREPARE_TIMESTEP(self, timestep, batch_size, hidden_dtype, **kwargs)

    reference_frames = max(1, int(kwargs.get("qrf_reference_frames", 1) or 1))
    target_sequence_length = int(kwargs.get("qrf_target_seq_len", 0) or 0)
    target_frames = max(1, int(kwargs.get("qrf_target_frames", 1) or 1))
    if target_sequence_length <= 0:
        return _ORIGINAL_PREPARE_TIMESTEP(self, timestep, batch_size, hidden_dtype, **kwargs)

    result = _ORIGINAL_PREPARE_TIMESTEP(self, timestep, batch_size, hidden_dtype, **kwargs)
    if not isinstance(result, (tuple, list)):
        return result

    zero_reference_timesteps = bool(getattr(self, "_qrf_zero_reference_timesteps", False))
    was_tuple = isinstance(result, tuple)
    values = list(result)
    extension_count = 0
    for index, value in enumerate(values):
        values[index], count = _walk_and_extend_item(
            value,
            target_sequence_length,
            reference_sequence_length,
            target_frames,
            reference_frames,
            zero_reference_timesteps,
        )
        extension_count += count
    _log(f"extended {extension_count} timestep/modulation object(s)")
    return tuple(values) if was_tuple else values


class _QrusheRUnpatchifyWrapper:
    def __init__(self, original_unpatchify, model_reference):
        self._original_unpatchify = original_unpatchify
        self._model_reference = model_reference

    def __call__(self, latents, **kwargs):
        prefix_length = int(getattr(self._model_reference, "_qrf_pending_reference_seq_len", 0) or 0)
        if prefix_length > 0:
            latents = latents[:, prefix_length:, :]
            self._model_reference._qrf_pending_reference_seq_len = 0
            self._model_reference._qrf_pending_reference_ranges = []
        return self._original_unpatchify(latents, **kwargs)


def apply_patchifier_wrap(model_instance) -> bool:
    patchifier = model_instance.patchifier
    if getattr(patchifier, "_qrf_multiface_wrapped", False):
        return False
    patchifier.unpatchify = _QrusheRUnpatchifyWrapper(patchifier.unpatchify, model_instance)
    patchifier._qrf_multiface_wrapped = True
    return True


def apply_global_patches(verbose: bool = False) -> bool:
    global _PATCHES_APPLIED, _PATCH_ERROR, _ORIGINAL_PROCESS_INPUT
    global _ORIGINAL_PREPARE_TIMESTEP, _ORIGINAL_PREPARE_PE, _VERBOSE
    _VERBOSE = bool(verbose)
    if _PATCHES_APPLIED:
        return True

    try:
        av_module, _ = _import_comfy()
        model_class = av_module.LTXAVModel
        if getattr(model_class, "_qrf_multiface_patches_applied", False):
            _PATCHES_APPLIED = True
            return True

        _ORIGINAL_PROCESS_INPUT = model_class._process_input
        _ORIGINAL_PREPARE_TIMESTEP = model_class._prepare_timestep
        model_class._process_input = _patched_process_input
        model_class._prepare_timestep = _patched_prepare_timestep

        positional_owner = None
        for candidate in model_class.__mro__:
            if "_prepare_positional_embeddings" in candidate.__dict__:
                positional_owner = candidate
                break
        if positional_owner is None:
            raise RuntimeError("Could not locate _prepare_positional_embeddings")
        _ORIGINAL_PREPARE_PE = positional_owner._prepare_positional_embeddings
        positional_owner._prepare_positional_embeddings = _patched_prepare_positional_embeddings

        model_class._qrf_multiface_patches_applied = True
        _PATCHES_APPLIED = True
        _PATCH_ERROR = None
        print("[QrusheRFork MultiFace] LTX multi-subject patches applied.")
        return True
    except Exception as error:
        _PATCH_ERROR = f"{type(error).__name__}: {error}"
        print(f"[QrusheRFork MultiFace] Patch installation failed: {_PATCH_ERROR}")
        return False


def install_on_model(model, *, zero_reference_timesteps: bool = False, verbose: bool = False) -> None:
    if not apply_global_patches(verbose=verbose):
        raise RuntimeError(f"Could not install LTX patches: {_PATCH_ERROR}")
    try:
        inner = model.model if hasattr(model, "model") else model
        diffusion_model = inner.diffusion_model if hasattr(inner, "diffusion_model") else inner
        apply_patchifier_wrap(diffusion_model)
        diffusion_model._qrf_zero_reference_timesteps = bool(zero_reference_timesteps)
    except Exception as error:
        raise RuntimeError(f"Could not access LTX diffusion model: {error}") from error
