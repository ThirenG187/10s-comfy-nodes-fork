"""BFS-style multi-reference token injection for LTX2/LTX-AV.

This module adapts the exact overlap/source-phase approach used by
alisson-anjos/ComfyUI-BFSNodes' ``LTX Identity Transfer`` node to support the
QrusheRFork multi-face subject specifications.

Reference latents are appended as clean, non-rendered token blocks after the
video tokens. Each block may have its own source-phase segment, TASS layout,
strength, and spatial gate. The blocks are removed before unpatchification.

The implementation is namespaced with ``qrf_bfs_*`` keys and instance flags so
it can coexist with the original 10S and BFS node packs when those nodes have
not already patched the same cloned model instance.
"""
from __future__ import annotations

import copy
import types
from typing import Any

import torch
import torch.nn.functional as F

# Must match the BFS/ltx-trainer TASS convention.
STRATA_SLOT_WIDTH = 1.5

_VERBOSE = False


def _log(*parts: Any) -> None:
    if _VERBOSE:
        print("[QrusheRFork BFS Hybrid] " + " ".join(str(part) for part in parts), flush=True)


def _shape(value: Any) -> str:
    try:
        if hasattr(value, "shape"):
            return f"T{tuple(value.shape)}"
        if isinstance(value, (list, tuple)):
            return f"{type(value).__name__}[{', '.join(_shape(item) for item in value)}]"
        return type(value).__name__
    except Exception:
        return "?"


def _find_ltxv(model):
    value = getattr(model, "model", model)
    value = getattr(value, "diffusion_model", value)
    return value


def _read_reference_specs(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    options = kwargs.get("transformer_options") or {}
    specs = kwargs.get("qrf_bfs_reference_specs")
    if specs is None and isinstance(options, dict):
        specs = options.get("qrf_bfs_reference_specs")
    if not isinstance(specs, (list, tuple)):
        return []
    return [
        dict(spec)
        for spec in specs
        if isinstance(spec, dict) and isinstance(spec.get("latent"), torch.Tensor)
    ]


def _rotate_reference_block(
    pe: Any,
    start: int,
    length: int,
    segment_value: float,
    theta: float = 10000.0,
):
    """Compose a source-phase rotation over one reference token range.

    Supports both current fused 2x2 rotation matrices and the legacy separate
    cosine/sine representation used by earlier ComfyUI LTX builds.
    """
    if length <= 0 or segment_value == 0.0:
        return pe
    if not isinstance(pe, (list, tuple)) or not pe:
        return pe

    first = pe[0]
    if torch.is_tensor(first) and first.dim() >= 3 and first.shape[-2:] == (2, 2):
        matrix = first
        rest = tuple(pe[1:])
        rotary_length = matrix.shape[-3]
        dimensions = torch.arange(rotary_length, device=matrix.device, dtype=torch.float32)
        rate = theta ** (-dimensions / float(rotary_length))
        phase = float(segment_value) * rate
        cosine = phase.cos().to(matrix.dtype)
        sine = phase.sin().to(matrix.dtype)
        phase_rotation = torch.stack(
            (
                torch.stack((cosine, -sine), dim=-1),
                torch.stack((sine, cosine), dim=-1),
            ),
            dim=-2,
        )
        output = matrix.clone()
        output[:, start:start + length] = torch.matmul(
            phase_rotation,
            matrix[:, start:start + length],
        )
        return (output, *rest)

    if len(pe) < 2 or not torch.is_tensor(pe[0]) or not torch.is_tensor(pe[1]):
        return pe
    cosine, sine = pe[0], pe[1]
    rest = tuple(pe[2:])
    rotary_length = cosine.shape[-1]
    dimensions = torch.arange(rotary_length, device=cosine.device, dtype=torch.float32)
    rate = theta ** (-dimensions / float(rotary_length))
    phase = float(segment_value) * rate
    phase_cosine = phase.cos().to(cosine.dtype)
    phase_sine = phase.sin().to(sine.dtype)

    index = [slice(None)] * cosine.dim()
    index[-2] = slice(start, start + length)
    index = tuple(index)
    original_cosine = cosine[index]
    original_sine = sine[index]
    cosine = cosine.clone()
    sine = sine.clone()
    cosine[index] = original_cosine * phase_cosine - original_sine * phase_sine
    sine[index] = original_sine * phase_cosine + original_cosine * phase_sine
    return (cosine, sine, *rest)


def _apply_tass_layout(
    reference_positions: torch.Tensor,
    target_positions: torch.Tensor,
    layout: str,
    *,
    strata_start: float | None = None,
) -> torch.Tensor:
    """Apply BFS/ltx-trainer overlap, ST-DRC, or strata coordinate layout."""
    if layout == "overlap":
        return reference_positions
    if layout == "st_drc":
        target_extent = target_positions.amax(dim=2, keepdim=True)
        reference_origin = reference_positions.amin(dim=2, keepdim=True)
        return reference_positions + (target_extent - reference_origin)
    if layout == "strata":
        if strata_start is None:
            raise ValueError("layout='strata' requires a strata_start")
        shifted = reference_positions.clone()
        reference_origin_t = shifted[:, 0:1, :].amin(dim=2, keepdim=True)
        shifted[:, 0:1, :] += strata_start - reference_origin_t
        return shifted
    raise ValueError(f"Unsupported reference layout: {layout!r}")


def _mask_to_token_scale(
    model_instance,
    mask: Any,
    latent: torch.Tensor,
    token_count: int,
    device,
    dtype,
) -> torch.Tensor | None:
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
        flattened = value.permute(0, 2, 1, 3, 4).reshape(
            -1, 1, value.shape[3], value.shape[4]
        )
        flattened = F.interpolate(
            flattened,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        value = flattened.reshape(
            value.shape[0], value.shape[2], 1, height, width
        ).permute(0, 2, 1, 3, 4)

    if value.shape[2] == 1 and frames > 1:
        value = value.expand(-1, -1, frames, -1, -1)
    elif value.shape[2] != frames:
        indices = torch.linspace(
            0,
            value.shape[2] - 1,
            frames,
            device=value.device,
        ).round().long()
        value = value.index_select(2, indices)

    try:
        mask_tokens, _ = model_instance.patchifier.patchify(
            value.expand(batch, -1, -1, -1, -1)
        )
        scale = mask_tokens.mean(dim=-1, keepdim=True)
    except Exception:
        scale = value[:, 0].reshape(value.shape[0], -1, 1)

    if scale.shape[1] != token_count:
        scale = F.interpolate(
            scale.transpose(1, 2),
            size=token_count,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
    return scale.to(device=device, dtype=dtype).clamp(0.0, 1.0)


def _install_instance_patches(ltxv, *, verbose: bool = False) -> None:
    """Install idempotent BFS-style patches on one cloned LTX model instance."""
    global _VERBOSE
    _VERBOSE = bool(verbose)

    if getattr(ltxv, "_qrf_bfs_multiface_patched", False):
        return

    original_process_input = ltxv._process_input
    original_prepare_timestep = ltxv._prepare_timestep
    original_prepare_pe = ltxv._prepare_positional_embeddings
    original_process_output = ltxv._process_output
    original_forward = getattr(ltxv, "_forward", None)

    if original_forward is not None:
        def _forward_capture_fps(
            self,
            x,
            timestep,
            context,
            attention_mask,
            frame_rate=25,
            transformer_options={},
            keyframe_idxs=None,
            denoise_mask=None,
            **kwargs,
        ):
            self._qrf_bfs_frame_rate = float(frame_rate)
            return original_forward(
                x,
                timestep,
                context,
                attention_mask,
                frame_rate=frame_rate,
                transformer_options=transformer_options,
                keyframe_idxs=keyframe_idxs,
                denoise_mask=denoise_mask,
                **kwargs,
            )

        ltxv._forward = types.MethodType(_forward_capture_fps, ltxv)

    def process_input(self, x, keyframe_idxs, denoise_mask, **kwargs):
        self._qrf_bfs_ref_len = 0
        self._qrf_bfs_blocks = []
        self._qrf_bfs_ref_frames = 0
        self._qrf_bfs_patches_per_frame = None

        output = original_process_input(x, keyframe_idxs, denoise_mask, **kwargs)
        specs = _read_reference_specs(kwargs)
        if not specs:
            return output

        from comfy.ldm.lightricks.model import latent_to_pixel_coords

        token_streams, pixel_streams, additional = output
        is_av = isinstance(token_streams, (list, tuple))
        video_tokens = token_streams[0] if is_av else token_streams
        video_coords = pixel_streams[0] if is_av else pixel_streams
        target_length = int(video_tokens.shape[1])
        self._qrf_bfs_target_len = target_length

        frame_rate = float(getattr(self, "_qrf_bfs_frame_rate", 25.0))
        target_max_t_raw = float(video_coords[:, 0, :].amax().item())
        blocks: list[tuple[int, int, float]] = []
        offset = target_length
        total_reference_frames = 0
        patches_per_frame: int | None = None

        _log(
            "process_input:",
            "target", _shape(video_tokens),
            "coords", _shape(video_coords),
            "refs", len(specs),
            "fps", frame_rate,
        )

        for fallback_index, spec in enumerate(specs):
            reference_latent = spec["latent"]
            if reference_latent.dim() == 4:
                reference_latent = reference_latent.unsqueeze(2)
            if reference_latent.dim() != 5:
                raise ValueError(
                    f"Reference {fallback_index + 1} latent must be [B,C,F,H,W], "
                    f"got {tuple(reference_latent.shape)}"
                )
            reference_latent = reference_latent.to(
                device=video_tokens.device,
                dtype=video_tokens.dtype,
            )

            reference_tokens, latent_coords = self.patchifier.patchify(reference_latent)
            reference_coords = latent_to_pixel_coords(
                latent_coords=latent_coords,
                scale_factors=self.vae_scale_factors,
                causal_fix=self.causal_temporal_positioning,
            )

            downscale_factor = float(spec.get("downscale_factor", 1.0) or 1.0)
            if downscale_factor != 1.0:
                reference_coords = reference_coords.clone()
                reference_coords[:, 1, ...] *= downscale_factor
                reference_coords[:, 2, ...] *= downscale_factor

            layout = str(spec.get("layout", "overlap"))
            strata_start_raw = None
            if layout == "strata":
                strata_slot = int(spec.get("strata_slot", fallback_index))
                strata_start_seconds = (
                    target_max_t_raw / frame_rate
                    + (strata_slot + 1) * STRATA_SLOT_WIDTH
                )
                strata_start_raw = strata_start_seconds * frame_rate
            reference_coords = _apply_tass_layout(
                reference_coords,
                video_coords,
                layout,
                strata_start=strata_start_raw,
            )

            reference_tokens = self.patchify_proj(reference_tokens)
            if reference_tokens.shape[0] != video_tokens.shape[0]:
                if reference_tokens.shape[0] == 1:
                    reference_tokens = reference_tokens.expand(
                        video_tokens.shape[0], -1, -1
                    )
                    reference_coords = reference_coords.expand(
                        video_coords.shape[0], *([-1] * (reference_coords.dim() - 1))
                    )
                else:
                    raise ValueError(
                        f"Reference batch {reference_tokens.shape[0]} does not match "
                        f"target batch {video_tokens.shape[0]}"
                    )

            token_scale = _mask_to_token_scale(
                self,
                spec.get("spatial_mask"),
                reference_latent,
                reference_tokens.shape[1],
                reference_tokens.device,
                reference_tokens.dtype,
            )
            if token_scale is not None:
                if token_scale.shape[0] == 1 and reference_tokens.shape[0] > 1:
                    token_scale = token_scale.expand(reference_tokens.shape[0], -1, -1)
                background_floor = float(spec.get("background_floor", 0.0) or 0.0)
                token_scale = background_floor + (1.0 - background_floor) * token_scale
                reference_tokens = reference_tokens * token_scale

            token_strength = float(spec.get("token_strength", 1.0) or 1.0)
            if token_strength != 1.0:
                reference_tokens = reference_tokens * token_strength

            reference_length = int(reference_tokens.shape[1])
            reference_frames = int(reference_latent.shape[2])
            if reference_frames > 0:
                current_ppf = max(1, reference_length // reference_frames)
                if patches_per_frame is None:
                    patches_per_frame = current_ppf

            video_tokens = torch.cat([video_tokens, reference_tokens], dim=1)
            video_coords = torch.cat(
                [video_coords, reference_coords.to(video_coords)],
                dim=2,
            )
            segment_value = float(
                spec.get(
                    "seg_value",
                    float(spec.get("source_id", fallback_index + 2))
                    * float(spec.get("phase_scale", 1.0)),
                )
            )
            blocks.append((offset, reference_length, segment_value))
            offset += reference_length
            total_reference_frames += reference_frames

        reference_length_total = offset - target_length
        self._qrf_bfs_ref_len = reference_length_total
        self._qrf_bfs_blocks = blocks
        self._qrf_bfs_ref_frames = total_reference_frames
        self._qrf_bfs_patches_per_frame = patches_per_frame
        additional = dict(additional)
        additional["qrf_bfs_ref_len"] = reference_length_total

        if is_av:
            token_streams = [video_tokens, *list(token_streams[1:])]
            pixel_streams = [video_coords, *list(pixel_streams[1:])]
        else:
            token_streams, pixel_streams = video_tokens, video_coords

        _log("appended blocks", blocks, "total_ref", reference_length_total)
        return token_streams, pixel_streams, additional

    def prepare_timestep(self, timestep, batch_size, hidden_dtype, **kwargs):
        reference_length = int(getattr(self, "_qrf_bfs_ref_len", 0) or 0)
        target_length = getattr(self, "_qrf_bfs_target_len", None)
        if not reference_length or target_length is None:
            return original_prepare_timestep(
                timestep,
                batch_size,
                hidden_dtype,
                **kwargs,
            )

        clean_references = bool(
            getattr(self, "_qrf_bfs_clean_reference_timesteps", True)
        )
        reference_frames = max(
            1,
            int(getattr(self, "_qrf_bfs_ref_frames", 1) or 1),
        )
        patches_per_frame = getattr(self, "_qrf_bfs_patches_per_frame", None)

        if timestep.dim() <= 1:
            timestep = timestep.reshape(-1, 1).expand(batch_size, target_length).contiguous()

        if timestep.dim() >= 2:
            current_length = int(timestep.shape[1])
            grid_mask = kwargs.get("grid_mask")
            full_grid_length = (
                int(grid_mask.shape[-1])
                if grid_mask is not None and hasattr(grid_mask, "shape")
                else None
            )

            if full_grid_length is not None and current_length == full_grid_length:
                count = reference_length
            elif current_length > target_length + reference_length:
                timestep = timestep[:, :target_length]
                current_length = target_length
                count = reference_length
            elif current_length == target_length:
                count = reference_length
            elif patches_per_frame and current_length * int(patches_per_frame) == target_length:
                count = reference_frames
            elif current_length in (
                target_length + reference_length,
                (target_length // max(1, int(patches_per_frame or 1))) + reference_frames,
            ):
                count = 0
            else:
                count = 0
                _log(
                    "prepare_timestep unexpected length",
                    current_length,
                    "target", target_length,
                    "ref", reference_length,
                    "ppf", patches_per_frame,
                )

            if count > 0:
                if clean_references:
                    reference_timestep = torch.zeros(
                        timestep.shape[0],
                        count,
                        *timestep.shape[2:],
                        device=timestep.device,
                        dtype=timestep.dtype,
                    )
                else:
                    reference_timestep = timestep[:, 0:1, ...].expand(
                        -1,
                        count,
                        *timestep.shape[2:],
                    ).contiguous()
                timestep = torch.cat([timestep, reference_timestep], dim=1)

            grid_mask = kwargs.get("grid_mask")
            if grid_mask is not None and hasattr(grid_mask, "shape"):
                gap = int(timestep.shape[1] - grid_mask.shape[-1])
                if 0 < gap <= reference_length:
                    padding = torch.ones(
                        *grid_mask.shape[:-1],
                        gap,
                        dtype=grid_mask.dtype,
                        device=grid_mask.device,
                    )
                    kwargs = dict(kwargs)
                    kwargs["grid_mask"] = torch.cat([grid_mask, padding], dim=-1)

        return original_prepare_timestep(
            timestep,
            batch_size,
            hidden_dtype,
            **kwargs,
        )

    def prepare_positional_embeddings(self, pixel_coords, frame_rate, x_dtype):
        positional_embeddings = original_prepare_pe(pixel_coords, frame_rate, x_dtype)
        blocks = list(getattr(self, "_qrf_bfs_blocks", []) or [])
        if not blocks:
            return positional_embeddings
        theta = float(getattr(self, "_qrf_bfs_rope_theta", 10000.0) or 10000.0)

        def rotate(video_pe):
            for start, length, segment_value in blocks:
                video_pe = _rotate_reference_block(
                    video_pe,
                    start,
                    length,
                    segment_value,
                    theta,
                )
            return video_pe

        # LTX-AV: [(video_pe, cross_video), (audio_pe, cross_audio)]
        if (
            isinstance(positional_embeddings, list)
            and positional_embeddings
            and isinstance(positional_embeddings[0], (list, tuple))
            and positional_embeddings[0]
            and isinstance(positional_embeddings[0][0], (list, tuple))
        ):
            video_pe, cross_video = positional_embeddings[0][0], positional_embeddings[0][1]
            positional_embeddings = [
                (rotate(video_pe), cross_video),
                *list(positional_embeddings[1:]),
            ]
            return positional_embeddings
        return rotate(positional_embeddings)

    def process_output(self, x, embedded_timestep, keyframe_idxs, **kwargs):
        reference_length = int(getattr(self, "_qrf_bfs_ref_len", 0) or 0)
        if reference_length:
            try:
                from comfy.ldm.lightricks.av_model import CompressedTimestep

                if isinstance(x, (list, tuple)):
                    x = [x[0][:, :x[0].shape[1] - reference_length], *list(x[1:])]
                    timestep_list = (
                        list(embedded_timestep)
                        if isinstance(embedded_timestep, (list, tuple))
                        else [embedded_timestep]
                    )
                    video_timestep = timestep_list[0]
                    if isinstance(video_timestep, CompressedTimestep):
                        patches_per_frame = max(
                            1,
                            int(getattr(video_timestep, "patches_per_frame", 1) or 1),
                        )
                        reference_frames = max(1, reference_length // patches_per_frame)
                        trimmed = copy.copy(video_timestep)
                        trimmed.data = video_timestep.data[
                            :, :video_timestep.num_frames - reference_frames
                        ].contiguous()
                        trimmed.num_frames = video_timestep.num_frames - reference_frames
                        timestep_list[0] = trimmed
                    elif (
                        hasattr(video_timestep, "shape")
                        and video_timestep.dim() >= 2
                        and video_timestep.shape[1] > 1
                    ):
                        timestep_list[0] = video_timestep[
                            :, :video_timestep.shape[1] - reference_length
                        ]
                    embedded_timestep = timestep_list
                else:
                    x = x[:, :x.shape[1] - reference_length]
                    if (
                        hasattr(embedded_timestep, "shape")
                        and embedded_timestep.dim() >= 2
                        and embedded_timestep.shape[1] > 1
                    ):
                        embedded_timestep = embedded_timestep[
                            :, :embedded_timestep.shape[1] - reference_length
                        ]
            finally:
                self._qrf_bfs_ref_len = 0
                self._qrf_bfs_blocks = []

        return original_process_output(
            x,
            embedded_timestep,
            keyframe_idxs,
            **kwargs,
        )

    ltxv._process_input = types.MethodType(process_input, ltxv)
    ltxv._prepare_timestep = types.MethodType(prepare_timestep, ltxv)
    ltxv._prepare_positional_embeddings = types.MethodType(
        prepare_positional_embeddings,
        ltxv,
    )
    ltxv._process_output = types.MethodType(process_output, ltxv)
    ltxv._qrf_bfs_multiface_patched = True
    _log("installed on", type(ltxv).__name__)


def install_on_model(
    model,
    *,
    clean_reference_timesteps: bool = True,
    verbose: bool = False,
) -> None:
    """Install the hybrid reference patch on a cloned ComfyUI MODEL."""
    ltxv = _find_ltxv(model)
    required = (
        "_process_input",
        "_prepare_timestep",
        "_prepare_positional_embeddings",
        "_process_output",
        "patchifier",
        "patchify_proj",
    )
    missing = [name for name in required if not hasattr(ltxv, name)]
    if missing:
        raise RuntimeError(
            "The supplied MODEL does not look like a compatible LTX model; "
            f"missing: {', '.join(missing)}"
        )
    _install_instance_patches(ltxv, verbose=verbose)
    ltxv._qrf_bfs_clean_reference_timesteps = bool(clean_reference_timesteps)
    ltxv._qrf_bfs_rope_theta = 10000.0
