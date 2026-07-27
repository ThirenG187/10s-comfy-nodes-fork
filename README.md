# 10S Multi-Face Identity Reinforcer `_QrusheRFork`

Experimental, namespaced fork of TenStrip's **LTX Face Identity Reinforcer**
for LTX2/LTX-AV and the LTX-Best-Face-ID LoRA.

This package registers one new node:

`🧑‍🤝‍🧑 LTX Multi-Face Identity Reinforcer _QrusheRFork`

Category:

`10S Nodes_QrusheRFork/Identity`

## What changed

- Up to four independent subjects.
- Two references per subject: a primary image and an optional alternate/cropped
  view of the **same** person.
- Detects all faces in a supplied target/first-frame image.
- Assigns identities `left_to_right`, `largest_first`, or by manual face index.
- Aligns each reference face to its assigned target face.
- Builds a separate spatial mask for each subject.
- Prepends each subject as a separate reference-token block.
- Applies a separate RoPE source phase range to each subject.
- Uses suffixed node IDs and `qrf_*` model-option keys to reduce collisions with
  the original 10S pack.

## Install

1. Extract the folder `10S-Comfy-nodes_QrusheRFork` into:

   `ComfyUI/custom_nodes/`

2. The final path should be:

   `ComfyUI/custom_nodes/10S-Comfy-nodes_QrusheRFork/__init__.py`

3. Restart ComfyUI.

The original TenStrip pack can remain installed. This fork does not register
copies of the other 10S nodes.

## Basic wiring

```text
Load Model
   ↓
Load LTX-Best-Face-ID LoRA
   ↓
LTX Multi-Face Identity Reinforcer _QrusheRFork
   ↓
Sampler
```

Connect these to the reinforcer:

- `vae`
- `target_latent`: the same latent dimensions used for sampling
- `target_image`: the same composition/first-frame image used by i2v
- `reference_image`: Subject 1 primary identity reference
- `reference_image_2`: optional second view of Subject 1
- `subject_2_reference_image`: Subject 2 primary identity reference
- Further Subject 3/4 inputs only when needed

`target_image` is required when two or more subjects are connected.

## First test settings

```text
assignment_mode: left_to_right
identity_strength: 1.0
subject_2_strength: 1.0
spatial_gating: mask_soft
face_padding: 0.15
source_id_base: 2.0
source_id_stride: 1.0
phase_scale: 1.0
background_reference_strength: 0.02
debug: true
```

For a two-person first frame, Subject 1 maps to the leftmost detected face and
Subject 2 maps to the next face. Use `manual` when detector ordering is not the
ordering you want.

## Important limitation

This is **first-frame multi-face identity binding**, not optical face tracking
on generated frames. LTX generates the sequence inside the denoising process,
so the node cannot re-detect every output frame before the next frame is
created. Separate source-phase ranges and face-region masks are intended to
reduce identity blending and swaps, but crossing subjects, severe occlusion,
or large composition changes can still cause drift.

## Detection

Detection priority is:

1. OpenCV YuNet (downloads its small ONNX model to the existing 10S cache on
   first use)
2. MediaPipe
3. OpenCV Haar cascade

Enable `debug` to see detections, assignments, latent shapes, and source IDs in
the ComfyUI console.

## Troubleshooting

- **No target faces detected:** use a clearer first frame, reduce extreme face
  angles, or confirm OpenCV/YuNet is available.
- **Wrong identity assigned:** set `assignment_mode=manual` and choose explicit
  `subject_N_face_index` values.
- **Faces blend:** reduce each strength to `0.75-0.9`, keep
  `spatial_gating=mask_soft`, or lower `background_reference_strength` to `0`.
- **Tensor mismatch:** capture the full console trace with `debug=true`; LTX and
  ComfyUI internals change frequently and may require a compatibility patch.

## Validation performed

- Python compilation for every package module.
- Static tests for independent source-phase ranges.
- Static tests for left-to-right and largest-first face ordering.

A full LTX2 generation could not be executed in the build container, so this
release should be treated as an experimental test build.

See `UPSTREAM_NOTICE.md` for attribution.
# 10s-comfy-nodes-fork
