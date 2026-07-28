# 10S Multi-Face Identity Reinforcer — BFS Hybrid `_QrusheRFork`

Experimental multi-face identity node for LTX2/LTX-AV and Best-Face-ID-style
LoRAs. It combines TenStrip-style face assignment/alignment with the exact
reference-token architecture used by BFS's **LTX Identity Transfer** node.

Registered node:

`🧑‍🤝‍🧑 LTX Multi-Face Identity Reinforcer BFS Hybrid _QrusheRFork`

Category:

`10S Nodes_QrusheRFork/Identity`

## What v0.2 changes

- Up to four independent identities, with two views per identity.
- Detects and assigns identities to faces in a group/composition image.
- Appends reference blocks **after** target video tokens, matching BFS.
- Removes reference blocks before rendering, so they are conditioning only.
- Exact `overlap`, `st_drc`, and temporal `strata` coordinate layouts.
- Independent source-phase RoPE segment for every reference block.
- Clean timestep-0 reference tokens by default.
- Optional BFS/ST-DRC reference-CFG.
- Optional injection of `target_image` as the first BFS reference block for
  guided T2V composition.
- Retains soft/hard spatial gating around each assigned target face.

## Install

Extract the folder into:

`ComfyUI/custom_nodes/10S-Comfy-nodes_QrusheRFork/`

Restart ComfyUI. The original 10S and BFS packs may remain installed; this node
uses a suffixed node ID and namespaced model-option keys.

## Wiring

```text
LTX model
  -> Load Best-Face-ID / compatible LoRA
  -> LTX Multi-Face Identity Reinforcer BFS Hybrid _QrusheRFork
  -> sampler
```

Connect:

- `target_latent`: the empty T2V latent or your I2V latent.
- `target_image`: group/composition guide containing all target faces.
- `reference_image`: Subject 1 identity.
- `subject_2_reference_image`: Subject 2 identity, and so on.
- Secondary inputs are alternate images of the **same** corresponding subject.

## T2V modes

### Assignment only

```text
target_image_mode: assignment_only
```

The target image is used only to detect face locations and assign the separate
identity references. It is not injected into the transformer.

### BFS guided T2V

```text
target_image_mode: assignment_and_bfs_reference
```

The target image is VAE-encoded and injected as the first clean BFS reference
block. This gives the model a direct composition/layout reference while the
separate subject blocks reinforce each identity.

Because it becomes reference block 1, source IDs are automatically allocated:

```text
target guide -> source_id_base
Subject 1    -> source_id_base + source_id_stride
Subject 2    -> source_id_base + 2 * source_id_stride
...
```

Start with `target_guide_strength=0.25-0.40`. A high value can preserve the
placeholder faces or appearance from the target image too strongly.

## Recommended first two-person T2V test

```text
assignment_mode: left_to_right
reference_layout: overlap
target_image_mode: assignment_and_bfs_reference
target_guide_strength: 0.35
identity_strength: 1.0
subject_2_strength: 1.0
spatial_gating: mask_soft
face_padding: 0.15
source_id_base: 2.0
source_id_stride: 1.0
phase_scale: 1.0
background_reference_strength: 0.02
zero_reference_timesteps: true
reference_guidance_scale: 1.0
debug: true
```

If identities are weak, test `reference_guidance_scale=2.0`. This adds a third
forward pass per denoising step and therefore increases generation time and
VRAM pressure.

## Layout selection

- `overlap`: references reuse the target coordinate range and are separated by
  source phase. This is the expected starting mode for Best-Face-ID-style LoRAs.
- `st_drc`: shifts reference coordinates beyond the target extent on all axes.
- `strata`: puts each reference in a separate temporal slot while keeping H/W
  overlapping the target.

The selected layout should match how the LoRA was trained. A different layout
is not merely a strength adjustment; it changes the coordinate convention.

## Important limitations

- The node binds identities using the supplied target image. It does not
  re-detect faces on every generated frame during denoising.
- Crossing people, strong occlusion, major pose changes, and faces leaving
  their original regions can still cause swaps or drift.
- Target-image injection is experimental when combined with a LoRA trained for
  only one reference block.
- A complete LTX generation could not be executed in the build environment;
  this package passed static and mocked configuration tests only.

See `UPSTREAM_NOTICE.md` for licensing and attribution.
