# Upstream notice

This experimental package combines and adapts mechanisms from two public
ComfyUI node projects:

## TenStrip / 10S-Comfy-nodes

Upstream: `https://github.com/TenStrip/10S-Comfy-nodes`

License: MIT. A copy is included as `LICENSE_TENSTRIP_MIT`.

Adapted mechanisms:

- Multi-backend face detection and face-region preparation
- Target-face assignment and reference-to-target alignment
- Best-Face-ID spatial gating and source-phase conventions

## alisson-anjos / ComfyUI-BFSNodes

Upstream: `https://github.com/alisson-anjos/ComfyUI-BFSNodes`

License: GPL-3.0.

The BFS `ltx_identity_overlap.py` implementation was inspected on 28 July
2026. This fork adapts its LTX Identity Transfer architecture:

- Reference latents appended as separate, non-rendered token blocks
- Exact overlap, `st_drc`, and temporal `strata` coordinate layouts
- Per-reference source-phase RoPE composition
- Clean reference timesteps and guide/grid-mask compatibility
- Reference-token trimming before model output/unpatchification
- Optional reference-CFG using a no-reference third forward pass

Because this package includes GPL-derived logic, the combined fork is
distributed under GPL-3.0. Changes are namespaced with `qrf_bfs_*` and/or
suffixed `_QrusheRFork`.
