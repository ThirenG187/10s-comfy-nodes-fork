# Upstream notice

This experimental custom node is a targeted fork of the identity-reference
mechanism in **TenStrip/10S-Comfy-nodes**, retrieved from the upstream `main`
branch on 28 July 2026.

Upstream project: `https://github.com/TenStrip/10S-Comfy-nodes`

The upstream README identifies the project as MIT licensed. The following
mechanisms were adapted:

- LTX reference-token injection and modulation-extension patches
- Best-Face-ID RoPE source-phase composition
- YuNet / MediaPipe / Haar face-detection fallbacks
- Reference face alignment, VAE encoding, and spatial gating

Changes in this fork are namespaced with `qrf_` and/or suffixed
`_QrusheRFork`, and add independent subject ranges, target-face assignment,
and up to four identities.
