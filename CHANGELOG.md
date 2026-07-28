# Changelog

## 0.2.0-qrf-bfs

- Replaced the approximate prefix reference patch with a BFS-style appended
  reference-token implementation.
- Added overlap, ST-DRC, and strata layouts.
- Added exact per-block source-phase ranges for fused and legacy RoPE formats.
- Added clean reference timesteps and grid-mask handling.
- Added optional target-image reference injection for guided T2V.
- Added reference-CFG.
- Retained four-subject target-face assignment and spatial gating.
- Changed the combined package license to GPL-3.0 due to BFS-derived logic.
