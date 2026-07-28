# QrusheRFork Multi-Face Test Image Pack

All people in this pack are fictional, AI-generated adults.

## Subject mapping

Use these references in this order when `assignment_mode` is `left_to_right`:

1. Subject 1: `ref_1_man_a.jpg`
   - Optional secondary view: `ref_1_man_a_alt.jpg`
2. Subject 2: `ref_2_woman_a.jpg`
   - Optional secondary view: `ref_2_woman_a_alt.jpg`
3. Subject 3: `ref_3_man_b.jpg`
   - Optional secondary view: `ref_3_man_b_alt.jpg`
4. Subject 4: `ref_4_woman_b.jpg`
   - Optional secondary view: `ref_4_woman_b_alt.jpg`

## Suggested tests

### Two people
- Subject 1: `ref_1_man_a.jpg`
- Subject 2: `ref_2_woman_a.jpg`
- Target: `target_2people.jpg`

### Three people
- Subject 1: `ref_1_man_a.jpg`
- Subject 2: `ref_2_woman_a.jpg`
- Subject 3: `ref_3_man_b.jpg`
- Target: `target_3people.jpg`

### Four people
Use all four subjects in numerical order with any `target_4people_*.jpg`,
`target_cafe_4people.jpg`, or `target_outdoor_4people.jpg`.

## Recommended initial node settings

- assignment_mode: `left_to_right`
- identity_strength: `1.0`
- subject_2_strength: `1.0`
- subject_3_strength: `1.0`
- subject_4_strength: `1.0`
- spatial_gating: `mask_soft`
- source_id_base: `2.0`
- source_id_stride: `1.0`
- background_reference_strength: `0.02`
- debug: `true`

Connect the chosen target image to both:
1. The QrusheRFork node's `target_image`
2. The workflow input used as the LTX first frame

The crops were upscaled from a generated contact sheet. They are intended for functional
experimentation with assignment, masking, and identity separation—not as a benchmark
for maximum image quality.
