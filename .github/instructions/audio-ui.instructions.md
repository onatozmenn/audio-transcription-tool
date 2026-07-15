---
description: "Use when creating or modifying the audio transcription app's frontend, visual design, components, states, or responsive behavior."
applyTo: "app/**/*.tsx,app/**/*.css,components/**/*.tsx"
---

# Audio transcription UI direction

Adapted for this product from <https://pols.dev/slop.md>. Apply these as design judgment, not as a mechanical ban list. Explicit user direction wins.

## Product character

- This is an operational transcription tool. Open on the usable language, upload, progress, transcript, and export workflow, not a marketing hero.
- Build the identity around one specific idea: an audio waveform resolving into transcript lines. Extend that mark or motion when a signature visual is needed; do not add generic SaaS artwork.
- Keep the interface quiet enough for long transcripts and repeated use. Prefer dense, aligned information over decorative card stacks.
- Hold a disciplined palette: warm green-black canvas, paper-white text, muted mint for ready states, cyan only for active processing, and amber/red only for warnings or errors.

## Avoid template tells

- Do not use a full-page grid, radial glow blobs, blue-purple gradients, floating cards, fake app windows, or atmospheric decoration unrelated to audio and text.
- Do not put icons or logos in colored rounded tiles. Use familiar control icons bare; use the custom waveform-to-text mark for identity and real brand SVGs for external services.
- Do not turn labels and metadata into pills by default. Reserve contained chips for a status that genuinely needs a boundary.
- Do not outline every surface or add broad all-around shadows. Separate work areas with tonal elevation and spacing; add a border only when it communicates an affordance or necessary boundary.
- Keep radii at 8px or less. Buttons must not jump, scale, or lift on hover; use a quiet tonal state change.
- Avoid default marketing compositions such as eyebrow-heading-copy-CTA stacks, filled-plus-outline button pairs, feature-card rows, and pre-footer CTA slabs.
- Keep the neutral system type stack unless a licensed or self-hosted face is chosen for a product-specific reason. Do not add a trendy web font merely to look designed.

## Interaction and craft

- Every visible control must work with pointer and keyboard input. Preserve focus visibility, labels, dialog semantics, disabled explanations, progress announcements, and reduced-motion behavior.
- Content is visible by default. Never depend on an entrance animation to reveal text or controls.
- Use motion only to explain live audio processing, progress, completion, or a direct interaction. Avoid decorative looping motion.
- Keep transcript text selectable and readable. Use tabular numerals for timestamps, durations, percentages, and counts.
- Check long English and Turkish strings. No text may clip, overlap, touch an edge, or cause horizontal overflow.

## Before finishing

- Run the focused lint, type, and test checks for touched files.
- Inspect the rendered app at 390x844 and 1440x900. Check overflow, alignment, text fit, focus states, open menus, empty/loading/ready/error states, and the export dialog.
- Click every control affected by the change. A polished static screenshot does not compensate for dead or broken interaction.