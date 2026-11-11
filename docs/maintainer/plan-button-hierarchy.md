# Button hierarchy and contrast (UI#13)

**Status: done 2026-10-26** (`UI#13`).

## Evidence and scope

Matt reported similar dark-grey-on-black buttons on 2026-10-26. Browser
inspection confirms that footer actions resemble small muted labels even
when available. Static inspection also finds enabled hover rules winning
over held footer styles and selected small labels using accent purple at
about 4.26:1 on the panel. Ordinary muted text and primary white-on-purple
already meet ordinary-text contrast; the main issue is affordance and state.

Improve primary/secondary/tertiary hierarchy within existing tokens and
layout, without moving controls under the pointer or redesigning the app.
Primary Generate/Queue remains prominent. Enabled secondary actions get a
readable label, visible surface/edge and consistent hover. Footer actions
look pressable. Selected tools use readable text plus an accent edge/fill;
aria-pressed remains authoritative. Held controls keep their reason on
press, distinct treatment and visible keyboard focus; their hover cannot
imitate an enabled state. Danger actions retain their explicit semantics.

## Verification

Review the cascade and computed enabled, selected, held, hover and focus
states in a real local browser. Check normal 1400×900, narrow 600×900 and
wide 2560×1305 / 2048×1040 at 125% where relevant. Preserve dimensions and
stable geometry, the cyan focus ring, accessible names and reason-on-press.
Use selection/More seeds, wide layout and selected inpaint/bin checks as
appropriate. Synthetic fake data only, one browser job at a time. Obtain an
independent review and record exact checks before a scoped local commit.

## Status

Built and independently reviewed, 2026-10-26. Secondary actions share the
existing panel surface, readable text and strong border; tiny caption tools
stay quieter. Footer actions now keep that surface and edge. Selected tools
use readable text with accent fill/edge. Held hover cannot imitate enabled
hover, and the existing reason-on-press behavior and cyan keyboard focus
remain. The button-state changes preserve control dimensions.

`smoke-button-states` passed enabled/selected text contrast (at least 4.5:1),
primary/secondary distinction, held hover and reason-on-press, keyboard focus,
state-transition geometry and widths at 600x900, 1400x900, 2560x1305 and
2048x1040 at DPR 1.25. Current selection and inpaint smokes also passed. The
preceding retained UI pass cleared wide, bin and inpaint checks; its only
failure was a selection test's temporary cap racing status polling, fixed by
overriding `selectionCap` directly and rechecked successfully. Normal and
narrow/wide screenshots were visually inspected.

Evidence: `%TEMP%/local-diffusion-ux-more-buttons-final-20261026.log`,
`local-diffusion-ux-resume-browser-20261026.log`,
`local-diffusion-ux-caption-tabs-final-20261026.log` and the retained
`local-diffusion-buttons-20261026-a1/` screenshots/browser receipt.
More seeds workflow and shared CPU verification are recorded in
[GEN#23](plan-more-seeds-ux.md). These are synthetic browser checks; manual
screen-reader/touch acceptance remains `UI#7`.
