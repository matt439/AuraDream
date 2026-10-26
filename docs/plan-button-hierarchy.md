# Button hierarchy and contrast (UI#13)

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

Static audit and before screenshots complete; implementation not started.
The size/crop batch is being qualified first. More seeds workflow changes
have their separate owner in [GEN#23](plan-more-seeds-ux.md).
