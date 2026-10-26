# Control sizing and Bundles review - 2026-10-26

Matt asked to review control size against importance and frequency, and the
Bundles password/count flow. This is a bounded review of the existing app,
not a new layout proposal. Common tasks are inferred from the composer and
result workflow, supported by Matt's explicitly reported preferences in
`docs/design/preferences.md`; no usage analytics or task timings are invented.
The actual in-app browser starts at 1275x712 on an isolated fake gallery on
port 9923. Only synthetic images and bundles are used.

## P1: the import password's scope is unclear (`UI#14`)

The password field is under **Import a file**, but the same field also gates
every Protected saved-bundle Import button. The Saved bundles section is
separated by upload controls, operation feedback and another section boundary.
In the observed dialog its Import button was roughly 300 px below the shared
password, with no adjacent label connecting them. A synthetic backup and a
repeated saved import verified that these are existing operations, not separate
password stores. Downloads remain encrypted and do not require the password.

Keep one import password and its existing retry lifetime, but group file and
saved imports together and explicitly associate Protected import actions with
that field. The new-bundle password must clearly apply to both backup and
bundle-and-clear, separately from unlocking an existing bundle. No secret is
logged, persisted or copied into a URL.

Built: creation and import passwords are explicitly labelled by purpose.
Selected-file and saved imports share one Import a bundle section; Protected
rows visibly name and link to the Import password. A held import needing only
that password focuses and reveals the shared field. No per-row secret was
introduced. Feedback stays near file/saved actions, with unchanged retry,
success and close clearing. Independent review is clear.

## P2: the required clear confirmation is understated (`UI#15`)

The actual dialog shows **Type 2 to confirm all gallery records will be
removed** as a small muted label beside a blank field and held clear button.
It does not explain upfront why typing is required or distinguish this gate
from creating a backup. The synthetic backup succeeded with the count blank.

The exact contract is the entire gallery's database-record count, including
all models and the recycle bin, checked again under the server reservation.
It is not an export quantity or the full-size-file count. Only destructive
**Bundle & clear** requires it; backups and imports do not. Explain the safety
reason, exact required value and other enabling conditions beside this group,
while retaining the backend check and verified archive before clearing.

Built: a separate clear group explains accidental-clear prevention, scope,
the exact count, verification before removal and when the action is enabled.
Back up is primary and explicitly needs no typed confirmation. A clear press
held only for its count focuses that field and reveals it if clipped inside
the dialog. The 360 px keyboard case verifies a wrapped count above the
scrollport becomes visible without submitting or moving the page behind it.
Independent review is clear; backend count/encryption/verification code did
not change.

## P2: frequent secondary actions have undersized targets (`UI#16`)

At 1275x712, the prompt's Examples/Check/Reset controls measure 19 px tall
with 10 px type, while Models and Unload are 37 px. Gallery selection and
batch actions measure about 20 px tall. Generate is already a distinct 43 px
primary action. The existing direct result More seeds and image actions are
discoverable, and fixed-seed entry retains space for the full value.

Evaluate modest target/readability increases for prompt and gallery tools,
using existing tokens and preserving their secondary visual weight. Keep
Generate prominent; avoid global scaling, hiding infrequent settings or
moving entire sections without the separate layout trials. Check compact,
narrow and wide viewports, keyboard focus and stable geometry across states.

Built: prompt tools below 1800 px are 24 px tall with 11 px type; the existing
wide caption sizes remain. Gallery actions use the small-button height token
(26 px normally, 28 px on the short wide tier), with 24 px collapsed
Picked/Clear widths. Generate stays 43 px at the normal tier, with its primary
fill; seed width and section order remain. All catalogue caption combinations
fit in the sizing check. Crowded 600 px selection/bin controls, hover/held/
selected geometry, keyboard focus and hit targets pass. Independent review
is clear.

## Qualification boundaries

All operations use synthetic CPU/fake-server data. This review cannot establish
GPU/device acceptance, image quality or real hardware timing. Existing click,
mixed-prompt and More seeds behavior remains part of regression coverage.

The focused backend safety/crypto/HTTP suite passes **155 tests**, and all six
documentation-map checks, Ruff and JavaScript syntax checks pass. Browser
coverage includes the extended Bundles flow (missing/wrong counts and
passwords, retries, cancellation, repeated imports, secret clearing, keyboard,
360/390/600/wide layouts and forced colours) and the new control-sizing smoke
(1400x900, 600x900, 1536x864 and 2048x1040, the last two at DPR 1.25).
All **13 relevant browser smokes pass** across their final runs: Bundles,
control sizing, wide, narrow, tags, grid, selection, More seeds, bin, snippets,
button states, More seeds tabs and mixed prompts. This closes the bounded
review; larger layout changes still belong to the separate task trials.

Two existing regression fixtures needed explicit pointer/click setup after the
sizing change. Re-render was intentionally deferred under a resting header
pointer; its fixture now leaves the header, proves the exact pick/scope/label
and checks visibility before clicking. Bin's unguarded Select button was
measured at y=919.5-945.5 in a 900 px viewport; its miss caused downstream
restore failures. That fixture now proves Select is hit and active before
clicking cards, retaining exact IDs, counts and restore outcomes. No product
selection or bin semantics changed.
