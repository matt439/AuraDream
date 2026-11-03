# Review 2026-11-03: personas and stories against today's app

Matt asked for a re-review of the personas
([research](design/research.md#mode-personas)) and user stories
([requirements](design/requirements.md)), then for the app to be aligned
with them. Eight read-only agents each took one area: the personas and
journeys, model setup, prompting, experiments and scope, curation, editing,
archives, and window sizes and accessibility. Each area's findings then went
to a skeptic who tried to refute them. Of 46 findings, 6 were refuted. The
rest were fixed on `design/persona-alignment`, or are listed under
[Open for Matt](#open-for-matt). No card run was made, and no task trial
(`DEV#1`). These checks show the app does what it claims. They do not show
that it is usable.

## What was out of date

The design set was written on 2026-10-18, about 500 commits ago.
requirements.md was reconciled on 2026-10-28; research.md never was.

- **research.md** still listed F02 (durable Finish run) and F04 (cache facts,
  Expand Stop) as open gaps. Both were built on 2026-10-18. Its Recover
  journey described phase 1 only.
- **Personas' current means** left out named setups, snippets, saved
  searches, Compare picked, Pick this run, Then refine, Skip and Move to
  front, Only the masked area, and Edit by instruction. **J08** and the
  **Edit** journey listed only Refine, Inpaint and Outpaint.
- **UX#12** said a sheet of the picks was built. It is deliberately not built
  ([design notes](design-notes.md#the-gallery-selection)).
- **docs/usage.md** never mentioned Edit by instruction, the Compilation and
  Attention selects, or the no-token preselect. It misplaced the cache badge
  and Notify paragraphs, and it understated where Space, Ctrl+A and Select
  work. It named the import password field by an old label.
- **design-notes.md** said instruction editing's stage 4 was on a local
  branch. It has been on main since PR #17.

All of these were corrected, and the design docs carry a dated 2026-11-03
reconciliation note. Goals, needs, non-goals and the candidate workbench
are unchanged.

## Fixed in the app

| Finding | Story | Change |
|---|---|---|
| S14-01 | UX#15 | More seeds and Swap tags previews name a queued row that is an inpaint of #N under its mask, an outpaint of #N with its padding, only the masked area, or an instruction edit (`submission_rules.pass_note`). Re-render has no per-row preview, and it refuses an edit or a resized inpaint. |
| S14-02 | UX#15 | When Skip refine passes is unticked, a Mass refine row notes that it is refined whole and its mask is not used, or that an edit is refined as an ordinary image. |
| S-UX7-02 | UX#7 | `cancel_requested` is in the job snapshot, and a cancel publishes a frame at once. Every tab watching the job shows Cancel as **Stopping…**, held, and a press says why. |
| EXP-02 | UX#11 | The picked-card menu offers the head's exact-size Re-render. |
| C-02 | UX#11, UX#12 | Pick this run's title names its scope (every checkpoint, whatever the search or Starred shows) and the cap. The Select title's Ctrl+A clause gives the cap. |
| S1-02 | UX#20 | Orientation on a square aspect is `aria-disabled` with its reason, and a press says why (it was `disabled`). |
| S1-04 | L8 | The keyboard help names Shift+click alone. |
| EXP-03 | P1, C7 | The Grid fold reads "prompt variables & sweeps", not "overnight sweeps", and so does the CLI's help. |
| S14-07 | UX#14 | The Outpaint Strength tip says "inward of the band". |

Checks: ruff and the Pyright gate are clean, `pytest -n auto` gives 8353
passed and 8 skipped, and every smoke was run with `run_smoke.py --all`.
smoke.json, smoke-jobs, smoke-context-menu and smoke-animagine gained
steps for the changes.

## Refuted

- EXP-04: Never-rated grid axes are optional fields in a folded section, so
  they are already secondary.
- C-03: UX#12 is Proposed on purpose, because its criteria are trial
  outcomes.
- R-03: Delete unstarred's by-hand recovery is a recorded decision.
- R-04: the bundle scope radio is visible when the dialog reopens.
- R-05: the subset hints already say that links can be left out.
- R-06: E11's line numbers are dated by design.

## Open for Matt

On 2026-11-03 Matt took the recommended answer to every question, then
gave the ratings:

1. **First-run persona (S1-01): no fourth mode.** The first-run work stays
   outside the design programme, owned by its
   [review](review-2026-11-01-first-run.md).
2. **Choosing the watched job (S-UX7-01): relabelled.** That part of UX#7 is
   Proposed, not Current; no watch entry point on queued rows is planned.
3. **Ratings: all Sometimes.** Saved searches, Edit by instruction, Only the
   masked area and Then refine
   ([later features](design/preferences.md#later-features-2026-11-03)).
4. **Models' loader options (prep S1-01): collapsed.** Compilation and
   Attention sit in a folded Performance disclosure until a recipe is
   admitted. The fold opens itself as the picker opens, never on a refresh,
   when a choice is admitted or the draft or saved choice is not the
   default.
5. **Smaller placements:**
   - Finish run's counts on the banner (EXP-01): built, on a line of their
     own under the banner's text.
   - The automatic startup backup section (R-01): hidden by default. It
     shows when the gallery already has something saved for it, or on a
     page opened with `?automatic-backup`.
   - Gallery beside and quieter sheet exports (C-06, C-07), and a cold guide
     summary in Models (prep S1-06): not taken.
6. **Cancel's width: left as it is.** A fixed width would shorten the Now
   line every time, to stop a brief shift while it reads Stopping….
