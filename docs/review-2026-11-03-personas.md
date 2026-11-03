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

None of these is built. Each one is a decision:

1. **First-run persona.** The 2026-11-01 first-run work serves a first-time
   user, which none of P1 to P3 is. Should there be a fourth mode (first run
   or reinstall), or should it stay outside the design programme, owned by
   its [review](review-2026-11-01-first-run.md)? (S1-01)
2. **Choosing the watched job.** UX#7 (Current) promises it and D5 is rated
   Often, but the page always watches the running job. You can relabel that
   part of the story as Proposed, or approve a zero-height watch entry point
   on queued rows. (S-UX7-01)
3. **Ratings** for features built after the questionnaire: saved searches,
   Edit by instruction, Only the masked area, Then refine.
4. **Models' loader options** sit above the table by a recorded decision,
   while UX#2's candidate wording puts them second. You can keep them there,
   or collapse the held Compilation/Attention selects until a recipe is
   admitted. (prep S1-01)
5. **Smaller placement choices:**
   - Finish run's counts on the banner's text line, not just its tooltip (EXP-01).
   - Gallery beside and the sheet exports made quieter (C-06, C-07).
   - A cold guide summary for the row chosen in Models (prep S1-06).
   - Hiding the declined Automatic startup backup section (R-01).
6. **Cancel's width.** "Stopping…" is wider than "Cancel", so Pause, Skip
   and ↑ shift left while it shows. They already did this on your own
   press; now every watching tab shows it too. Giving Cancel a fixed width
   would shorten the Now line.
