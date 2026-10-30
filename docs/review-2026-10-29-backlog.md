# Review: backlog from the 2026-10-29 TODO audit

**Status:** written 2026-10-29 for the TODO audit's additions that need no
plan of their own: `PA#12`, `UI#32`, `UI#33`, `GEN#33`, `GEN#34`, `GAL#23`,
`GEN#36`, `GEN#38` and `UI#38`. `PA#12`, `UI#32`, `UI#33`, `GEN#33`,
`GEN#34`, `GEN#36` and `UI#38` are built (2026-10-30, below); the other
two are OPEN. Matt
approved adding every audit item on 2026-10-29. Each section below covers
one row: what is missing (preference ids from `docs/design/preferences.md`,
file:line checked against the source on 2026-10-29), the design and the
rules it must keep, non-goals, and the stages with the checks that close
each one. None of these rows needs the card.

The audit's draft row texts were checked against the source, and four
claims were corrected here:

- `PA#12`: an accepted suggestion is a caret-range insert through
  `execCommand('insertText')`, not a whole-box `replaceBoxText` write.
- `UI#33`: the hold on a lightbox over a sheet already has its words
  (`SELECT_ON_SHEET`); it is tested by `sheetMode`, since a cell click
  opens the lightbox with no origin.
- `GEN#36`: Versions should count seeds up as the single Refine dialog does,
  not step over taken seeds as More seeds does.
- `GEN#38`: gallery rows record neither the offload mode nor whether an
  image was a process's first, so neither can be filtered out row by row.

## PA#12 Tag suggestions in the negative prompt

**Status:** done 2026-10-30, both stages, on `feat/todo-backlog-2026-10-29`.
Built as designed, through the Refine dialog's editor list rather than the
prompt box's own functions:

- Server: `GET /api/tags/suggest` takes `field` (`prompt`, the default, or
  `negative`; anything else is a 422). With `negative`, `rank_suggestions`
  gets an empty `avoid`; no `field` gives today's answer byte for byte.
  Tests: `test_the_negative_box_ranks_the_recommended_negative_as_any_other_tag`
  (`tests/test_tags.py`) and
  `test_the_negative_box_asks_for_suggestions_without_the_demotion`
  (`tests/test_api_analysis_tags.py`).
- Page: `#negative` sits in a `.neg-editor` with
  `#negative-tag-suggestions`, named "Negative prompt tag suggestions" so it
  is told apart from the prompt's in the same panel. The design's
  "parameterised by the box rather than copied" was met by the editor list
  the Refine dialog's prompt already uses (`scheduleEditorSuggestions` with
  a new `field` option, `renderEditorSuggestions`, `currentTagRange(el)`,
  `preselects`), rather than by parameterising `requestTagSuggestions` and
  `applyTagSuggestion`, which carry the prompt's chips, typing state, prose
  guard and Find tag row, none of which the negative takes. The Refine
  dialog's keydown became the shared `editorListKeydown`. An accept is
  `applyNegativeSuggestion`'s `insertText` (fallback `setRangeText` with an
  `input`), whose `input` runs the box's own listeners (`syncAnalysisStale`,
  the preview, `persistPanel`). `syncCfg` closes a list up as the box greys;
  `writeBox` closes it on a whole-box write; a model switch and a non-Danbooru
  checkpoint close it. The shared lookup timer is touched only while the
  negative's list is up (`hideNegativeSuggestions`), so a blur or a greyed
  box never drops the chip editor's or the Refine dialog's lookup. Taken
  while Matt was away (2026-10-29), revisit if wanted: this route over
  parameterising the prompt box's functions, and the list's own name.
- Checks: `smoke-tags` (7878) gained the steps listed in stage 2, all
  passing, with the layout probe (negative field and Generate unmoved with
  the list open and after Tab).

**Evidence.** A2 (use a negative prompt) and B3 (use tag autocomplete) are
both Often (`preferences.md:27`, `:47`). On the tag checkpoints, the
recommended negative prompt is itself a Danbooru tag list (Animagine,
Illustrious, NoobAI and the others), but suggestions are offered only in the
prompt box:

- `requestTagSuggestions` returns unless `document.activeElement ===
  els.prompt` (`web/tags.js:440-474`).
- The listeners are on `els.prompt` alone (`web/app.js:115-148`).
- The one list, `#tag-suggestions`, is controlled by `#prompt`
  (`web/index.html:163-173`). `#negative` has no `aria-controls`
  (`web/index.html:341`).

On the server, `rank_suggestions` never puts first a tag of the guide's
recommended negative prompt unless it is typed whole
(`backend/tags.py:171-216`). The route always
passes that set (`tag_lookup.negative_tags`, `backend/tag_lookup.py:61-68`;
`backend/api_prompt.py:205-221`). In the negative box that demotion is
backwards: the guide's negative tags are the ones wanted there.

**Design.**

- The negative box gets its own list: a second
  `ul.tag-suggestions[role=listbox]`, positioned absolutely under the box as
  the prompt's is (`web/styles/fields.css:439-441`). It hangs over the
  fields below and takes no room. `#negative` gets the same combobox
  attributes as `#prompt`.
- Caret range, preselection and keys are the prompt box's own functions,
  parameterised by the box rather than copied:
  - `currentTagRange(el)` already takes the element (`tags.js:317`).
  - `preselects` selects the first item only from three typed characters
    that it starts with.
  - Tab and Enter accept, ArrowDown selects, and Escape closes the list
    alone.
- An accept replaces the item at the caret through
  `execCommand('insertText')`, falling back to `setRangeText`, exactly as
  `applyTagSuggestion` does (`tags.js:376-398`). One Ctrl+Z gives back what
  was typed. This is a caret-range edit, so the whole-box rule (`writeBox` /
  `replaceBoxText`) does not apply. Any whole-box write that does reach the
  negative must still use those functions and close the list, as `writeBox`
  closes the prompt's (`tags.js:83-89`).
- The request names its box: `GET /api/tags/suggest?q=...&field=negative`.
  The server keeps deciding the order. With `field=negative`,
  `rank_suggestions` gets an empty `avoid` set and the order is otherwise
  unchanged. Guide tags are not promoted, because no author sentence asks
  for that. Omitting `field` keeps today's answer byte for byte.
- A greyed negative is `disabled` (`web/sizes.js:305-311`), so it never
  takes focus and never opens a list. A box that greys while its list is
  open closes the list in `syncCfg`.
- The Find tag row (`renderTagSuggestions`, `tags.js:403-438`) is left off
  the negative's list. The finder writes into the prompt
  (`openTagFinderFor`).
- `tagVocabularyEnabled` (Danbooru checkpoints only, `tag-chips.js:567`)
  gates the negative as it gates the prompt.
- What today's functions hard-wire to the prompt is taken from the box:
  `requestTagSuggestions` compares `els.prompt` before and after the answer,
  `caretTagIndex` reads `els.prompt` (`tags.js:305-313`), and
  `applyTagSuggestion` focuses `els.prompt` and then runs the prompt's
  follow-ups (`promptReplaced`, `syncTagSelection`, `schedulePreview`,
  `scheduleTagValidation`). On the negative an accept runs only what the
  negative's own `input` listener runs (`syncAnalysisStale`,
  `web/app.js:124`) and `persistPanel`. The hybrid-prompt prose guard
  (`currentPromptParts`, `tags.js:235-238`) reads only the prompt's
  classified layout; the negative has none, so its requests skip that guard.

**Rules that apply.** Suggestions add no height: the list floats, and
nothing moves under 1800 px. Every suggestion list is a named listbox with
`aria-activedescendant` (design-notes, Headings and landmarks). An
in-flight answer must not reopen a dismissed list (`tagSuggestionEpoch`).
Only tag names reach Danbooru (design-notes, Animagine).

**Non-goals.**

- Chips, validation colouring or a Tags helper for the negative.
- Find tag in the negative.
- Suggestions on checkpoints without the Danbooru vocabulary.
- Reordering the user's negative text.

**Stages and completion criteria.**

1. Server flag. A pure `tests/test_tags.py` case: with the flag, a guide
   negative tag can come first for a short query. Without it, the ranking
   is unchanged. A route test (`tests/test_api_analysis_tags.py`) shows the
   flag reaching `rank_suggestions`, and an absent flag gives the same
   answer as today.
2. The page. New `smoke-tags` steps on the 7878 Animagine fake:
   - typing in the negative opens its list, and the prompt's list stays
     closed;
   - Tab accepts a preselected item, and one Ctrl+Z restores the typed
     text;
   - Escape closes only the list;
   - with the negative greyed (CFG under the threshold), the box cannot be
     focused and no list appears;
   - the prompt box's existing suggestion steps still pass.

   A layout probe shows the negative field and Generate at the same
   coordinates with the list open and closed.

## UI#32 Contact-sheet cell menu and Delete

**Status:** done 2026-10-30, both stages, on `feat/todo-backlog-2026-10-29`.
The file:line claims below still held; built as designed:

- `contextTarget` gained the `sheet` kind (`#sheet-grid .sheet-cell[data-id]`,
  live records only), its record from the new `sheetCellRecord(cell)` in
  `sheet-render.js`, which `sheetKey` and `openMoreSeedsFromKey` now share.
  `imageItems` takes the cell as its own place: Star is the cell's `.star`;
  Reuse settings, Refine…, Inpaint… and Move to recycle bin are `syncedItem`s
  over `syncReuseButton`, `syncRefineTool`, `syncInpaintTool` and
  `syncBinWriteButton`, running `reuse`, the new `refineFromCard` (the card's
  ⤢ closure moved into `gallery.js` so both run it), `inpaintFromCard` and
  `deleteImage`. Open clicks the cell's image, so the lightbox walks the
  sheet as a cell click does. Show run as sheet is held with
  `SHEET_SHOWS_RUN` ("The sheet shows this run already") where
  `sheetShownRun()` is the record's run, and chosen says it under the
  gallery's head and on the cell ("Shown already"). The bin item carries
  Del, since the cell now has the key.
- Delete in `sheetKey` (no Ctrl or Shift, a repeat ignored) runs
  `deleteImage`; `forgetImages`' existing `sheetFocusPlace` and
  `restoreSheetFocus` put focus on the cell now in that place, or the last.
  The cell image's `aria-keyshortcuts` and the help's Contact sheet group
  list Delete; the Menu/Shift+F10 help row names the contact-sheet cell in
  place of "Contact-sheet cells keep the browser's menu".
- `cardNote` on a cell had no styling (More seeds' refusal already put one
  there): `.sheet-cell > .card-note` joins the card's and the tile's rules.
- Checks: `smoke-context-menu` (7908) and `smoke-sheet` (7880) gained the
  stage steps below, all passing; `smoke-shortcuts` follows the new help
  words. `smoke-sheet-zoom`, `smoke-gallery`, `smoke-refine`,
  `smoke-refine-unavailable`, `smoke-more-seeds-tabs`,
  `smoke-shortcut-dispatch` and `smoke-copy-feedback-ownership` still pass.
  The `.empty` case is checked on a hole the smoke adds, since the 3-seed
  run it opens is wrapped and has none. Taken while Matt was away
  (2026-10-29), revisit if wanted: the held item's words and its
  "Shown already" gist.

**Evidence.** F9 (view a run or search as a contact sheet), L7 (right-click
menus) and J4 (bin individual images) are all Often (`preferences.md:120`,
`:208`, `:172`). The sheet has neither a menu nor a delete:

- `contextTarget` knows three hosts: the lightbox stage, `#gallery
  .card[data-id]` and `#viewer figure[data-id]` (`web/context-menu.js:18-37`).
  A right-click on a `.sheet-cell` gets the browser's menu.
- A sheet cell carries an image and a star button, nothing else
  (`web/sheet-render.js:435-484`). `sheetKey` handles F, the zoom keys and
  the arrows (`sheet-render.js:700-723`). There is no Delete.
- Binning from a sheet today means opening the lightbox and pressing Delete
  (`web/app.js:724-730`).
- The sheet group in the shortcut help has no Delete row
  (`web/shortcuts.js:59-72`).

**Design.**

- `contextTarget` gains a fourth kind, `sheet`, for
  `#sheet-grid .sheet-cell[data-id]` that is not `.empty`. The record is
  taken from `sheetRecords`, or from `sheetLayout.source` for a source cell,
  as `sheetKey` finds it (`sheet-render.js:706-708`).
- The menu shows `imageItems`, never `pickedItems`. The sheet's cells are
  not picked (`SELECT_ON_SHEET`, `web/gallery-select.js:89`). Each item
  keeps the menu's rule: it presses the place's own button, or where the
  place has none, uses `syncedItem` with the card button's own sync
  function. Logic is never copied.
  - Star: the cell's `.star`.
  - Reuse settings, Refine…, Inpaint… and Move to recycle bin: `syncedItem`
    over `syncReuseButton` (`web/gallery.js:805`), `syncRefineTool`
    (`gallery.js:545`), `syncInpaintTool` (`web/inpaint.js:138`) and
    `syncBinWriteButton` (`web/bin.js:28`), running what the card's buttons
    run (`gallery.js:440-458`). ❐, Outpaint…, Edit by instruction… and More
    seeds already use `syncedItem` (`context-menu.js:148-158`).
  - Open: the cell's own open, `openLightbox` over the sheet's walk, as a
    cell click does through `thumb` (`sheet-render.js:425`, `:445`;
    `web/viewer.js:656`). A cell click passes no origin today; only the
    swipe ladder opens with `'sheet'` (`sheet-render.js:255`).
  - Show run as sheet: kept. Where the sheet already shows that run
    (`sheetShownRun`), it is held with that reason.
- Delete on a focused cell (its image or its star; no Shift, no repeat)
  calls `deleteImage(rec)`, the ⌦ path: into the bin at once, no question,
  and the Restore note from `noteBinned`. `forgetImages` already records the
  sheet's focus place before the redraw (`sheetFocusPlace`,
  `web/gallery-actions.js:498-500`, `:638`). The redraw puts focus on the
  cell now in that place, or the nearest populated one, with
  `preventScroll`.
- The cell image's `aria-keyshortcuts` (`sheet-render.js:461-462`) and the
  sheet help group gain Delete.

**Rules that apply.**

- Held items are `aria-disabled` and say why when chosen.
- The per-image delete never confirms.
- `forgetImages` is the one place the page drops binned ids.
- Copy feedback belongs to the latest press on its own host. The cell is
  the host, through the existing `host.closest('[data-id]')` check.
- The menu closes on scroll (`contextMenuAway`), so it never floats over a
  scrolled sheet.

When the stage lands, `docs/rules.md`'s right-click paragraph names the
sheet cell among the hosts.

**Non-goals.**

- Picks on the sheet.
- A Delete key on gallery cards, which keep ⌦.
- Delete permanently from a sheet: sheets show live images only.
- Bin-mode sheets.

**Stages and completion criteria.**

1. Menu. New `smoke-context-menu` steps (7908) that open the menu on a sheet
   cell by right-click and by Shift+F10:
   - the item list matches a card's, minus Open's card wording;
   - a held item (Show run as sheet on a cell of the run the sheet shows)
     says its reason when chosen;
   - Copy seed reports on the cell;
   - an `.empty` cell keeps the browser's menu.
2. Delete. `smoke-sheet` steps:
   - Delete on a focused cell bins it, and the Restore note appears;
   - focus lands on the cell now in that place, and on the last cell when
     the deleted one was last;
   - Restore brings the cell back;
   - holding the key bins one image;
   - the help dialog lists the key.

   `smoke-sheet-zoom` still passes.

## UI#33 Pick from the lightbox

**Status:** done 2026-10-30, stage 1, on `feat/todo-backlog-2026-10-29`.
The file:line claims below were re-checked; the lightbox's keys are now at
`web/app.js:729-739` and its menu at `web/context-menu.js:135-201`; the others held before this change.
Built as designed:

- The key is X, the proposal: taken while Matt was away (2026-10-29),
  revisit if wanted. It sits with F/R/S/Delete in app.js's lightbox keydown,
  so it keeps their guards (no Shift, no repeat, never in a field, a menu or
  with a command modifier).
- `pickFromLightbox(rec)` (`gallery-select.js`) toggles through
  `toggleSelected(rec, els.lbError)`; `toggleSelected` and `refuseOverCap`
  gained the `slot` a cap refusal is said in (the gallery's by default, not
  mirrored when it is the lightbox's). It says "#N picked (3 picked)." or
  "#N unpicked (2 picked).", with " Compare picked has ended." where the
  unpicked image was a member of `lbPickedPair`; the pair is emptied and a
  picked comparison ended with `setCompare(null)`, the image staying shown.
- `lightboxPickHeld(rec)` answers `BIN_HELD` in the bin or on a binned
  record, then `SELECT_ON_SHEET` while `sheetMode`; the key says it in the
  lightbox's line and `lightboxPickTitle` is the menu item's title. The item,
  "Pick #N" / "Unpick #N" after Star with the X hint (so its
  `aria-keyshortcuts` is X), is spread into `imageItems` for the lightbox
  only, since a falsy entry there is a group's line.
- The help's Lightbox group has an X row (`web/shortcuts.js`).
- `smoke-lightbox` and `smoke-select` gained the steps listed below; no
  pytest was added, since nothing in the backend changed (the classic-script
  guard covers the edited scripts).

**Evidence.** E12 (pick images or ranges across filters) and F1 (open
full-size images and browse between them) are Often (`preferences.md:106`,
`:112`). Picks are made only on gallery cards:

- `toggleSelected` is called from the card click and Space handlers alone
  (`web/gallery-select.js:179`, `:188`, `:216-222`).
- The lightbox's keys are F, R, S, Delete, M, the zoom keys and the arrows
  (`web/app.js:706-735`), and its menu has no pick item
  (`web/context-menu.js:126-174`).

So judging keepers at full size means closing the lightbox to pick each one.

**Design.**

- A lightbox menu item, "Pick #N" or "Unpick #N", calls
  `toggleSelected(rec)`. The cap is respected through `refuseOverCap`
  (`gallery-select.js:251-258`), which today always writes to
  `els.galleryError`; it gains the slot to write to, so a refusal in the
  lightbox goes to the lightbox's line (`els.lbError`) rather than the
  hidden gallery's.
- A key does the same. It is not P: unmodified P is the workspace route
  back to the prompt, and that plan keeps it free of any lightbox meaning
  ([plan-shortcut-workflow.md](plan-shortcut-workflow.md#stage-4-prompt-route)). The proposal is X, which no
  page handler uses today; Matt may name another key before stage 1. It
  takes the same guards as F/R/S: no Shift, no repeat, never in a field or
  a menu.
- What the pick did is said in `els.lbError`, for example "#12 picked (3
  picked)". The lightbox toolbar gets no new button, so no control moves.
  On close, the count beside Select shows the change through
  `syncSelection`.
- Held, with the reason as the item's title and said on the key:
  - in the recycle bin, or on a binned record (`BIN_HELD`);
  - in a lightbox over a shown sheet (`sheetMode`), with `SELECT_ON_SHEET`,
    since the sheet's cells are not picked. The test is `sheetMode`, not
    `lbOrigin === 'sheet'`: a cell click opens with no origin, and only the
    swipe ladder passes `'sheet'` (`sheet-render.js:255`,
    `web/viewer.js:656`).

  Viewer and gallery lightboxes may pick: picks hold across filters.
- Compare picked: `lbPickedPair` (`web/lightbox-compare.js:10`) is a
  snapshot of two picks (`gallery-select.js:374-381`). Unpicking either member clears it and ends
  the picked comparison, as a list change does (`web/lightbox.js:15-18`).
  Picking a third image leaves the pair alone.

**Rules that apply.**

- Hold through `aria-disabled` with the reason, and a held press says why.
- `syncSelection` derives every mark from the state.
- The cap is `selectionCap()`.
- A shortcut is listed in the help's Lightbox group and in
  `aria-keyshortcuts`.
- Letter keys never replace typing or modified browser shortcuts.

**Non-goals.**

- Range picks from the lightbox.
- A visible pick badge on the lightbox picture.
- Picks on sheet cells.
- Changing Space or P.

**Stages and completion criteria.**

1. Item and key. `smoke-lightbox` (7884) steps:
   - pick and unpick by item and by key;
   - the said line;
   - after close, the card is `.selected` and the count is right;
   - a key press inside a field does nothing.

   `smoke-select` (7898) steps:
   - the cap refusal in the lightbox line;
   - the bin hold, and the sheet-origin hold with `SELECT_ON_SHEET`;
   - Compare picked, then unpicking one member, ends the pair and keeps the
     view;
   - the help dialog lists the key.

## GEN#33 Re-render at a tier

**Status:** done 2026-10-30, all three stages, on
`feat/todo-backlog-2026-10-29`. The file:line claims below were re-checked:
each held in substance, though rows built earlier in the run had moved the
page's Re-render call and `api_generate.py`'s refusals down by some 20-25
lines. Built as designed, with these choices taken while Matt was away
(2026-10-29), revisit if wanted:

- `sizes.nearest_preset` measures a size against each aspect's *named*
  ratio (16/9 for "16:9"), in log terms, not against the tier's entries:
  rounded to the 16 grid, 9:16 at 1.5 MP (928x1664) sits farther from 9:16 at
  1 MP (768x1360) than 4:7 at 1.5 MP does, so entry to entry sent a 1 MP 9:16
  to 4:7. `sizes.ASPECT_TOLERANCE` is 0.02, about twice the table's own
  worst rounding (0.96%, 2:3 and 3:2 at 4 MP); 1000x700 is 4.9% off 3:2.
  `tests/test_sizes.py` holds both, and every preset mapping to its own
  aspect at every tier.
- Any refusal in any size group refuses the whole press, budget included
  (the size past it named, as Re-render always said it): nothing is queued
  in part. All unmatched is a 422 naming each id and size.
- Tier mode takes picked cards only (`ids`); a filter with a tier is a 422,
  as the non-goal says.
- `expected` is accepted beside `width`/`height` too and checked when sent;
  the page sends it only from the tier question.
- The Picked item sits between Compare picked and Seeds to grid. The
  question is `askConfirm` with a "Re-render" OK; `openAsk` gained
  `destroy` (default true) so that OK is not painted red as Delete's is.
- As in Swap tags, the jobs are submitted one after another under
  `rerun_admission`. A later submit failing after an earlier one was admitted
  is answered as an error, without More seeds' partial receipt: every
  refusal that can be checked ahead is checked before the first submit.

Tests: `tests/test_sizes.py` (stage 1) and `tests/test_api_rerender_tier.py`
(stage 2: two aspects in two jobs and `more`, one size's budget, unmatched
and all unmatched, the preview queueing nothing, `expected`, the request's
either/or, an inpaint off its size, and the run id without `tier` equal to
the one-size formula). Stage 3: `smoke-select`'s last steps; the existing
head-layout steps pass, with the menu's order and the Picked title updated
for the new item.

**Evidence.** G7 (re-render at another size), A3 (aspect presets) and A4
(resolution tiers) are Often (`preferences.md:135`, `:28-29`). Today
`RerenderRequest` takes one `width` and `height` for every record
(`backend/schemas_rerun.py:15-46`). The page sends the panel's
`els.width`/`els.height` (`web/generation.js:582-599`).

That makes "these keepers at 2 MP" impossible for a mixed-aspect selection.
A 3:4 portrait and a 16:9 landscape both come out at the panel's one
shape. The inpaint refusal also bites, since an inpaint must stay at its
source's size (`api_generate.py:536-544`).

Swap tags already splits a press into one job per size and answers the rest
in `more` (`backend/api_swap.py:242-254`). Mass refine and More seeds do the
same (`api_more_seeds.py:90`).

**Design.**

- `RerenderRequest` gains `tier: float | None`, one of `sizes.TIERS`.
  Exactly one of `tier` or `width`+`height` is set, enforced by a
  validator. A request without `tier` behaves exactly as today, with the
  same run ids.
- A new pure helper, `sizes.nearest_preset(width, height, tier)`, maps a
  record's aspect to the `SIZES` entry at that tier whose ratio is nearest.
  The tolerance is a named constant in `sizes.py`, tested against the
  table's own worst rounding (the module docstring's "less than 1% framing
  error"). A record outside it is answered `unmatched` with its size and
  is not guessed or queued.
- The server decides and splits:
  - Cells are built per record at its mapped size through
    `grid.cells_from_records`.
  - Every existing refusal runs per size group: foreign, missing masks,
    mismatched outpaints, resized inpaints, unusable inputs, refine steps.
  - `check_budget` runs per distinct size.
  - The count limit stays `max_grid_images` in total.
  - One job per size, each with `grid.run_id(group, ...)`.
  - The answer is the first job at the top level and the rest in `more`,
    plus `unmatched`.

  `startJob` already registers `more` (the split-receipt repair,
  `docs/review-2026-10-29-split-extra-queue.md`).
- A preview route, `POST /api/rerender/preview`, returns the plan
  (per-size counts, unmatched ids, pruned and planned) without queueing.
  Like Swap tags' (`/api/swap-tags/preview`, `api_swap.py:172-187`) it has
  no side effects and is not a model-dependent request. The queueing
  request carries `expected`, the ids the preview answered, so a selection
  that changed in between is refused rather than queued unseen.
- Entry. A Picked menu item, "Re-render picked at N MP…", where N is the
  panel's chosen tier (`pickedTier`, `web/state.js:299`). Because it is a Picked
  menu item, the right-click menu on a picked card offers it too
  (`pickedItems`). It is held with a reason when the panel's size is not a
  preset. A press asks for the preview and shows it in the existing
  `openAsk` question ("3 at 1440×1440, 2 at 1264×1632; #9 (1000×700)
  matches no preset"). OK queues. The gallery head gets no new control: its
  rows are full at 1400 px (design-notes, Re-render, "The head holds
  still").

**Rules that apply.**

- `applied()` remains the one normalisation for every cell.
- `CellKey` and existing run ids do not change.
- The server decides the sizes, and the page only shows them.
- `grid.cell_from_record` and `submission.resolve_selection` stay the only
  routes from records to cells.
- Held items are `aria-disabled` with the reason.

**Non-goals.**

- Tier mode over a filter with nothing picked.
- Guessing a size for an unmatched aspect.
- Changing the fixed-size Re-render button or its label.
- Mixed-size single jobs.

**Stages and completion criteria.**

1. Helper. Pure tests (a new `tests/test_sizes.py`, or cases beside the
   SIZES tests in `tests/test_profiles.py`):
   - every preset maps to itself at its own tier and to its aspect's entry
     at the others;
   - near-preset sizes within the tolerance map;
   - a 1000×700 record is unmatched.
2. Route. Tests beside the existing rerender ones (`tests/test_api_grid.py`
   or `tests/test_api_rerun_queue.py`) on the fake:
   - two aspects give two jobs and `more`;
   - a budget refusal for one size;
   - unmatched records are listed and not queued, and all unmatched is a
     422;
   - the preview queues nothing, and a selection changed since the preview
     is refused against `expected`;
   - a request without `tier` gives the same run id as before.
3. Page. `smoke-select` steps:
   - the menu item's label and its hold;
   - the question lists the sizes and the unmatched ids;
   - OK queues the jobs;
   - the existing head-layout checks still pass.

## GEN#34 Swap tags Add rows

**Status (2026-10-30): built, both stages.** `tag_swap._place` and
`TagSwap.add` (stage 1, `tests/test_tag_swap.py`, `tests/test_api_swap.py`),
the dialog's at end / at start toggle (stage 2, `web/tag-swap.js`, the
Add-row step at the end of `smoke-swap`). Choices the design left open,
taken while Matt was away (2026-10-29), revisit if wanted: the control is a
small toggle button in the arrow's fixed-width slot, shown while Find is
empty, defaulting to the end; a tag two Add rows add is refused like a
find two rows share; the start's style is read from the first item even
where later items use underscores; a prose add says "needs editing by hand
- tags added to prose: + night (end)", the phrase wording unchanged.

**Evidence.** G8 (swap or remove tags and rerun images) is Often
(`preferences.md:136`). Swap tags can replace and remove tags but cannot
add one:

- `TagSwap.find` has `min_length=1` (`backend/schemas_rerun.py:128-133`).
- `tag_swap.parse_row` raises "a row needs a tag to find" for an empty find
  (`backend/tag_swap.py:62-75`).

"These keepers again with `night` added" is therefore Reuse, edit and
Generate per image, which is the work Swap tags was built to remove
(design-notes, Swap tags).

**Design.**

- `TagSwap` gains `add: Literal["start", "end"] | None = None`. A row with
  `add` has an empty `find` and a non-empty `replace`; the validator refuses
  any other combination. `parse_rows` refuses a tag that is both an Add
  row's and any row's Find, so every row still reads the original prompt
  at once.
- `tag_swap.swap` places an Add row's tags as comma items at the text's
  start or end, using the separator the text already uses between items, so
  a `,\n` layout keeps its line breaks. The tags take the prompt's spelling
  (underscores, escapes). `_styled` (`tag_swap.py:119-137`) reads that from
  the item a replacement stands in for; an Add row has none, so it reads
  the convention from the item it is placed beside (the first item for
  start, the last for end).
- An empty box (often the negative) gets the added tags as its whole text,
  joined by `, `. Today `swap` answers an empty text `unchanged`
  (`tag_swap.py:302-303`); that stays so for rows without Add.
- A tag already present is not written again. The outcome notes "already
  had X", the existing wording (`tag_swap.py:273`). An image where every
  row then changes nothing is `unchanged`.
- A text that `detect_style` reads as all prose is `by_hand`: where a tag
  goes in a sentence is an edit, not a rule. XML and JSON prompts stay
  `structured`.
- The preview marks the added spans (`Outcome.spans`). The snapshot's
  `swap` names the row ("+ night (end)", from `Row.describe`).
- Placement is only start or end, as the user chooses. No checkpoint's
  documented order is applied or invented; Check's Use reordered remains
  the way to reorder.
- In the dialog, an empty Find makes the row an Add row, with a start/end
  choice. The rows box keeps its fixed three-row height and each row keeps
  its height, so Queue never moves. The exact control is chosen in stage 2
  against that constraint.

**Rules that apply.**

- The server rewrites, and the page shows its answer.
- Strict, no partial application.
- Per image, never the whole job.
- One job per size.
- The request limits (16 rows, 1000 characters) are held on the page.
- `applied()` for every cell.
- Never invent a documented order.

**Non-goals.**

- Inserting at a position other than start or end, such as after a named
  tag.
- Adding into prose.
- Weighted insertion.
- Any analyser finding or `cite`.

**Stages and completion criteria.**

1. Rewrite. `tests/test_tag_swap.py` cases:
   - add at start and at end, in comma, `,\n` and underscore styles;
   - an already-present tag gives the note, and `unchanged` when nothing
     else changes;
   - prose gives `by_hand`, and XML gives `structured`;
   - an Add beside a replace row;
   - the negative field, empty and not;
   - an Add tag that is also a Find is refused.

   `tests/test_api_swap.py`: the schema refusals, and that a non-Add request
   keeps its current answer.
2. Dialog. One `smoke-swap` (7900) step: an Add row on two images previews
   the marked addition and queues, and Queue's coordinates do not move
   while rows are added.

## GAL#23 Mistyped search keys

**Evidence.** E5 (search by prompt words) and E9 (search from an image
setting or prompt) are Often (`preferences.md:99`, `:103`). `split_search`
treats any token whose key is not in `FILTERS` as prompt text
(`backend/search.py:282-299`), so `sed:42` searches the prompts for the
word "sed:42". When that finds nothing, `syncGalleryEmpty` says only "No
image matches this search" (`web/gallery.js:139-170`) and gives no clue that
the filter was never applied. The page already holds the key list
(`SEARCH_KEYS`, `web/gallery-search.js:11`, held equal to `search.FILTERS`
by `tests/test_storage.py`).

**Design.**

- When the search finds nothing, `syncGalleryEmpty` checks each
  `word:value` token that `searchKey` rejects. If the word is near exactly
  one of `SEARCH_KEYS` (edit distance at most 2, a transposition counting
  as 1, the distance less than the key's length), the message adds "sed:42
  was searched as prompt words; did you mean seed:42?" with a small ghost
  button, "Search seed:42".
- The button replaces that token in the search box and asks again
  (`resetGallery`), as `showGallerySearch` writes the box, but without its
  scroll: the empty line sits under the head, which is already in view.
- Tokens near no key, such as `masterpiece:1.2`, add nothing. A prompt
  that contains a colon still searches for itself.
- The recycle bin's empty line gets the same hint.
- Nothing is said when the search has results.

**Rules that apply.**

- The page never scrolls by itself.
- The empty line is in the grid's flow under the head, so the head's
  height is unchanged.
- The server decides the grammar, and the hint only reads the page's
  mirror of the key list.
- The search box is not a prompt box, so `writeBox` does not apply.

**Non-goals.**

- Changing the grammar or refusing unknown keys.
- Autocomplete in the search box.
- Hints on non-empty results.

**Stages and completion criteria.**

1. One `smoke-gallery` (7885) step:
   - `sed:42` shows the hint, and the button searches `seed:42`;
   - `masterpiece:1.2` shows the plain empty line;
   - the head's height is the same before and after.

## GEN#36 Mass refine versions

**Status (2026-10-30): built, both stages.** `RefineBatchRequest.versions`,
`refine_batch.plan(versions=)` with `Planned.more`/`cells`/`drop`, the
route's per-version pruning, budget refusal and the preview's
`queued_images`/`versions` (stage 1: `tests/test_refine_batch.py` -
seeds counting up one cell each, the first version's key unchanged, wrap
at `2**63 - 1`; `tests/test_api_refine_batch.py` - two cells per record
at `seed` and `seed + 1` across two size jobs, one version in the gallery
queuing only the other (and 4 the two it lacks), `versions=1` the same
preview and the run id `grid.run_id` gives the plan's cell, the
images × versions refusal by count). The dialog's Versions select
(stage 2, `web/refine-batch.js`, two steps of `smoke-refine-batch`: the
doubled count, Refine and Cancel unmoved and the select on the controls'
one row at the boxes' height; a picked image queued at 2, held rendered
at 2 again and 4 queuing the two it lacks). Choices the design left open,
taken while Matt was away (2026-10-30), revisit if wanted: the select is
the row's last field, after Steps, so Strength and Scale keep their
places; Versions opens at 1 every time and is not remembered; the count
reads "6 passes will be refined (3 images × 2 versions)", with ", k
already made or queued" inside the brackets where some are held, the
button "Refine 6 passes" and the dialog's title still the images; later
versions' labels are `#<id> v2` and so on; the budget counts the images
the plan would queue (before the gallery is asked) × versions; a
partly held image's row note is "k of V versions already in the gallery
or queued".

**Evidence.** H4 (try several strengths or versions) and H5 (mass-refine
picked or filtered images) are Often (`preferences.md:146-147`).

- The single Refine dialog has Versions (1, 2 or 4): seeds counting up
  from the seed box, a pass being one call per image (`web/refine.js:576-584`,
  `:722-757`; `web/index.html:1211-1212`).
- Mass refine has one cell per image at the record's own seed and no
  versions (`RefineBatchRequest`, `backend/schemas_rerun.py:215-238`;
  `refine_batch.plan`, `backend/refine_batch.py:67-140`). Its dialog offers
  strength, scale and steps only (`web/index.html:1685-1725`).

**Design.**

- `RefineBatchRequest` gains `versions: Literal[1, 2, 4] = 1`, the single
  dialog's choices.
- Each queued record becomes `versions` cells at seeds `seed`, `seed + 1`,
  and so on through `wrap_seed`, each with `num_images=1`. These are the
  seeds the single dialog's Versions give. One cell per version means
  `prune_done` and `not_yet_rendered` prune per image. A single cell with
  `num_images > 1` is done only when every image is in the gallery, and a
  batch with some of its images there is rendered again in full, those
  images included (`cell_keys` and `prune_done`, `backend/grid.py:760-784`).
- The first version is today's cell unchanged, so `versions=1` keeps every
  key, run id and status. A record is `rendered` only when all its
  versions are, and otherwise queues the missing ones.
- The audit's draft row said to step over taken seeds as More seeds does
  (`more_seeds.next_seeds`). That is not used: it would make the same press
  twice queue new seeds instead of the no-op "already rendered" that Mass
  refine promises.
- The price is the number of queued cells. The total stays within
  `max_grid_images` (images × versions), refused by count. Time is per size
  from `refine_cost_ms`, as now.
- One strength stays one strength. The dialog's "One strength for a batch"
  refusal is unchanged.
- In the dialog, a Versions select joins the Strength/Scale/Steps row. The
  dialog's height at 1400×900 is unchanged and Refine does not move. The
  count says "N passes (M images × V versions)".

**Rules that apply.**

- `grid.cell_from_record` for every cell; `grid.expand()` stays the one
  cell loop.
- Seeds wrap at `2**63 - 1`.
- `applied()` for every cell.
- `CellKey` and run ids do not change for `versions=1`.
- One job per size, pinned to `expected`.

**Non-goals.**

- Strength lists. Several strengths per image is Then refine's open
  decision ([plan-then-refine.md](plan-then-refine.md#decision-several-strengths-per-image)).
- Random version seeds.
- Versions for Swap tags or Re-render.

**Stages and completion criteria.**

1. Server. Tests in `tests/test_refine_batch.py` and
   `tests/test_api_refine_batch.py`:
   - `versions=2` gives two cells per record at `seed` and `seed + 1`;
   - with one version already in the gallery, only the other queues;
   - `versions=1` gives the same run id and statuses as before;
   - the budget refusal counts images × versions;
   - wrap at the seed ceiling.
2. Dialog. One `smoke-refine-batch` (7901) step: Versions 2 shows the
   doubled count and queues the passes, and Refine's coordinates are the
   same with the select present.

## GEN#38 Gallery timing in the model picker

**Evidence.** K2 (compare model guidance, timings or examples) is Often
(`preferences.md:189`). The picker's Speed fact comes only from the
retained card samples in `model_benchmarks._SAMPLES`, through `speed_for`
(`backend/model_benchmarks.py:150-215`) and `model_info.details_for`
(`backend/model_info.py:380-431`), which stays cold by design. It is shown
in the Speed cell and its note (`web/model-switch.js:244-249`, `:288-299`).

The gallery's own history of a checkpoint (`images.duration_ms`) is used for
ETAs (`gallery_reports.step_cost_ms`, `backend/gallery_reports.py:40-68`) but
never shown here.

The rows record `quantization` and `execution`, but not the offload mode or
whether an image was a process's first (`backend/storage.py:144-183`). So
"mixed offload" and "first of process" cannot be filtered out row by row.

**Design.**

- A new pure-SQL helper in `gallery_reports` answers per checkpoint in one
  grouped query, off the event loop:
  - which rows: the latest direct renders (no `refine`, no `edit`, Eager
    `execution`, `duration_ms > 0`) under the selected quantization;
  - the size they used most;
  - at that size and the most common step count, the median seconds per
    image and how many images that median is of.
- The `/api/models` route (`backend/main.py:728-797`) adds the helper's
  answer to each model's details inside the existing `_detached` call.
  `model_info` stays cold, and a missing database answers nothing.
- It is shown as "This gallery: ~N s at W×H, S steps (median of n)". It is
  added to the Speed note, the tooltip and description that `modelNotes`
  already builds, so no row grows. It is labelled as this gallery's
  history, not a measurement. The note says that offload is not recorded per
  image, and that the median keeps a single slow first image from moving
  the figure.
- It never sets `speed_label` or the category. With no rows, nothing is
  said.

**Rules that apply.**

- The catalogue stays cold.
- Timing claims are the card's (`docs/verified.md`). This line is labelled
  as gallery history and never feeds a category.
- No new visible row height in the picker. `smoke-model-hover` holds rows
  still under the pointer.
- Writes go in place (`syncModelDownloadFacts`), never replacing a row
  under the pointer.

**Non-goals.**

- A visible second line in the table.
- Adding an offload or first-of-process column.
- ETAs.
- Changing `_SAMPLES`.

**Stages and completion criteria.**

1. Helper and route. A helper test on a planted throwaway database:
   - refine, edit and non-Eager rows are left out;
   - the most-used size and step count;
   - the median and n;
   - none for an empty history.

   `tests/test_api_model_switch.py`: the field appears per model and
   `speed_label` is unchanged.
2. Page. One `smoke-model-hover` step: the Speed note includes the line on
   the fake after images are planted, and no row moves.

## UI#38 Caps Lock notice on bundle passwords

**Status:** done 2026-10-30, stage 1, on `feat/todo-backlog-2026-10-29`.
The file:line claims below were re-checked: the three inputs now sit at
`web/index.html:1803-1829` and the `.bundle-password` rules at
`web/styles/bundles.css:41-47`; nothing read `getModifierState` before.
Built as designed: each `.bundle-password` grid holds one
`p.caps-lock-note` (`aria-live="polite"`, empty while off, so it stays in
the accessibility tree), and `bundles.js`'s `watchBundleCapsLock` sets its
text on `keydown`/`keyup` and clears it on blur. Choices taken while Matt
was away (2026-10-29), revisit if wanted:

- A `focus` event carries no modifier state, so focus repeats the state the
  last key event in any of the three fields reported (held in memory only,
  `bundleCapsLock`); before any key it shows nothing.
- One notice per grid, not per field: the new-password pair shares one,
  centred beside both inputs, in the space after the input column
  (absolutely placed from grid line 3 to the grid's right edge, wrapping
  there if narrow). In the one-column layout it sits at the right of the
  first label's row (see the review fixes below).
- The colour is `--warn`; the notice holds no pointer.

Review fixes (2026-10-30): between 521 and about 600 px the two-column grid
left the notice too little room, so it wrapped word by word and, near
521 px, ran past the grid's edge. The grids now go one column below 620 px
(was 520 px), and the notice stays on one line (`white-space: nowrap`), at
the right of the first label's row there rather than "under the field's
label" as the design said, since an out-of-flow line under the label would
cover the input (taken while Matt was away (2026-10-29), revisit if
wanted). `smoke-bundles` also checks the notice at 560 and 600 px and that
it neither wraps nor overflows its grid; that check fails on the first
build's CSS.

`tests/test_bundle_caps_lock.py` holds the markup, the out-of-flow rules
and the watcher's reads; `smoke-bundles` checks the behaviour at 1400x900
and 360x800.

**Evidence.** J11 (password-protect a bundle) and J12 (bundle and clear the
working gallery) are Often (`preferences.md:179-180`). The three password
fields, `#bundle-password`, `#bundle-password-again` and `#bundle-unlock`
(`web/index.html:1789-1816`), give no Caps Lock notice; no page script
reads `getModifierState`. A password cannot be recovered, and a cleared
gallery is lost without it (`#bundle-password-hint`).

**Design.**

- On `keydown`, `keyup` and `focus` of each password field,
  `event.getModifierState?.('CapsLock')` shows or hides "Caps Lock is on".
  Blur hides it.
- It is a small `aria-live="polite"` line positioned absolutely beside the
  inputs in `.bundle-password`'s grid (`web/styles/bundles.css:41-47`),
  which gains `position: relative` to hold it, so its coming and going
  moves nothing. It is not written into the long hint.
- Under the 520 px single-column layout it sits under the field's label,
  still out of the flow.
- Nothing about the password is stored or sent, only the modifier state.

**Rules that apply.**

- The page never moves a control under the pointer.
- CSS uses the `:root` tokens.
- Any wide-tier rule goes in the section's `@media (min-width: 1800px)`
  block (`bundles.css:92`).
- No `localStorage` use.

**Non-goals.**

- A show-password toggle.
- Strength meters.
- Notices on other fields.

**Stages and completion criteria.**

1. One `smoke-bundles` (7897) step:
   - a synthetic `keydown` whose `getModifierState('CapsLock')` is true
     (`new KeyboardEvent('keydown', { modifierCapsLock: true })` in
     Chromium) shows the notice;
   - false hides it, and so does blur;
   - the Back up button's and the inputs' coordinates are unchanged
     either way.
