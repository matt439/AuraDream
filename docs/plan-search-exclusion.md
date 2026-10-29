# Plan: gallery search exclusion

**Status:** `GAL#21` open, 2026-10-29 (added by the 2026-10-29 TODO audit).
Nothing is built. Stages 1 and 2 are ready CPU/fake/browser work. Stage 3
waits on Matt's answer to [the strict-scope decision](#decision-strict-scopes);
until then, strict scopes keep refusing every exclusion, as they do today.

## The defect

Prompt search and searching from an image's settings are both reported Often
([preferences](design/preferences.md) `E5`, `E9`). Delete unstarred in the
current scope is Often too (`J6`). The grammar has no way to say "without".
A typed exclusion does the opposite of what it says:

- `split_search` (`backend/search.py:282-297`) treats any token whose key is
  not in `FILTERS` as free text. `-blonde` has the key `-blonde` and
  `-seed:1` has the key `-seed`, so both become free text.
- `list_images` quotes each free-text token as an FTS5 phrase
  (`backend/storage.py:1030-1031`). The `images_fts` table
  (`backend/storage.py:186-187`) uses the default tokenizer, which drops the
  `-`. Checked 2026-10-29 on an in-memory FTS5 table: `MATCH '"-blonde"'`
  returns the row `a blonde girl`, the same as `"blonde"`. So `-blonde`
  returns exactly the images the user meant to drop.
- `-seed:1` becomes the phrase `"-seed:1"`, which matches only prompts that
  contain the words `seed 1`. Usually that is nothing, and nothing says why.

Strict scopes are not affected today. `strict_scope`
(`backend/search.py:306-320`) refuses any free text, so `-word` and `-key:value`
are already refused by Delete unstarred (`backend/api_gallery.py:29-43`), the
filtered delete (`backend/gallery_delete.py:133`, `:238`), keeper rates
(`backend/gallery_reports.py:223`) and `clear --where` (`backend/clear.py:100`).
`tests/test_search_orientation.py:120-125` pins `-size:portrait` as
unknown-key free text that strict scopes refuse, and
[the orientation plan](plan-gallery-orientation.md#search-contract) records
that "negation gain[s] no new meaning". This plan changes both on purpose.

## Contract

A token whose first character is a single `-`, with at least one more
character and that character not `-`, is an **exclusion** of the rest of the
token (the operand).

- **`-key:value` with a known key** is the complement of `key:value`. It is
  read with the same `filter_clause`, so the same values parse and the same
  ValueError refuses a bad one (`-seed:abc` becomes a 422 rather than an empty
  page). The SQL is `COALESCE((clause), 0) = 0`. A row where the clause is
  NULL, such as `lora LIKE ?` on a row without a LoRA, did not match
  `key:value`, so it is kept. The invariant is that, under the same other
  filters, `X` and `-X` together count every row exactly once.
- **`-word` (anything else)** excludes the rows whose prompt the full-text
  index matches for that word:
  `id NOT IN (SELECT rowid FROM images_fts WHERE images_fts MATCH ?)`, with
  the excluded words quoted as phrases and joined by `OR`. A query that has
  only exclusions needs no positive MATCH. `prompt` is `NOT NULL`
  (`backend/storage.py:148`), so the LIKE form needs no NULL handling.
- **Unchanged:** a lone `-`, a token starting `--` (for example `--ar`) and
  a quoted token (`"-x"`) stay free text, as today. A `-` inside a value
  (`cfg:0-1`, `seed:-5`, `date:2026-10-26..2026-10-28`) is still read by the
  value-side grammar, which is untouched.
- Terms still AND together. `seed:>10 -seed:12 fox -blonde` is a valid query.
  A contradiction (`seed:1 -seed:1`) matches nothing, which is the answer the
  SQL gives. It is not refused.

### LIKE fallback

The fallback (`backend/storage.py:1041-1053`) handles a token with an odd
number of double quotes (`26"`) by swapping `where[-1:]`. That only works
because the one search predicate is appended last (`docs/rules.md`, the
`storage._LIVE` paragraph: "before the search predicate, which the LIKE
fallback swaps as the last"). With exclusions there can be two text
predicates. The fallback must keep their positions (or append both last and
swap both), and must move both to LIKE together, so that a positive word and
an excluded word are never matched in two different ways. An excluded word
becomes `prompt NOT LIKE ? ESCAPE '\'`, one per word. The `rules.md` sentence
is reworded to match. Favorites, `_LIVE`, `binned` and the checkpoint scope
must still apply after a fallback; the existing `26"` test
(`tests/test_storage.py:533`) gains an excluded-word twin.

### Where it applies automatically

Every reader built on `list_images` gets exclusion without changing its own
code: the gallery and the sheet, the bin's view, the other-checkpoint count
(the `others` clause), and the filtered rerun producers that resolve through
`submission.resolve_selection` (`backend/submission.py:446`). Each producer
keeps its own ceilings and confirmations. Recent runs takes no query and is
not extended.

### Saved searches

`SavedSearchFields.valid_query` (`backend/schemas_saved_searches.py:50-56`)
runs `split_search`, so writes accept a valid exclusion and refuse
`-seed:abc` the way they refuse `seed:abc`. The read path has a catch:
`saved_searches._read` (`backend/saved_searches.py:39-65`) runs
`SavedSearch.model_validate` on every stored item, and any ValueError makes
the whole collection unreadable ("Saved searches could not be read"). A query
saved before this change, such as `-seed:abc`, was then valid free text. It
must not lock the user out of every saved search. Reading a sidecar keeps
today's length and Unicode checks but does not refuse an item over grammar
added after it was written. That item's Apply reports the gallery's 422
reason, as any invalid search does. The raw query text is never rewritten.
`web/saved-searches.js` keeps writing the stored query verbatim
(`applyGalleryFilter`, `web/saved-searches.js:269-274`); it does no parsing of
its own.

## Frontend

`web/gallery-search.js` mirrors the server's reading. The server still
decides what a term means; the page only arranges words in the box.

- `searchKey(word)` (`:26-31`) reads through one leading `-`, so the page
  knows `-seed:1` is the `seed` filter, negated. Exactly how is up to the
  implementer: a second helper or a `{key, negated}` result both work.
  `SEARCH_KEYS` (`:11`) and its parity test (`tests/test_storage.py:282-287`)
  are unchanged; there is no new key.
- `mergeSearch` (`:37-51`): a pressed `key:value` replaces every term with
  that key, negated or not, where the first one stood (`-seed:1` plus ⌕
  `seed:5` gives `seed:5`). This is the existing "a filter replaces the box's
  terms with its key" rule extended to exclusions. A plain word pressed
  removes its own exclusion, compared case-insensitively.
- `mergePromptSearch` (`:57-66`): "Search for this prompt" keeps the box's
  filters and its exclusions (`-key:value` and `-word`) and replaces only the
  positive free words.
- `promptSearchWords` (`:71-73`): a prompt word that starts with `-` would turn
  into an exclusion once written into the box. Its leading hyphens are
  stripped. The FTS tokenizer drops them anyway, so the stripped word matches
  the same rows that the hyphenated phrase matched before.
- The ⌕ buttons write no exclusions, and nothing new goes on the panel,
  whose rows are full at 1400x900.
  The Search help tip (`web/index.html:830-848`) gains one line:
  "**-word** or **-key:value** leaves those images out". The tip is
  absolutely positioned (`web/styles/fields.css:222-224`), so the line moves
  no control. A smoke still checks that nothing in the panel moves under
  1800 px.

## Decision: strict scopes

Strict scopes (Delete unstarred, the filtered delete, `clear --where`,
keeper rates in `stats`) are where a wide match costs most. An exclusion
widens the scope: `-seed:1` passes the "a filter is required" check
(`backend/gallery_delete.py:244-245`) and selects nearly the whole gallery.
That hazard is not new, since `seed:>0` already does the same, and the
existing guards still hold. The browser shows the count, the count must be
typed back, and `delete_matching` pins it as `expected` (409 on a mismatch).
`clear` shows its preview and asks for `clear`.

Matt to choose:

- **A (recommended).** Strict scopes accept `-key:value`, which is exact SQL
  with no fuzzy index. They keep refusing `-word`, which goes through FTS
  like any free text. The refusal names the term: "`-blonde` is free text;
  strict scopes take key:value terms and their exclusions".
- **B.** Strict scopes refuse every exclusion, with a message that names
  exclusion instead of calling `-seed:1` free text.

Until Matt answers, stages 1 and 2 ship behaviour B, which matches what is
refused today. `backend/clear.py:114`'s `"model:" in args.where` check also matches
`-model:`. It only adds a warning line, which is harmless, and stage 3 keeps
it.

## Non-goals

- No OR or grouping, no quoted phrases with spaces, no `NOT` keyword, and no
  exclusion of the negative prompt (FTS indexes `prompt` only).
- No "exclude this" ⌕ button and no new panel control.
- No new search key, column, migration or index. `CellKey`, `run_id` and
  `applied()` are untouched; this is a read filter only.
- No change to how positive terms or values parse.

## Stages and completion criteria

### Stage 1: grammar and gallery reads

`backend/search.py` (exclusion parsing, `split_search` returning the
excluded words beside the free text), `backend/storage.py` (the NOT IN and
NOT LIKE predicates, the two-predicate fallback), `strict_scope` behaviour B,
and the saved-search read tolerance.

Done when pytest covers:

- `-word` drops FTS matches and keeps the rest.
- `-key:value` for one integer key, one float key, one text key, `date:`,
  `size:` (exact, orientation and MP) and `inpaint:`.
- The NULL-row invariant: `lora:ink` plus `-lora:ink` equals all rows, and
  the same for `mu:0.5` and `tcfg:1.5` (rows without a mu or a true CFG
  give a NULL clause).
- An exclusion-only query.
- The `26"` fallback with an excluded word.
- Favorites, bin and checkpoint scope, including `others`, with exclusions.
- The unchanged cases: `-` and `--ar` stay free text; `cfg:0-1` and
  `seed:-5` stay the same filters as today.
- `-seed:abc` gives a 422.
- A saved-search sidecar holding `-seed:abc`, written in the old format,
  still reads.
- Strict scopes refuse `-seed:1` and `-blonde` with the new wording.

`tests/test_search_orientation.py:120-125` is updated for `-size:portrait`.
Ruff and the pinned Pyright gate pass.

### Stage 2: page mirror and docs

`web/gallery-search.js` as above, plus the tip line. `docs/usage.md`'s search
table (`docs/usage.md:589-605`) gains the row
``| `-blonde`, `-seed:42`, `-lora:ink` | Leaves those images out; images
without the value (no LoRA, no mu) stay in, so `-lora:ink` keeps the ones
without a LoRA |``. Also updated: `docs/rules.md`
(the `search.py` module-map line, the fallback sentence and the
`mergeSearch` paragraph), and `docs/design-notes.md` under "The search
grammar", with a pointer from the orientation plan's search contract.

Done when:

- The Node VM supplement (pattern of
  `tests/test_search_orientation.py:19-45`) checks `searchKey('-seed:1')`.
  `mergeSearch('-seed:1 fox', ['seed:5'])` gives `fox seed:5` with the seed
  where `-seed:1` stood. `mergePromptSearch` keeps `-blonde` and `-seed:1`.
  `promptSearchWords('a -style cat')` contains no `-`.
- One `smoke-gallery` step types `fox -blonde`. The cards that load have no
  `blonde` in their prompt. The total equals the server's
  `/api/gallery?search=` total. No panel control's rectangle changes at
  1400x900.

### Stage 3: strict scopes (after Matt's answer)

This stage runs only if Matt chooses A. `strict_scope` accepts
`-key:value`. Done when pytest shows these preview counts:

- `matching_preview` for `-run:X` plus the count for `run:X` equals the
  scope's total.
- `delete_matching` with an exclusion still answers 409 on a stale
  `expected`.
- `keep_rates` for `-lora:none` excludes binned and other-model rows.
- `clear --where "-seed:1"` prints its preview and refuses without
  confirmation (owned scratch gallery only).

`smoke-gallery` or `smoke-delete` presses Delete unstarred with an excluded
term and sees the count dialog. If Matt chooses B, this stage closes with no
code change and the decision recorded here.

No stage needs the card, a model load or a download.
