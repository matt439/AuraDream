# Review 2026-09-28 — implementation checklist

Companion to `review-2026-09-28.md`. One line per entry of the review, in the
review's own order and with its own numbers, titles and `file:line` anchors
(which refer to commit `e7d2386`, the tree the review was made against - the
lines have moved since). Status as of 2026-09-29, after the commits listed at
the end.

| Section | Entries | Done | Partial | Not done |
| --- | --- | --- | --- | --- |
| HIGH | 1 | 1 | 0 | 0 |
| MEDIUM | 23 | 23 | 0 | 0 |
| LOW | 57 | 56 | 1 | 0 |
| Refuted | 1 | — | — | — |
| Set aside at synthesis | 11 | — | — | — |

The refuted candidate and the eleven sub-claims set aside at synthesis asked
for nothing to be done; their sections below say where what survived of each
went.

Legend: `[x]` done, `[~]` partial (what is missing is in the note), `[ ]` not
done. The commit named is the one that closed the entry - several, where a
lane's fix was finished by a follow-up, a phase-4 fix or the last fix round;
the note says where the fix departs from what the review suggested. The docs
each fix needed landed in `4ab3fa0` (phase 3), in the later commit that made
the change or in `39e9863`'s corrections, and an entry names a docs commit
only where a doc change was part of the fix itself. The test suite went from
1007 passed (at `e7d2386`) to 1656 passed and 4 skipped on Windows (at
`8931fae`; the skips are a file-symlink test and three Linux-only ones).
`web/` has no test suite and was exercised in headless Edge against the fake
backend: 15 smoke scripts, each on a fresh fake server, all passing with no
failed step at `8931fae`. The lane commits were rebuilt from the lanes'
patches and were not each tested; the whole suite ran on each phase's final
tree.

Under WSL 2 the final tree (`8931fae`, only `docs/` changed in the working
tree) passes too, with `LOCAL_DIFFUSION_QUANT` and
`LOCAL_DIFFUSION_MODEL_ID` unset after sourcing `wsl/env.sh` (the defaults,
FLUX.1 Krea [dev] in fp8, as on Windows): 1648 passed and 12 skipped, the
same 1660 as Windows, and the four tests Windows skips ran and passed, run
again by name. With `env.sh` sourced as it is, which exports Nunchaku and a
lite-infer Krea 2 checkpoint, six tests that assume the default checkpoint
and quantizer fail (`test_conditioning.py:103`, `test_nested_models.py:129`
under fp8, int8 and none, `test_storage.py:367` and `:397`): the
environment, not Linux or this branch. The twelve skips are the two Windows
job-object tests and ten real-tokenizer cases whose tokenizers WSL's Hub
cache does not hold. The run used `python -m pytest`, since the WSL venv's
`bin/pytest` still names the pre-rename `krea-local` interpreter, and synced
no dependency: `uv.lock` and `pyproject.toml` are as on `main`.

Verified on the card and recorded in `verified.md` under "Prompt analyser".
On 2026-09-28, only the critic-server lane's runs of the real pinned pair
(b11206 `llama-server`, Qwen3-4B Q8_0), on Windows and under WSL 2. A FLUX.1
Krea [dev] critique took 21.1 s on Windows and 38.1 s under WSL 2 (the GGUF
read from `/mnt/c`); the server's own count of the question, 1,072 tokens,
was its `prompt_n`; the listener check passed (the job's pid in the TCP
table, the session's socket inode under WSL); and after each stop no server
was left and the card was at its baseline. A 2,500-character Greek prompt
the estimate admitted was refused on the server's count, unasked (3.1 s;
27.8 s under WSL). The same day's other rows - the Ubuntu tree's libraries,
the tokenizer behind the estimate, the stop under full CPU load and
kill-on-close - are CPU checks or ran a stand-in server. On 2026-09-29, the
real critic end to end through the app's own routes on the final tree
(`8931fae`, `--defaults`, the image model never loaded), four critiques on
Windows, each answering 200 and leaving no `llama-server.exe`: FLUX.1 Krea
[dev] twice, 11.8 and 13.1 s at 1,177 prompt tokens, +5,570 MiB over the
idle app's baseline on the second (the first run's poll wrote nothing); and
Animagine XL 4.0 twice with the Danbooru index reachable, 20.3 and 18.3 s
at +5,626 and +5,687 MiB - an unknown tag ahead of `1girl`, which Check
calls checked by design, and one whose category decides the order, which
it calls "Partly checked". Those answers also showed what validation lets
through: issues restating Check's `guide.negative_prompt` and
`guide.quality` findings, which `critic._RESTATES` has no pattern for; a
citation misapplied (A3, the card's tag-format sentence, for a missing
negative prompt); and rewrites that pasted the card's recommended negative
prompt into the prompt and added "[series]" placeholders, which only the
rewrite's lint flagged. A CPU check the same day, under "Behaviour", ran
the real `single_file.sweep_work` over this machine's conversion cache: it
removed the three empty work directories that conversions interrupted on
2026-09-17 had left beside Illustrious's tree, and touched no published
tree. Not verified: a critic reorder under partial placement on the real
server, since both Animagine answers' reorders changed words and were set
aside before that check ran; the WSL critique with the GGUF on ext4, the
default, was never measured, warm or after a page-cache drop (F78);
kill-on-close was not tried with `llama-server`; and no image generation
was re-measured in this series, so no render time, peak or pixel hash in
`verified.md` comes from it.

## Confirmed findings

### HIGH

- [x] F1. `backend/prompt_rules.py:863` CJK lettering target always 'exact': Apply quotes description and splits words — `ad94ac3`, `53ba37b`. A Chinese target is exact only where whitespace, punctuation, a quotation mark, a bracket or the end of the text ends it; stopped at 的/之, at the twelve-character cap or (`53ba37b`) at a placing verb with 在/于, it keeps the finding but gets no Apply. Beyond the review: Latin lettering running on past a space or an ASCII mark is not exact, 字样 and a character count end the lettering for certain, and a bare 在 is deliberately no stop (see What remains).

### MEDIUM

- [x] F2. `backend/prompt_rules.py:836` English lettering triggers read description as lettering and offer Apply — `ad94ac3`, `46b721e`, `bae8d66`, `14f0e24`, `c7a33b9`. A describing first word gives no target (a closed list, -ly adverbs and, since `46b721e`, after a noun trigger a participle before a preposition of place), "a book of spells glowing" is no trigger (`_NOUN_USE`, in place of the review's strict bearer adjacency), and lowercase words after a participle trigger, or anything not in capitals after a noun trigger, keep the finding without an Apply - not after a finite verb, so "says open" keeps its one click. `bae8d66`'s stop at a negation inside a participle's words left `ends` False, so the critic took the description after it as lettering ("love without limits above the stage"); `14f0e24` reads the words on through the negation to where they would stop without it, `ends` True, for the critic alone, and `c7a33b9` takes lettering only from words that start before the rules' stop (`head`), so "without pictures" alone is flagged, and lets a negation after "or" or "than" close the lettering ("now or never in bold type"). The residue and the costs are in What remains.
- [x] F3. `backend/prompt_rules.py:1452` Repeated, lowercase and underscore rules rewrite text inside quotation marks — `ad94ac3`, `53ba37b`. The repeated, lowercase and underscore rules skip lettering (`ad94ac3`), and `53ba37b` goes past the review: each stretch of lettering - quoted, unquoted or in 《》 - is one comma unit that no tag rule rewrites and the reorder moves whole, and a move that would change quoted text is refused.
- [x] F4. `backend/prompt_rules.py:239` CJK lettering in ASCII single quotes after an ideograph is not read as quoted — `ad94ac3`; test `e563622`. `QUOTED`'s single-quote pair may touch Han, kana and Hangul (`_WORD_NOT_CJK`); the ERNIE `validate` case the review asked for came with the critic lane.
- [x] F5. `backend/critic.py:982` critic.locate matches substrings, so quotes, parts and lettering land inside words — `e563622`. `critic.locate` matches whole words where the quote starts or ends in a spaced script; beyond the review's CJK exemption, Thai, Lao, Myanmar and Khmer match anywhere too, and `_` counts as a space.
- [x] F6. `backend/critic.py:1095` Lettering accepted past an exact end: description passes as text to render — `e563622`. The critic reads the rules' `ends` flag (`rendered_targets`) rather than `exact`: past a sure end only words starting with a capital or a digit run on, so Title Case ("Cafe On Main") runs on as well as capitals. The neighbouring path `bae8d66` opened through a guessed end, counted under F2, closed with it (`14f0e24`).
- [x] F7. `backend/critic.py:1305` Rewrite dropped as invented lettering when only the apostrophe or dash differs — `e563622`. Apostrophes, quotation marks and dashes fold one character for one in `locate`, `_flat`, the reorder and the unchanged check, more characters than the review listed; its optional warning for lettering whose punctuation changed was not built.
- [x] F8. `backend/critic.py:949` is_reorder casefolds pieces, so 'Reordered' may change case of unquoted lettering — `e563622`. A kept reorder is rebuilt from the prompt's own pieces in the critic's order, with the critic's separators between them; where Check placed the order itself, it is Check's own text.
- [x] F9. `backend/critic.py:1168` Restated check drops issues whose quote merely sits near a finding — `f9f4ae8`. A quote restates a finding only if it takes in some of the finding's words and runs at most the slack past them; beyond the review, surrounding punctuation is trimmed first, so "no people," still counts.
- [x] F10. `backend/critic.py:869` encoder.unknown_chars spans make the critic's quoted issues vanish as 'restated' — `f9f4ae8`. Only guide, author, heuristic and weighting spans can make an issue restated (the review's allow-list option, with `author.*` added), and a single word, measured by whitespace, gets no slack.
- [x] F11. `backend/critic.py:1160` Word-count and weighting patterns drop unrelated critic issues with a false note — `f9f4ae8`, `f816e11`, `7a633d0`, `bb2193c`, `8931fae`. `reported` holds the prompt's findings only, the measured note is given only where the count is in the guide's range and not XML or JSON, and a count of words given to a part is kept (`_SOME_PART`, `_PART_COUNT`, `_MANNER`, each word pinned by a test row). Since `bb2193c` the word just ahead of "description" decides whose it is, so "keep the lighting description under 5 words" is kept: a determiner or a word for the whole makes it the prompt's (`_DETERMINERS` and `_WHOLE_WORDS`, one list each for both readings since `8931fae`, the critic's "revised", "new" and "final" among the words), and a part may be named in up to four words. "The prompt reads as long" is a length claim, and "as long as" none. Beyond the review, a negative-prompt finding still restates an issue that names the negative prompt; it remains a reading of the wording (see What remains).
- [x] F12. `backend/main.py:636` Order shown as checked when the tag-category lookup failed — `7381c9c`, `7cbb55f`, `06ab194`, `14f0e24`. `order_checked` stays true offline, as the review's first option: `categories_detail` says why the lookup lost categories, and a separate `order_detail` names the unplaced items where their slot decides the order, the page's "Partly checked" and the critic's `placed_partly` reading one predicate. Since `14f0e24` a tag and its copy are no pair, so "1girl, zzkx oc, zzkx oc, masterpiece" is checked; a copy in another case still counts, which errs toward "Partly checked". Anima says "artists not all checked" rather than "did not run". With the real Danbooru index on 2026-09-29, "1girl, green hair, zzkx oc, smile, outdoors, masterpiece" said "Partly checked", naming "zzkx oc", and "zzkx oc, 1girl, ..." did not, by design.
- [x] F13. `backend/prompt_rules.py:459` Newline- or fullwidth-comma tag lists read as tags but the order is never placed — `ad94ac3`. The order stands down when a newline or `，` splits a comma item (`_placeable`), and the quality rules count those pieces as present; the optional ASCII-comma finding was not added, since no guide sentence says it.
- [x] F14. `backend/guides.py:42` Critic may cite guide sentences that their own linked source contradicts — `d18c2fa`. "English is the most precise language" is removed, not merely listed unsourced, and schnell's prose states this app's 512 tokens with BFL's 256 kept as an unsourced, badge-only sentence - the review's rewording option, not `max_sequence_length=256`.
- [x] F15. `backend/prompt_sources.py:888` App-written guide sentences are cited as the authors' guide — `d18c2fa`. 19 app-voice sentences across 10 guides are listed unsourced, Z-Image's sentence matches its card, and the fallback keeps its G sentences with url None and no "authors" wording (the review's second option). Sentences on how the pipeline reads a prompt (Kohaku's 77, NewBie's 512/2048, ERNIE's 2048) stay citable.
- [x] F16. `backend/critic.py:767` Translated and borrowed excerpts cited on the page as the authors' own words — `f9f4ae8`. A citation carries its excerpt's note - whose it is, translated or not - onto the issue, its tooltip and its label, and the fallback's url-less sentences show as General with a general-advice note; `original` is not exposed.
- [x] F17. `backend/prompt_encoding.py:880` Cut message quotes the prompt's opening for unspaced CJK text — `3457a6f`. Wide and fullwidth characters (not Hangul) stand alone, and a walk back of more than 20 characters is dropped for a quote from the cut itself - rather than every Unicode P* character counting as a boundary, which would have split "snow-covered".
- [x] F18. `backend/critic.py:264` A non-UTF-8 install marker makes /api/status 500 on every poll — `85a0412`. A marker that does not decode counts as no install (`ValueError` caught); the review's raw-bytes compare would have re-downloaded every Windows install, whose marker is written with CRLF.
- [x] F19. `backend/critic.py:1586` Critic cleanup can turn a finished critique into a 500 and orphan a server — `fab577c`; `7381c9c`, `9738fb5` for shutdown and Unload. The stop takes the whole tree even after the child has exited - a Windows job object the child joins while suspended (rather than `taskkill /T`), or a POSIX session; the log's removal is guarded, and a kill not waited out is a `CriticError` that lifespan, `handle_exit` and Unload log and go past. On POSIX a hard kill or a terminal hang-up still orphans the server, as documented.
- [x] F20. `backend/prompt_encoding.py:1099` Tokenizer lock held over unbounded Hub I/O blocks Unload and shutdown — `3457a6f`. Loads run on daemon threads outside the lock, concurrent checks share one, and a Check waits at most `LOAD_WAIT` (30 s) - a bounded wait, not a Hub timeout, which huggingface_hub offers only process-wide; Unload never waits.
- [x] F21. `web/sheet.js:157` Sparse comparison sheets pin var:variant instead of giving it the rows — `86298e4`. The variant axis and the axis across it are both chosen from every varying axis - the review's second option, extended to the cross axis, since exempting `VARIANT_AXIS` alone would still pin a sparse sheet whose stars each have their own seed.
- [x] F22. `web/generation.js:475` A finished comparison opens an empty sheet while Favorites is ticked — `86298e4`. A finished comparison unticks Favorites and says so in the sheet note until the next query - the review's first option, not a flag that bypasses the filter.
- [x] F23. `backend/stats.py:51` Documented 'stats --by prompt' comparison recipe cannot tell versions apart — `0597f55`; README `4ab3fa0`. `stats --by variant` gives each version its row and cells fold whitespace; rows still rank by rate, first appearance (`MIN(id)`) breaking ties, and the documented recipe moved from `--by prompt` to `--by variant`.
- [x] F24. `tests/test_critic.py:517` No test pins that the critic sees the negative prompt as applied() runs it — `f9f4ae8`, `f816e11`. Test only, the code was right: `build_context` is pinned both ways with the `check_fits` case, and `f816e11` pins the real `PromptCritic.critique` path, whose forwarding phase 4 found unpinned.

### LOW

- [x] F25. `backend/prompt_encoding.py:638` CLIP control tokens typed in another case go undetected — `3457a6f`. Typed text is compared with the stored control token through the tokenizer's own normalizer; the review's "never classify a control id as unknown" was not adopted, since T5's `<unk>` is itself a control token.
- [x] F26. `backend/prompt_encoding.py:315` NetaYume negative row claims to share its budget with the system instruction — `3457a6f`. `EncoderSpec.role_for(field)` gives a bare negative prompt, NetaYume's, no role - one of the review's two suggested forms.
- [x] F27. `backend/prompt_rules.py:944` NetaYume system-prefix rule finds its span in lower(), shifting offsets — `ad94ac3`. Case-insensitive search on the original text, whitespace tidied only at the cut; beyond the review, a comma on either side of the cut becomes one.
- [x] F28. `backend/prompt_rules.py:678` Negation rule warns on descriptive 'avoid' and on titles such as 'Dr. No' — `ad94ac3`, `46b721e`, `14f0e24`. "avoiding" is no head, "avoid" and a capital "No" ask for nothing mid-clause - after a word or a title's abbreviation, spaced or not ("Dr.No"), since `14f0e24` only one that goes before a name (Dr., Mr., Mrs., Ms., Mt., Prof.), so "Main St. No cars." warns again - and a negated phrase is extended through a run of capitals rather than trimmed. `ad94ac3` also read "No" before any title-case word as a name, which silenced real absences; `46b721e` removed that, so the review's title-case skip is out, at the documented cost that a title opening a clause warns.
- [x] F29. `backend/critic.py:1186` 'X before Y' order issues are read as 'X first' and dropped as conflicting — `f9f4ae8`. "X before Y" is read as a pair; pairs and firsts the documented order reverses are left out on their own before the rest are compared, and a negated instruction asks for nothing. "X after Y" is not read.
- [x] F30. `backend/critic.py:967` JSON reorder check accepts 1.0 to 1 or true to 1 with no key moved — `e563622`, `f816e11`. A JSON reorder must equal the prompt with its keys sorted and differ unsorted, a repeated key refused; beyond the review's `json.dumps(sort_keys=True)`, `f816e11` compares numbers as written (`_Written`), so 0.50 written 0.5 beside a key move is set aside.
- [x] F31. `backend/critic.py:1228` A wrapped rewrite with a period after the closing quote is dropped, not unwrapped — `e563622`, `f816e11`. A wrapped rewrite is unwrapped whatever sentence punctuation follows the closing mark, spaced or not: a stop stays, closed up and added only where the text has none, and a comma, semicolon or colon goes, where the review said to append it. A wrapped copy of the prompt is dropped and said so.
- [x] F32. `backend/critic.py:1402` Reordered echo differing only in case or punctuation reported as 'changed words' — `e563622`. The note follows `is_reorder`'s own normalisation, which also folds how quotation marks, apostrophes and dashes are typed, so an echo differing only in those, in case or in a closing stop is silent.
- [x] F33. `backend/critic.py:1069` Validation removes over-length rewrites and nulled parts without a set_aside note — `e563622`; README `4ab3fa0` and `39e9863`. Lines for nulled parts and text to render, an over-length rewrite or reorder and an unchanged rewrite (the review's first option); the README now names what is still dropped without one - a reordering that moves nothing, and a seventh issue only a server ignoring the schema can send.
- [x] F34. `web/analysis.js:607` The critic's reordered can be any permutation but is offered as 'the documented order' — `3064f9b`, `e563622`, `9738fb5`. The page's tooltip no longer claims the documented order, and the server decides: where Check placed every item only its own reorder stands; where it placed only some (the page's "Partly checked", one predicate since `7cbb55f`), a reorder that keeps the comma items whole and that the order rule flags nothing in stands as "allowed" - the review's alternative, applied only there.
- [x] F35. `backend/prompt_sources.py:943` G sentences link to the guide's page even where their source is another page — `d18c2fa`; design notes `39e9863`. A sentence links the page of the analysis entry it cites, `sentence_urls` only for uncited ones, and summary sentences were split to carry one page each - the review's two options combined - so rule and critic links went from 15 mismatches to 0. The design notes now say which guides the per-sentence test pins.
- [x] F36. `backend/prompt_sources.py:108` Sentence splitter fuses WAI's negative-prompt recipe with an app statement — `d18c2fa`. The recipe is quoted and closed with a stop, and the app's sentence stands alone, unsourced.
- [x] F37. `backend/critic.py:659` Token estimate undercounts some scripts; check_fits and max_tokens misjudge fit — `fab577c`. The server counts the question itself (`/apply-template`, `/tokenize`, after the listener check) and `check_fits` and `max_tokens` use that count; a per-script estimate, scaled by 0.6, only refuses before a server what cannot fit - rather than the review's one token per non-ASCII character or a fetched `tokenizer.json`.
- [x] F38. `backend/critic.py:853` Typed chat-control text reaches llama-server unescaped — `e563622`. Chat-control text in both messages gets a zero-width joiner after its `<`, taken out of the answer exactly; finding lines are defused like the rest rather than reworded, and the optional Check warning was not added.
- [x] F39. `backend/critic.py:398` An interrupted first-use install leaks its extraction work directory — `85a0412`, `41300f0`, `9738fb5`. The critic's install and (`41300f0`) the four model converters sweep leaked work directories under their locks on every call; instead of the review's `<target>-*` glob, each takes only tempfile's exact name as a real directory holding only what it makes (the critic's since `9738fb5`), so a user's `-backup` copy survives. On 2026-09-29 the real `single_file.sweep_work` on this machine's cache removed the three empty work directories interrupted conversions had left on 2026-09-17, and nothing else.
- [x] F40. `backend/critic.py:449` GGUF hash-mismatch advice 'delete it and retry' loops on a symlinked HF cache — `85a0412`. Both of the review's alternatives: the cached entry is unlinked and downloaded once with `force_download=True`, since forcing alone does not recover a Hub cache without symlinks, and the error names the resolved file.
- [x] F41. `backend/config.py:34` A symlinked LOCAL_DIFFUSION_CRITIC_MODEL is labelled with its blob hash — `85a0412`. `_env_path` makes the path absolute without following links.
- [x] F42. `backend/critic.py:1554` A configured critic server that fails to start returns a bare 500 — `fab577c`. An `OSError` from the spawn is a `CriticError` (502) naming the setting; the review's X_OK check in `_server_path` was not added.
- [x] F43. `backend/critic.py:1560` The 401 probe does not prove the listener is the critic's own server — `fab577c`. Before the key goes out, the port's listener must be the server's own process tree (the TCP table on Windows, /proc/net/tcp under Linux and WSL), and nothing visible listening refuses; no Unix socket, since urllib has no transport for one.
- [x] F44. `backend/critic.py:227` Linux critic needs libgomp1 and glibc 2.38, but nothing checks or says so — `85a0412`, `fab577c`. A glibc below 2.38 hides the pinned build, `preflight` names a missing library's apt package before any download, `wsl/setup.sh` prints the `sudo apt install` line rather than running it, and an exit 127 names the package; a missing library keeps the critic offered.
- [x] F45. `web/analysis.js:496` Critic status line names Qwen3-4B and a llama.cpp download under overrides — `7381c9c`. `/api/status` sends a `CriticStatus` (label, pinned, what is still to fetch and its size); an override keeps its label and loses the time and GPU claims. The real app on 2026-09-29, installed, sent `{"label": "Qwen3-4B Q8_0", "pinned": true, "fetch": [], "fetch_bytes": 0}`.
- [x] F46. `backend/main.py:926` Re-render drops a comparison's variant labels from the new rows — `0597f55`, `72f41d7`. Re-render carries each row's variables; beyond the review, `72f41d7` carries them through `generate --from-image` too, only while the prompt pair is the image's own, a comparison version through `expand(variants=...)`.
- [~] F47. `web/sheet.js:125` Sheet variable order depends on which images survive filtering or deletion — `86298e4`; docs `4ab3fa0`. Under Favorites or a search narrower than the run the sheet reads the order from the whole run (`learnSheetOrder`), pinned in `smoke.json`. Missing: rows Delete unstarred removed still move a value down for the run - the review's declared-index column was not added, and the gap is documented, the review's stated minimum.
- [x] F48. `web/analysis.js:515` Critic pre-check holds nothing, and a Check click during it drops the critique — `3064f9b`. The critic's hold is set before its pre-check and Check is disabled meanwhile, both parts of the suggestion; a replaced pre-check says nothing was sent.
- [x] F49. `web/analysis.js:23` The comparison's Seeds count can disagree with the Seeds select — `3064f9b`. Both of the review's alternatives: the payload reads the select, and Seeds and Your version get `autocomplete="off"`.
- [x] F50. `web/analysis.js:133` Changing reference images does not mark the check stale — `3064f9b`. The check records its references and a change marks it stale, badged "references changed since"; Reuse reaches that through `syncCfg`, so it needed no call of its own.
- [x] F51. `web/analysis.js:62` Cut link excerpt can end in half an emoji — `3064f9b`. The excerpt slices by code point, as suggested, not by grapheme cluster, so a ZWJ sequence at the cut splits into whole emoji.
- [x] F52. `web/analysis.js:490` Ask the critic is disabled while it runs, dropping keyboard focus to <body> — `3064f9b`. The button is aria-disabled while the critic runs, and focus moves to the answer when it lands - the review's first option, not a live-region announcement.
- [x] F53. `web/analysis.js:151` 'Check again' hides itself when the re-check lands, dropping focus — `3064f9b`. Focus moves to the panel heading (`tabindex="-1"`), as suggested.
- [x] F54. `web/tags.js:522` Crossing the 901 px breakpoint re-appends the check panel and drops focus — `86298e4`. Focus and selection are saved and restored around the move, with `moveBefore` where the browser has it, as suggested.
- [x] F55. `web/analysis.js:102` Two-column Check leaves focus on the button, results 40+ Tab stops later — `3064f9b`. Focus moves to the panel heading, one of the review's two options; beyond it, `scroll-margin-top` clears the sticky top bar.
- [x] F56. `backend/stats.py:125` stats crashes on redirected Windows output when a prompt has CJK or emoji — `0597f55`. `stats`, `clear`, `generate` and, beyond the review, `reindex` reconfigure stdout to UTF-8 with backslashreplace - on Windows always, elsewhere when redirected - and leave a stream without `reconfigure` alone; the block is inline in each, held to one behaviour by a test.
- [x] F57. `tests/test_api_analysis.py:341` No test covers the critique route's post-prepare 409 re-check — `fb889e9`. The review's test.
- [x] F58. `tests/test_api_analysis.py:607` Shutdown order and handle_exit's critic stop are unpinned — `fb889e9`. The order is pinned as critic, jobs, weights, and `handle_exit` is tested on `Server`, moved verbatim to module level; beyond the review, the critic's stop is pinned off the handler's thread.
- [x] F59. `tests/test_api_analysis.py:368` The critic's prepare() running on a daemon thread is unpinned — `fb889e9`. Driven through `httpx.ASGITransport` from pytest's main thread and checked against the `concurrent.futures` registry, since under TestClient every thread is a daemon and the review's check would be vacuous.
- [x] F60. `tests/test_api_analysis.py:86` Check's 'off the loop, never the gate' is unpinned — `fb889e9`. A Check answers while a running job holds the gate, and every tokenizer load, the critique's analysis and the rewrite's lint are recorded off the loop. On the final tree the lint test's loop half is vacuous, since `3457a6f` gave tokenizer loads threads of their own; that lane's `test_check_runs_on_a_daemon_thread` holds the critique's analysis and lint off the loop.
- [x] F61. `backend/main.py:236` 'Never from /api/status' for prompt_encoding is unpinned — `fb889e9`. `/api/status` under every catalogue checkpoint, before and after a load, never reaches the tokenizer loader, and a final Check shows the recorder is the loader Check uses.
- [x] F62. `tests/test_critic.py:174` Install tests never cover bad cached archives, stale stamps or the exec bit — `85a0412`. A wrong-bytes archive of the right size, the stamp's path, size and mtime, and the exec bit, with the chmod recorded on every host.
- [x] F63. `tests/test_critic.py:398` _stop's wait and kill are untested: FakeProcess can never time out — `fab577c`. `FakeProcess` times out a set number of times (`stubborn`) and records each stop step and whether the log was there at each wait.
- [x] F64. `backend/critic.py:529` The critic's loopback proxy bypass is untested — `fab577c`. A fresh copy of the module is loaded under a proxy environment, since the opener reads it at import.
- [x] F65. `tests/test_critic.py:376` _server_env's LLAMA_* filter is pinned only by harmless variables — `fab577c`. The test sets the dangerous variables the review named, plus MODEL, N_PARALLEL and WEBUI.
- [x] F66. `tests/test_critic.py:1214` The unsourced-sentence test is vacuous: no Grounding lists any sentence — `d18c2fa`. A real sentence listed unsourced by monkeypatch leaves the citations and the system message, and the existing test walks every guide's list.
- [x] F67. `tests/test_api_analysis.py:226` The tag lookup's prose gate is untested: test prompts never reach it — `fb889e9`. Comma-separated prose that `category_items` would send is asserted never to reach the tag index, on Animagine and Anima.
- [x] F68. `tests/test_prompt_encoding.py:589` 'measure never raises' is tested only for OSError/ValueError/RuntimeError — `3457a6f`. The four error types over both guards, plus a load failure and Z-Image's template path.
- [x] F69. `tests/test_critic.py:888` validate's dropped count is unasserted where issues are set aside — `f9f4ae8`. `dropped` is asserted where issues are set aside, the measured-length and documented-order paths included.
- [x] F70. `tests/test_critic.py:704` No test answer carries its own cite/url, so server-side resolution is unpinned — `f9f4ae8`. A test answer carries stray cite, url, source and note keys, and the resolved citation wins.
- [x] F71. `backend/prompt_rules.py:1133` _order's XML/JSON and 256-item stand-downs are untested inside check() — `ad94ac3`. One predicate, `_placeable`, for `order_checked`, `reorder_tags` and `_order`, tested inside `check()` on over-cap, JSON and XML prompts and by a property test.
- [x] F72. `backend/schemas.py:423` prompt_variants' 8-item upper bound is untested — `0597f55`. 9 variants are a 422 and exactly 8 are accepted.
- [x] F73. `tools/smoke-analysis.json:115` Declared-order sheet is unpinned, and its stated motivating bug did not exist — `86298e4`; design notes `4ab3fa0`. The false rationale is corrected, and the declared-order step is in `smoke.json` (an owl grid, `size = small | big`) rather than smoke-analysis.
- [x] F74. `tools/smoke-analysis.json:185` The narrow-layout smoke step passes on the initial markup alone — `86298e4`. The frame is laid out wide before it loads and the wide host asserted, then it is narrowed to 760 px and both nodes asserted home, chips first.
- [x] F75. `tests/fakes.py:358` Fake tokenizer budgets for templated Qwen encoders exceed the real ones — `3457a6f`. A run of newlines is one token, as Qwen's BPE makes it, rather than one per newline, so the clean budgets match too (klein's 500).
- [x] F76. `tests/fakes.py:235` FakeCritic skips check_fits, so it answers prompts the real critic refuses — `fb889e9`; docs `4ab3fa0` and `39e9863`. The fake calls the real `check_fits` before its delay, on the unscaled estimate standing in for the server's count; the docs and the fake's comments now say fake and real can differ near the window's edge. No smoke step.
- [x] F77. `docs/verified.md:418` verified.md's Prompt analyser evidence was deleted after the commit — `4ab3fa0` and `39e9863`. No doc cites a scratchpad: what survived is in `%TEMP%/local-diffusion-analyser-validation/` with a sidecar per file and `INDEX.md`, the four prompts are in the doc, and the 2026-09-28 rows' scripts and lane reports are there under `review-2026-09-28/`, copied whole without sidecars. Beyond the review, it says which rows can still be re-checked and how; the download times cannot be. The 2026-09-29 rows have `review-2026-09-28/final-2026-09-29/`, written by the run itself.
- [x] F78. `docs/verified.md:316` The WSL 'GGUF on ext4' critique timing was measured from tmpfs — `4ab3fa0`. The row says the 9.7 s read the GGUF from tmpfs and the default on ext4 was never measured, and the README's `/mnt/c` advice keeps only the measured cost; the measurement the review asked for was not taken.
- [x] F79. `docs/design-notes.md:2224` Design notes base 'about 15 seconds' on an unrecorded non-pinned GGUF run — `fab577c`, `4ab3fa0`. "About 15 seconds" rests on the pinned pair's rows in `verified.md` (8.1-16.0 s, median 10.5; 7.4-22.3 s, median 13.2), builder C's run is kept as history on a different GGUF, and the `MIN_TOKENS` comment cites `verified.md`.
- [x] F80. `docs/verified.md:349` The FLUX slowdown record omits the parent-commit run that clears this commit — `4ab3fa0` and `39e9863`. The parent-commit run is in the row, a paragraph and the design notes: through the CLI, on another prompt, timed by two PNGs' one-second save stamps, "the ~42 s predates the analyser" rather than "clears this commit", and a run that never went through e7d2386's gate wait. Beyond the review, the 2.7x claim is withdrawn: its 15.4 s was an 8-step image.
- [x] F81. `docs/verified.md:50` verified.md mixes MiB, GiB and MB within the same rows — `4ab3fa0` and `39e9863`. VRAM in `nvidia-smi`'s MiB and file sizes in decimal MB/GB with byte counts, row by row; `39e9863` carried it to the intros' "5-6 GB" and "1.3 GB", the design notes and a `critic.py` comment.

## Refuted

- `tests/test_prompt_encoding.py:898` The real-tokenizer test cannot verify Qwen-Image's window or ERNIE's 2047 budget, so REAL leaves them out and every case skips on the machine that has those tokenizers — nothing to do, and nothing was done for it (the same test gained klein's and Z-Anime's clean budgets, under F75). The narrow residue its verifier named is still true: `REAL` has no case for Qwen-Image's Qwen2.5-VL or ERNIE's tokenizer files, while the arithmetic, the constants and the shared Qwen prefix count are pinned.

## Set aside at synthesis

Nothing to do in any of them; each line says where the part that stood went.

- Sub-claim: inverting the unsourced filter goes unnoticed — deleting the filter, the half that stood, is F66's, pinned in `d18c2fa`.
- Sub-claim: unknown-character spans swallow Cyrillic prompts — F10 as filed: CJK text and T5-unknown symbols (`f9f4ae8`).
- Sub-claim: sending the whole multi-line tag prompt to Danbooru exceeds the README's promise — not carried into F13; no change.
- Sub-claim: row 50's baseline was converted by 1024, and the intro's 1.75-1.79 GB range is one reading — F81 fixed the real unit mix.
- Sub-claim: the final-ux/final-fix paths name no session and could never be found — F77 recovered what survived of them.
- Sub-claim: sorting variable-less images first on the sheet is a regression — the parent commit sorted the same way; no change.
- Sub-claim: a normal interpreter exit mid-extraction always leaks the work directory — F39 as filed: an open file on Windows, or a hard kill.
- Sub-claim: more than 8 prompt variants would break the sheet's layout — F72 pins the API's bound.
- Sub-claim: after the critic answers, the next Tab starts from an unhelpful place — F52 moves focus to the answer.
- Sub-claim: two NewBie XML versions always print identical stats rows — F23 folds a cell's newlines.
- Sub-claim: a regression in the route's CriticError mapping would go untested — F76 closed the fake's fidelity gap.

## Follow-ups and re-verification

The lanes found follow-ups beyond their findings, fixed before phase 4:
lettering as one protected region through every tag rule and the reorder, and
Chinese lettering ended at a placing verb (`53ba37b`); the model converters'
leaked work directories (`41300f0`); `generate --from-image` carrying variables
(`72f41d7`); shutdown (`7381c9c`) and Unload (`9738fb5`) going on past a critic
that will not stop; and the critic's sweep narrowed to its own work
directories, one server-side rule for a critic reorder where Check placed only
some tags, and stale comments (`9738fb5`).

Phase 4 re-ran every code finding's failure scenario on `9738fb5` in clean
worktrees, against the base and against mutants of each fix, without trusting
the lanes' reports; a second pass confirmed the six not found fixed. What that
found, and what followed:

- **F28 had regressed.** The first fix read a capital "No" before any
  title-case word as a name, so "Kyoto. No English text.", "No Apple logo" and
  "No People on the beach" lost their warning, and no test pinned the branch.
  `46b721e` treats a capital "No" like "avoid", a name only mid-clause, and
  pins the absences and "Dr.No".
- **F2 was partial.** Noun triggers still quoted description in one click
  ("with the words shimmering across the chest"), and a negation inside a
  participle's words went unwarned: `46b721e`. Its review found the new
  negation stop cutting "do not enter" to "do" and "no pain no gain" to "no
  pain": `bae8d66` stops only where the negation rule itself reads an absence
  (`_absence_at`), and lets the critic read on past that stop.
- **F11 was partial.** Counts given to a part ("add at least 5 words for the
  lighting") were still left out with the false measured-length note:
  `f816e11`. Its review found partitive counts ("only 3 words of the prompt
  are about the lighting") newly left out, gaps within its own class and some
  real claims kept: `7a633d0`, with a test row for each word of the
  part-count reading and all 114 of its single-alternative mutants killed.
- **F24 and F30 were partial, and F31 had gaps.** The real `critique()`
  forwarding was unpinned, a respelled JSON number rode a key move, and a
  rewrite with a spaced stop or a comma after its closing mark was still
  dropped: `f816e11`, which also stopped telling a prose prompt on a tag
  checkpoint "Check places these tags".
- **F47 was partial**, the deletion half, and stays so: its documented gap
  landed in `4ab3fa0`.
- **"Partly checked" disagreed with the critic.** The page still keyed it on
  `categories_detail`, so an unknown tag ("zzkx oc") or a prose prompt's
  clauses showed the order as checked where the critic's `placed_partly` said
  partly. `7cbb55f` made one predicate (`prompt_rules.unplaced_items`) set
  `order_detail` for the page and `placed_partly` for the critic, naming the
  items and why. Its review found one-item prompts and settled unknown tags
  called partly, and a withheld category called unknown: `06ab194` names an
  item only where its slot decides the order (`_undecided`) and a withheld
  category as having no usable one.

A final re-verification on `06ab194`, against `b742a27` and `4ab3fa0` and
with mutants, found F11, F12, F24, F28, F30 and F31 fixed, and F2 still
partial, with one regression from `bae8d66` in F6's class. The docs, F77-F81
among them, were checked against `4ab3fa0`: the corrections that still held
after phase 4 are `39e9863`, and the two about F11 and F28 were dropped,
since phase 4 had made them untrue.

A last fix round took that regression and the residues the re-verification
listed, one lane for the rules and one for the critic's validation. Each
change was reviewed with mutants and a comparison of base and head over the
authors' examples and the tests' prompts, and what each review found was
repaired:

- **F2's regression.** `14f0e24`: for the critic, `rendered_targets` reads a
  participle's words on through a negation to where they would stop without
  it, `ends` True, so "a banner reading love without limits above the stage"
  gives "love without limits" and a rewrite quoting the whole run is flagged;
  the quote and negation rules keep the stop. Its review found the critic
  then taking the negation's own words for lettering (a rewrite quoting only
  "without pictures" passed), and "a poster reading now or never in bold
  type" still cut to "now or": `c7a33b9` carries the rules' stop as `head`,
  before which lettering must start, and lets a negation right after "or" or
  "than" close the lettering.
- **F28's residue, in part.** `14f0e24`: a "No" stays mid-clause only after
  an abbreviation that goes before a name (Dr., Mr., Mrs., Ms., Mt., Prof.),
  so "Main St. No cars.", "Main St.No cars parked" and "Martin Luther King
  Jr. No crowds." warn.
- **F12's residue.** `14f0e24`: a tag and its copy are placed alike under any
  categories and no longer form a pair, so "1girl, zzkx oc, zzkx oc,
  masterpiece" is checked; `_undecided`'s docstring and the design notes no
  longer say a pair `_ordered` keeps may be out of order.
- **F11's residue and the regroup note.** `bb2193c`: the word just ahead of
  "description" decides whose it is, "the prompt reads as long" is a length
  claim, and the note for a critic reorder that splits the comma items says
  "other clauses" on a prose prompt and "other tags and clauses" on a hybrid
  one. Its review found the critic's words for its own version ("keep the
  revised description under 80 words") taken for parts, "your lighting
  description" still the prompt's in a count while a manner read it as a
  part's, and "as long as" read as a length claim: `8931fae`.

The repairs were not reviewed again. The whole suite and the 15 smokes ran
on `8931fae`, and the older alternatives of the length pattern were
re-checked there with mutants (What remains).

## What remains

Open after the last fix round, and not in the docs as stated here:

- **F2's residue.** After a noun trigger, a participle with no preposition of
  place, or an irregular one ("set", "lit up", "woven into it",
  "flickering"), is still a target with `ends` True: no Apply, as the design
  notes say of "set in bold type", but the critic keeps it as text to render
  and does not flag a rewrite quoting it. The negation stop's cost is wider
  than the notes' examples ("love without limits", "dogs not allowed"): "we
  are not open", "gone but not forgotten" and "say no to war" warn as
  absences, and "no shoes no shirt no service" is cut to "no shoes no shirt"
  with a warning on "no service".
- **F28's residue.** A span still ends inside a name broken by a dot or a
  lowercase particle ("no boats in St" for St. Ives, "without tourists in
  Rio" for Rio de Janeiro, "Avoid Dr" for Dr. Smith), and "Dr." for a street
  still reads as a title, so "a house on Elm Dr. No cars." is silent - the
  price of keeping "Dr. No" silent, which the notes do not name.
- **F11's residue.** A word for the whole still reads as a part in "too long
  a full description" and "the recommended length of the full description",
  branches the last round left alone, though the notes make "the full
  description" the prompt's; and the four older alternatives of the length
  pattern - "brief", the shaping verbs ("lengthen", "extend" and the rest),
  "at N words" opening an issue and "an N-word minimum" or maximum - have no
  test row: on `8931fae` each one's mutant still passes `test_critic.py` and
  `test_api_analysis.py`.
- **Restatements the critic's validation does not catch**, seen in the real
  run on the final tree and recorded in `verified.md`'s rows only: an issue
  repeating Check's `guide.negative_prompt` or `guide.quality` finding is
  kept, since `critic._RESTATES` has no pattern for either. The design notes
  list the rules it covers, and neither is among them: their "quality
  boilerplate" is the prose checkpoints' heuristic, not a tag guide's
  missing finish.

Accepted, and worded as trade-offs in the docs:

- **F47:** a skipped first cell, a late resume or rows Delete unstarred
  removed can still move a value down for the run; recording each cell's
  declared index would close all three, and needs a new column.
- **F2's heuristics:** lowercase lettering that opens with a describing word
  ("a sign that says by appointment only") gets no finding; after a noun
  trigger, lowercase lettering shaped like a participle of place ("with the
  words dancing in the rain") gets none, and Title Case ("Happy Birthday")
  loses its one click; lettering holding a negation further in ("a banner
  reading love without limits") warns as an absence until it is quoted or in
  capitals, though the critic takes it whole as text to render.
- **F28:** a title that opens a clause ("No Country for Old Men poster on a
  wall") warns as an absence, its span taking in the whole name.
- **F1:** a description that starts with a bare 在 (写着营业中在门口) keeps a
  wrong Apply, and a slogan with a placing verb of its own and no request
  before it (我们站在一起) loses its Apply.
- **The critic's edges.** On POSIX only, a hard kill of the app or closing its
  terminal orphans the server, which frees its VRAM after the idle timeout but
  stays alive and listening. Where /proc/net/tcp is empty or incomplete (WSL 1,
  gVisor) the listener check refuses every critique; on other platforms the
  key and the 401 probe are the only guard, and the owner is read once, just
  before the first keyed request. Where Windows refuses a job, the child runs
  alone and only it is stopped. A cached `prepare()` still lets
  huggingface_hub fetch `/api/agent-harnesses`, at most once a day, unless
  `HF_HUB_OFFLINE` or `HF_HUB_DISABLE_TELEMETRY` is set. The estimate is short
  on some scripts (Turkish, Polish, Cantonese, rare Han), which the server's
  count covers once it runs, and the fake refuses on the unscaled estimate, so
  near the window's edge fake and real can differ. The length-claim reading is
  a reading of the critic's wording, so a phrasing unlike the pinned ones can
  go either way, and a word ahead of "description" that `_WHOLE_WORDS` does
  not list names a part ("keep the vivid description under 80 words"). The
  warning for a rewrite that changes only its lettering's punctuation was not
  built, and a reordering that moves nothing or a seventh issue is dropped
  without a line. What survives validation is still the model's suggestion.
- **Elsewhere:** `wsl/setup.sh` prints the `sudo apt install` line rather
  than running it, and the CLI refuses a plain prompt holding a literal
  `{word}`, from `--from-image` too, since every CLI run goes through
  `expand()` (older than this series).

## Commits

In order, oldest first, each corresponding to one or more groups of the
review's fix order (1 lettering, 2 the critic's filters, 3 labels, 4 status
and the critic's lifecycle, 5 comparison and stats, 6 page focus, 7 rule and
encoder edges, 8 test gaps, 9 docs and evidence) or to a follow-up:

| Commit | Fix order | What |
| --- | --- | --- |
| `ad94ac3` | 1, 3, 7, 8 | Rules lane: guessed lettering ends get no Apply (F1, F2), rules leave lettering alone (F3), CJK single quotes (F4), newline and fullwidth-comma tag lists (F13), the NetaYume prefix span (F27), negation heads (F28), one order predicate (F71) |
| `3457a6f` | 4, 7, 8 | Encoder lane: tokenizers loaded outside the lock (F20), the CJK cut snippet (F17), control tokens in any case (F25), NetaYume's negative role (F26), `measure` never raises (F68), the fake budgets (F75) |
| `85a0412` | 4, 8 | Critic install: undecodable markers (F18), leaked work directories (F39), a corrupt cached GGUF (F40), linked paths (F41), glibc and libraries (F44), install tests (F62) |
| `fab577c` | 4, 7, 8, 9 | Critic server: the process tree (F19), the estimate and the server's count (F37), spawn errors (F42), the listener check (F43), exit 127 (F44), stop, proxy and environment tests (F63-F65), the `MIN_TOKENS` comment (F79) |
| `d18c2fa` | 3 | Sources: contradicted and app-voice sentences (F14, F15), each sentence's own page (F35), the WAI recipe (F36), the unsourced test (F66) |
| `0597f55` | 5, 8 | Grid and stats: `stats --by variant` (F23), re-render variables (F46), UTF-8 output (F56), the 8-variant bound (F72) |
| `fb889e9` | 8 | Route tests: the post-prepare 409, shutdown order, daemon threads, Check off the loop, status never measures, the prose gate, the fake's `check_fits` (F57-F61, F67, F76) |
| `3064f9b` | 3, 5, 6 | Page analysis: the Reordered tooltip and Compare (F34), the pre-check hold (F48), the Seeds select (F49), stale references (F50), the emoji excerpt (F51), focus (F52, F53, F55) |
| `86298e4` | 5, 6, 8 | Page sheet: sparse variants (F21), Favorites unticked (F22), declared order under a filter (F47), focus across the breakpoint (F54), the smoke steps (F73, F74) |
| `e563622` | 1, 2, 3, 7 | Critic lettering: the ERNIE single-quote test (F4), whole-word quotes (F5), sure ends (F6), folded punctuation (F7), reorders in the user's text (F8), JSON reorders (F30), wrapped rewrites (F31), echo notes (F32), `set_aside` lines (F33), Check's reorder where it placed the tags (F34), control text (F38) |
| `f9f4ae8` | 2, 3, 8 | Critic restated: quotes near a finding (F9), encoder spans (F10), length claims (F11), whose words a citation is (F16), the negative prompt test (F24), order pairs (F29), `dropped` and citation tests (F69, F70) |
| `7381c9c` | 3, 4 | Labels and status: Partly checked (F12), the critic's status line (F45), shutdown past a critic that will not stop |
| `53ba37b` | 1 | Follow-up: lettering one item through every tag rule and the reorder (F3), Chinese lettering ended at a placing verb (F1) |
| `41300f0` | 4 | Follow-up: the model converters' work directories (F39) |
| `72f41d7` | 5 | Follow-up: `generate --from-image` carries variables (F46) |
| `9738fb5` | 3, 4 | Follow-ups: Unload past a critic that will not stop (F19), the critic's sweep takes only its own work directories (F39), one rule for a critic reorder under partial placement (F34), stale comments |
| `4ab3fa0` | 9, and every group's docs | Phase 3: README, CLAUDE.md and AGENTS.md, the design notes and `verified.md` brought up to the fixes; the analyser's evidence recovered and cited (F77-F81) |
| `46b721e` | 1, 7 | Phase 4: a clause's own capital "No" warns (F28); after a noun trigger only capitals get the one click, a participle of place is no lettering, and a participle trigger's words stop at a negation (F2) |
| `bae8d66` | 1 | Phase 4 repair: a participle's words stop only at a negation the negation rule reads there (F2) |
| `f816e11` | 2, 3, 8 | Phase 4: part word counts (F11), JSON numbers as written (F30), any trailing punctuation unwrapped (F31), the real `critique()` test (F24), the order note's wording |
| `7a633d0` | 2 | Phase 4 repair: F11's regression, gaps and over-correction |
| `7cbb55f` | 3 | Phase 4: "Partly checked" from the placement the critic reads (F12) |
| `06ab194` | 3 | Phase 4 repair: only where a missing category decides the order; a withheld category named as unusable (F12) |
| `39e9863` | 2, 3, 8, 9 | Corrections from checking the docs against `4ab3fa0`, those still true after phase 4: the fake critic's refusal (F76), what `set_aside` leaves unsaid (F33), the per-sentence link test (F35), the evidence's re-check and the 2026-09-28 folder (F77), the parent-commit run (F80), units (F81) |
| `14f0e24` | 1, 3, 7 | Last fix round: lettering holding a negation ends for the critic where its words would without it (F2), "No" after St., Jr. or Sr. opens a clause (F28), a tag's own copy is no pair in "Partly checked" (F12) |
| `c7a33b9` | 1 | Last fix round repair: lettering read on through a negation starts before the rules' stop, and a negation after "or" or "than" closes the lettering (F2) |
| `bb2193c` | 2, 3 | Last fix round: a part named with its description keeps its count, "the prompt reads as long" a length claim (F11), the regroup note names clauses on prose and hybrid prompts |
| `8931fae` | 2 | Last fix round repair: the critic's words for its own version and every determiner in both readings, plurals alike, and "as long as" no length claim (F11) |
| (this commit) | — | This checklist |
