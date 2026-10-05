# Seven-area review — 2026-10-05

A review of seven areas the 2026-10-04 UI review never reached, run on 2026-10-05
against `e63f004` (main, with that review implemented), and a record of how each of
its findings was fixed on the branch `review-2026-10-04-leftovers`. That review's
completeness critic had listed these areas as gaps it would not brief, and the check of
what the review left (`docs/review-2026-10-04-ui.md`, "Leftovers — 2026-10-05, second
pass") named them again:

- **ERNIE-Image Turbo**: Expand, its English translation, and Chinese prompts in
  the prompt check
- **NewBie Exp0.1**: XML prompts in the prompt box, the grid, job names and the
  critic
- **Z-Anime Base and NetaYume v4**: sizes, Mu at a fixed shift, the negative
  prompt, the examples and Refine
- **Anima Turbo v1.1**: the artist @ check, and a checkpoint with no LoRA, Mu or
  refine pass
- **under 901 px**: one column and narrow windows
- **two tabs** on one server
- **screen reader**: what the page gives one - the accessibility tree, the tab
  order and the live regions

The review itself changed no code. `docs/review-2026-10-05-areas-findings.json` is
the machine-readable version of this file.

**Method.** Seven finders, one per area, drove headless Edge over the DevTools
protocol (`tools/browser.mjs`, or a copy with extra steps) against
`tools/fake_server.py` instances of their own, each with a throwaway gallery, from a
detached worktree of `e63f004`. They worked at 2560x1305, 2048x1040@1.25 and
1400x900. The narrow finder worked at 600x900, 800x900 and 900x700, and at the wider
sizes where a finding reached them. They used real mouse and key events where it
mattered and measured with `getBoundingClientRect` and computed styles.

- Some ran the fake through a scratch wrapper that left `tools/fake_server.py`
  unchanged and added what the stock fake lacks: a slow, failing or over-long Expand
  and a failing translator (ERNIE); the real backend's LoRA refusal (NewBie, Anima);
  artists in the fake tag index and an index that fails for chosen tags (Anima).
- The two-tab finder drove two windows of one browser, sharing `localStorage`.
- The screen-reader finder read the accessibility tree
  (`Accessibility.getFullAXTree`), walked Tab, and recorded every write to a live
  region with a MutationObserver. No real screen reader (NVDA, JAWS, Narrator) was
  run, so what would be spoken is inferred from those writes.

Each finding went to a skeptical verifier, one per area. The verifier reproduced it
on a fake of their own, checked it against CLAUDE.md and the design notes, tried the
fix in the page where possible, and wrote a verdict, severity and recommendation. A
merge step folded together findings that two areas had both made, and grouped the
rest into nine fix packages. That is 15 agents in all.

Two process slips touched the review worktree, never the main checkout. The ERNIE and
NewBie finders' first wrappers imported `backend.config` before setting their
`LOCAL_DIFFUSION_*` paths. For about a minute (ERNIE) and 25 s (NewBie) each wrote
`local-diffusion.db` and four files in `outputs/` there, all gitignored. Both finders
deleted them, as NewBie's did two screenshots its first run saved there. Afterwards
`git status --porcelain` was empty in the worktree and the checkout. A first ERNIE run
on a port another agent's server held was abandoned.

| Area | Found | Confirmed | Adjusted | Refuted | Duplicate | Kept |
| --- | --- | --- | --- | --- | --- | --- |
| ERNIE | 15 | 9 | 5 | 1 | 0 | 14 |
| NewBie | 12 | 8 | 4 | 0 | 0 | 12 |
| Z-Anime and NetaYume | 12 | 4 | 8 | 0 | 0 | 12 |
| Anima | 11 | 8 | 3 | 0 | 0 | 11 |
| Under 901 px | 12 | 5 | 6 | 0 | 1 | 11 |
| Two tabs | 11 | 9 | 2 | 0 | 0 | 11 |
| Screen reader | 19 | 8 | 11 | 0 | 0 | 19 |
| **All** | **92** | **51** | **39** | **1** | **1** | **90** |

The 90 kept become **85 entries** once the five made in two areas are folded in (A1,
A24, A56, and two into A25). Severity is the verifiers': **19 medium and 66 low**, none
high. The finders had rated one high (ernie-1, A19), which came down to medium because
its path is rare, and 35 medium. Frequency is the most frequent estimate among an
entry's sources. Entries are ordered by severity, then frequency.

**Set aside.**

- **ernie-8, refuted.** The English translation hides at the first edit, and no image
  keeps an English prompt. This is the design notes' deliberate choice, and it misleads
  no one.
- **narrow-12, a duplicate of the 2026-10-04 review's U15.** Below 1800 px, Examples,
  Check and Reset are 19 px tall and the ? badges 14 px; U15's fix enlarged them in the
  wide tier only, by decision.

**Not reached.**

- a real screen reader;
- the real ERNIE rewriter's output (its language and length), the real Marian
  translator and the 7.7 GB first-use download;
- the real llama.cpp critic: the fake critic cannot write an XML block reorder, so
  those cases were checked in Python and injected;
- the GPU;
- hidden background tabs, with their timer throttling;
- forced colours, reduced motion and touch.

**How it was fixed.** In ten worktree packages, the second wave of the leftovers work.
Each branched from `91b4caa`, the branch after the first wave, where 33 smokes and
1,997 tests passed. Each was implemented, reviewed against CLAUDE.md's rules and for
coverage of its findings, fixed and rechecked (39 agents), then merged:

| Package | Entries | Commits | Merge |
| --- | --- | --- | --- |
| `w2-jobs` | 13: A2, A3, A4, A11, A21, A31, A49, A63, A76, A77, A78, A79, A85 | `a4dfa2f`, `7e3a2ff`, `97e960a`, `5306823`, `5c11d48`, `d1c11ee` | `8bc6efe` |
| `w2-expand` | 12: A6, A8, A14, A15, A16, A17, A19, A32, A59, A60, A61, A66 | `4703441`, `e92c8cc`, `6e18c84`, `8edd609` | `2bedf55` |
| `w2-rules` | 12: A10, A33, A34, A42, A43, A44, A57, A58, A62, A68, A69, A73 | `7d17925`, `a954774`, `9961881`, `a5369e9`, `58db19f`, `5627802`, `e66505e`, `c4d44c3`, `00c1df6`, `6599b49` | `e4ce22e` |
| `w2-sheetlb` | 10: A7, A27, A37, A39, A47, A50, A51, A54, A65, A82 | `8a31d22`, `51b32f9`, `033b6fc`, `1877979`, `bedb13f`, `aba5f6e`, `bf79894`, `20bb5df` | `9a46d62` |
| `w2-controls` | 9: A20, A23, A24, A25, A26, A41, A56, A70, A71 | `8c3326d`, `c039137`, `4573040`, `789519b`, `f4dbd43`, `4a0dd5d`, `d6e1535` | `b8d5352` |
| `w2-refine` | 8: A1, A30, A40, A55, A64, A72, A81, A84 | `72044fa`, `b5dda4f` | `5f2121a` |
| `w2-narrow` | 7: A5, A12, A13, A46, A67, A74, A75 | `2e72181`, `8260559`, `424208f`, `0b237c8`, `5edb718`, `b7740a1`, `ae1b5ae`, `838d190`, `4343213`, `af273e0`, `acfa587` | `f8be93f` |
| `w2-a11y` | 7: A9, A22, A29, A35, A38, A45, A52 | `2287ea9`, `104e98a`, `2a9cee6`, `efa60fd`, `d3efdd0`, `0d42de2`, `6c98b34`, `4547521`, `23fe470`, `252672d`, `6b838fb`, `5e9f362`, `d0d41c6`, `75eaadb`, `bd44548`, `e98aeff` | `aef7022` |
| `w2-gallery` | 7: A18, A28, A36, A48, A53, A80, A83 | `2d8e7d5`, `6fc374e`, `5e720df`, `22558e9`, `83639ba`, `0a61f90`, `971d4f5`, `080cc9c`, `0be449d` | `e1d918a` |
| `w2-integ` | the first wave's leftovers | `a59f46e`, `85a510e`, `df6e0f1`, `7704c4c`, `21aa99d`, `eef855a`, `abeee45`, `58b75e8`, `7a9c3fe`, `7fca06f`, `956c300` | `16bdcca` |

`w2-integ` took the 52 minor points that the first wave's rechecks left:

- The Output caption is the group's hidden name on a two-column page under 1800 px and
  on a wide window up to 1150 px tall; at 1400x900 its line had put Generate wholly
  below the window at scroll 0.
- Four dock rules are scoped away from Gallery beside; with an idle viewer they had let
  the check outgrow its column.
- A narrow portrait's docked result row is as wide as its buttons need.
- The stacked check's cap has a floor of 15% of the window.
- Each remaining point was checked against the merged code.

It also made a smoke `expect` that throws or rejects fail; `browser.mjs` had counted it
as passed (`eef855a`). Four fixes followed the merges: `dcd1d1d`, `230c08e`, `183e001`
and `66bcf14`.

**Outcome.** Of the 85 entries, **84 are fixed and one partly**: A68, whose optional
part was left out by decision. A11 ended its implementation partly done, and its fixer
finished it. A76's last part, Expand's own line saying what holds the card, was taken
in the final pass (K5). An entry's *Later* line names the integration review's findings
(I1-I18, below) that went on in the same place.

The commits listed are each package's implementation and review-fix commits that touch
the entry, plus the first wave's where it did part of the work; the package's merge is
in the table above. CLAUDE.md, AGENTS.md, the design notes and the README were brought
up to all of it afterwards (`c4515b9`, `f9de976`, `a3ab988`, `564f94d`).

## Findings

Each entry gives:

- a header line with severity, frequency, areas, the verdict and the package;
- **Observed**: what the finder measured, as the verifier reproduced it, at `e63f004`;
- **Recommendation**: the verifier's, in short;
- **Fixed**: what changed, and the commits.

A closing line gives the workflow's ids.

### MEDIUM

#### A1. The Refine dialog's Strength/Versions/Seed row overflows its column below 1600 px and at 1700, 1920 and 2048x1040@1.25

**medium** · every-session · Z-Anime and NetaYume, under 901 px · confirmed · `w2-refine`

**Observed.** Every open showed a horizontal scrollbar and clipped the dice: the dialog's scrollWidth 473 against 458 at 600 to 1599 px, 1413 against 1398 at 2048x1040@1.25, 1424 against 1398 at 1920x1080. A regression from `4aeaee7`'s `min-width: max-content` on the Strength sub, which took the text box's default 20-character width.

**Recommendation.** `min-width: 0` (or min-content with a caption that does not wrap) on that sub, the Seed's 16-digit minimum kept, and a no-sideways-scroll expect in smoke-refine and smoke-wide.

**Fixed.** The overflow itself went with `45e75f9` (wave 1: the Strength sub sized to its caption, not its text box). `w2-refine` keeps the caption on one line (`white-space: nowrap`) and holds the dialog to no sideways scroll at 1400x900, 2560x1305 and 2048x1040@1.25 in smoke-refine, smoke-wide and smoke-z-anime. `45e75f9`, `72044fa`.

*zanime-2, narrow-1*

#### A2. The topbar's live status text is rewritten with the same checkpoint name at every 3 s status poll

**medium** · every-session · screen reader · adjusted · `w2-jobs`

**Observed.** On an idle FLUX fake at 2560x1305 `#status-text` (aria-live polite) got a new text node at each poll: 3 rewrites in about 10 s, 3 more over a 9 s two-job run, so a screen reader can repeat the name all session.

**Recommendation.** Build the text first and write it only when it differs, as the critic's line and the tag summary already do.

**Fixed.** `refreshStatus` writes `#status-text` only on a change, with the checkpoint's id as its title. Another tab's "busy in another tab" is an aria-hidden sibling (`#status-aside`) outside the live region, hidden at 600 px and under. `97e960a`, `5306823`, `d1c11ee`.

*a11y-1*

#### A3. The job's progress text, a status region, is rewritten at every denoising step

**medium** · every-session · screen reader · adjusted · `w2-jobs`

**Observed.** A 12-step image wrote '1/12' to '12/12' about 310 ms apart (14 writes); a Repeat 3 run rewrote 'image 1/3' with identical text at every step.

**Recommendation.** Write `#progress-text` only on a change and keep it to coarse states, the step count aria-hidden.

**Fixed.** `setProgressText(spoken, shown)`: a single image's step count is an aria-hidden half beside a hidden "denoising…", each counter is written once, and the mirrors and the tab's title read `progressShown()`. `97e960a`.

*a11y-2*

#### A4. A job's successful end and a job joining the queue are never announced

**medium** · every-session · screen reader · adjusted · `w2-jobs`

**Observed.** After '12/12' the only live change was the progress text emptied. `#finish-note` ('Finished 1 image in 4s, seed … Use seed') and `#queue-note` are not live regions; only failures and cancels reached `#error` (role=alert).

**Recommendation.** One visually hidden polite line, written once per job end and when a job is queued, and nothing where `#error` already says the outcome.

**Fixed.** `#job-said` holds the finish line's sentence without its buttons, and `Queued: <job>` when a press queues one. A job that ends at the load over the viewer's card is said there too. `97e960a`, `d1c11ee`.

*a11y-3*

#### A5. In one column a job's progress, finish line and result land below the window, and New image is never offered for a viewer below it

**medium** · every-session · under 901 px · confirmed · `w2-narrow`

**Observed.** At 800x900 from the page's top, Ctrl+Enter left the live tile at y 1077-1719 and the result at 1022-1664, wholly below, with New image hidden. Offered for a viewer above, the button sat in the static panel, as far out of view.

**Recommendation.** Under 901 px a fixed pill, offered for a viewer below too ('New image ↓'); its press still does the only scroll.

**Fixed.** `viewerOutOfSight` answers above or below and the button's arrow follows a scroll that jumps past the viewer. Under 901 px it is a pill centred under the sticky topbar, withdrawn when the job leaves nothing to show and on the user's return to the panel; a press before an image has landed focuses the live tile. Adds `tools/smoke-narrow.json` (7889). `5edb718`, `ae1b5ae`, `af273e0`, `b7740a1`.

*narrow-3*

#### A6. An expansion's result is squeezed into the sticky panel

**medium** · every-session · ERNIE · confirmed · `w2-expand`

**Observed.** A 330-character expansion with a 1,240-character translation: the four-row English box showed 18-24% of its text, the prompt box was scrolled to its end, and Generate went below the window at 2048x1040@1.25 (1086-1130 in 1040) and at 1400x900, while the results column stood idle.

**Recommendation.** Caret and scroll at the start; from 1800 px the translation at the head of the results column at its own height.

**Fixed.** Both boxes start at their top. From 1800 px `placeTranslation` moves the English to the head of the results (up to 40vh), and beside the docked check into the image column's first row; below 1800 px it stays under the prompt. `e92c8cc`, `8edd609`, `230c08e`.

*ernie-4*

#### A7. An XML prompt's line breaks are collapsed into one paragraph in the result column and the lightbox

**medium** · every-session · NewBie · adjusted · `w2-sheetlb`

**Observed.** NewBie's example ran together (`<character_1> <n>hatsune_miku</n> <gender>1girl</gender>…`). At 2048x1040@1.25 the result column's prompt was 798 px tall and pushed the facts below the column's bottom; the lightbox's the same.

**Recommendation.** `white-space: pre-line`, capped at a few lines with its own scroll, since pre-line alone makes the block about 25% taller.

**Fixed.** `.result-prompt` and `#lb-prompt` keep their line breaks and stop at `--prompt-cap` (twelve lines), scrolling inside themselves; four lines in the one-column lightbox. `51b32f9`.

*newbie-3*

#### A8. Expand is held with `disabled` and no reason

**medium** · common · ERNIE · confirmed · `w2-expand`

**Observed.** A click left focus on `<body>` within 300 ms and after a failure; during a job or a critique a press did and said nothing.

**Recommendation.** aria-disabled with the reason as its title, and a press that says it, as every other held button.

**Fixed.** `syncExpandButton`, its one writer, never sets `disabled`: the reason (expanding, the server not answering, a job queued or running, the critic, another tab's job) is its title and a press says it, while running what will become of the result. Focus stays on Expand through the request. `e92c8cc`, `8edd609`, `d1c11ee`.

*Later:* K3 (each reason said as a sentence), `936ba28`; I11 (a press held by this tab's job is a muted note), `936ba28`.

*ernie-3*

#### A9. Checkpoints without tag chips get no live token count or cut

**medium** · common · NewBie · adjusted · `w2-a11y`

**Observed.** NewBie's own two-character example fills 488 of Gemma's 511 tokens; a three-character version is cut at code point 1464 inside character 3's `<clothing>`, so `<general_tags>` is never read, and nothing says so until Check. FLUX's T5 and the Qwen and Gemma encoders were the same.

**Recommendation.** Run the measure without chips too and show one line under the box: amber when over, 'cut unknown' for an uncounted encoder, never 'no cut'; not in a live region.

**Fixed.** `#prompt-tokens` under the box gives the tightest encoders' count from three quarters of the budget, amber past it and naming where reading stops; the crossing is said once in `#tag-summary`. It is asked again after an offline spell and on a checkpoint change, hidden where no tokenizer is known, and its showing or going waits while the pointer is on a control below it. `efa60fd`, `23fe470`, `252672d`, `5e9f362`, `d0d41c6`, `75eaadb`.

*Later:* I7 (past 4000 characters the first 4000 are measured), `936ba28`.

*newbie-2*

#### A10. On Anima a prompt of tags plus sentences reads as prose, so the artist @ check never runs and nothing says so

**medium** · common · Anima · confirmed · `w2-rules`

**Observed.** 'masterpiece, best quality, score_7, safe, 1girl, wlop, A girl stands in the rain…' gave 'Nothing to flag', with `categories_detail` null.

**Recommendation.** Ask the index about the tag-shaped items on a guide whose rules read artists, or at least say the check was skipped.

**Fixed.** `prompt_rules.lookup_items` asks the index about a prose prompt's tag-shaped items (never its clauses) where `wants_artist_categories`, and `_artist_prefix` reads each comma item that is not a sentence or lettering. `58db19f`.

*anima-1*

#### A11. A tab not watching a job never learns that another tab started one

**medium** · common · two tabs · confirmed · `w2-jobs`

**Observed.** With tab A making two images, tab B read ready, held nothing (Unload, Ask the critic, Expand) and said nothing; its Unload then said 'a job is still running; cancel it first' with no Cancel in view, and its gallery and Runs picker missed A's images until a reload.

**Recommendation.** The server's job state on the status poll, kept as a page flag rather than attaching to the job.

**Fixed.** `ModelStatus.active_jobs`; `elsewhereBusy` holds Unload, the critic and Expand with the reason, says it in the topbar beside the name, and leaves Generate live; at that job's end the gallery and the Runs picker catch up. `a4dfa2f`, `97e960a`, `5306823`, `d1c11ee`.

*Later:* I3 (no catch-up redraw under an open sheet), `936ba28`; I5 (the catch-up owed after this tab's own empty job), `936ba28`.

*twotabs-1*

#### A12. Jobs and runs are named by an XML prompt's opening markup, so NewBie jobs that differ in an attribute read alike

**medium** · occasional · NewBie · confirmed · `w2-narrow`

**Observed.** Three queued jobs differing only in hair colour read `1 image · 1024×1024 · <character_1> <n>hatsune_miku</n> <gender>1girl</gender>…` in the Now row, the queue rows, Cancel's title and the Runs picker.

**Recommendation.** Strip XML tags before cutting; keep the prompt as written in the titles.

**Fixed.** `promptGist` drops `<name>`/`</name>` tags (prose, Qwen's control text and `<lora:…>` stay) for the Now line, the queue rows, Cancel's title and the Runs labels; the Now line's title keeps the markup. `424208f`, `838d190`.

*newbie-7*

#### A13. A grid variable inside a long multi-line prompt makes the grid preview print each resolved prompt whole

**medium** · occasional · NewBie · confirmed · `w2-narrow`

**Observed.** {hair} with three values in NewBie's 29-line XML grew the preview to 2,264 px at QHD, and Run grid sat at y 3462-3510 in a 1305 px window.

**Recommendation.** List only the lines that change, whole, in a capped box of its own after `.totals`.

**Fixed.** A prompt written over lines is listed by its variable lines, whole and numbered ('prompts, line 4 of 29:'), in a box after the axes line at most eight lines tall that scrolls inside itself; `.totals` stays the first child. `8260559`, `838d190`, `acfa587`.

*newbie-1*

#### A14. The first Expand silently starts a multi-gigabyte download and load

**medium** · occasional · ERNIE · confirmed · `w2-expand`

**Observed.** ERNIE's ~7.7 GB rewriter and then the 312 MB translator: the page showed no model state, clock, cost or cancel, and a press on the topbar's Load meanwhile erased the 'Expanding…' line.

**Recommendation.** An expander entry on the status from file checks, as the critic's `describe()` has; say the first-use cost and what Expand waits on.

**Fixed.** `ModelStatus.prompt_expander` (loaded, and what a first use fetches, from file checks). Expand's title names the cost; while it runs the line under the prompt names each step in order with an aria-hidden clock; a failed load answers 503 with the status's sentence; the translator loads one copy, not a second 312 MB conversion. `4703441`, `e92c8cc`, `6e18c84`, `8edd609`.

*Later:* K5 (Expand names another tab's critic or Expand holding the card), `936ba28`.

*ernie-6*

#### A15. After an Example replaced the prompt, an Expand leaves the Example's Undo up, and neither it nor Ctrl+Z brings the user's prompt back

**medium** · occasional · ERNIE · confirmed · `w2-expand`

**Observed.** Expand assigned `.value`, which empties the undo history; the Example's Undo then did nothing.

**Recommendation.** Write through `writeBox` and offer Expand's own Undo.

**Fixed.** `25f0a01` (wave 1) writes the result as typing would. `w2-expand` writes it through `rewriteBox` and offers 'Prompt expanded: …' with an Undo that restores the replaced prompt and its English; an Example's Undo the running line displaced is offered again if the expansion fails. `25f0a01`, `e92c8cc`, `6e18c84`.

*ernie-2*

#### A16. Each tab writes the whole panel to its store, so one change in an older tab erases another tab's prompt, grid and steps

**medium** · occasional · two tabs · confirmed · `w2-expand`

**Observed.** Tab B typing a seed stored prompt '', grid steps '', the fold closed and Steps unset over tab A's 87-character prompt, '20,28,36' and Steps 40; A reloaded to an empty panel.

**Recommendation.** Merge: write only the keys this page changed since it read the store.

**Fixed.** `writePanel` writes the keys this page changed (`panelBase`) over what the store holds; an empty store takes the whole panel; after a Reset the store stays forgotten until the panel changes. `e92c8cc`, `6e18c84`.

*twotabs-3*

#### A17. Reset's Undo is written below the window at 1400x900 and in one column, and Ctrl+Z recovers nothing

**medium** · occasional · under 901 px · adjusted · `w2-expand`

**Observed.** The Undo sat at y 941-963 at 1400x900 and 891-913 at 800x900, out of view; `resetPanel` emptied the boxes with `.value = ''`, so Ctrl+Z in the prompt gave back nothing.

**Recommendation.** Say it under the prompt, where Reset was pressed, and empty the boxes through the undo stack.

**Fixed.** Reset's note and Undo go under the prompt (`#prompt-note`), 'Reset undone' too, and `rewriteBox` empties the boxes, so Ctrl+Z restores them. `e92c8cc`, `66bcf14`.

*Later:* K1 (the note in the emptied box's corner at every width, Generate unmoved), `936ba28`; I12 (Reset also clears a Reuse's note), `936ba28`.

*narrow-4*

#### A18. Delete unstarred counts and deletes images another tab made that this tab has not shown

**medium** · occasional · two tabs · confirmed · `w2-gallery`

**Observed.** Tab B showing 18 cards asked for 20 to be typed after A made two more; its running-job line and the 409's hint keyed on B's own busy, so both were missing when the job was A's.

**Recommendation.** Bring the grid up to date first and say how many are new; ask the server about jobs.

**Fixed.** Outside a sheet the grid is brought up to date before the question, which says how many of the count were made since the page loaded, counted by the server (`newer_than`); the server's `job_active` drives the running-job line and the 409 says a job is running; focus never falls to `<body>` when the reload hides the button. `6fc374e`, `0a61f90`, `080cc9c`.

*twotabs-4*

#### A19. An Expand that ends replaces a live Reset or Reuse Undo in the panel's line

**medium** (finder: high) · rare · ERNIE · adjusted · `w2-expand`

**Observed.** Reset 0.3 s into a delayed Expand: 'Panel reset. Undo' was replaced by 'The prompt changed while expansion was running…' with no button while the snapshot was still held, and Reset had already forgotten the store, so the panel was lost. Reuse the same. The verifier rated it medium, the path being rare.

**Recommendation.** Do not write `#error` over a live Undo; say nothing where a Reset, Reuse or Example replaced the prompt on purpose.

**Fixed.** Reset's Undo moved under the prompt (A17). Expand's messages go through `expandSay`, under the prompt while a Reuse's Undo holds the panel line, and a result whose box was rewritten whole while it ran is dropped without a word. `e92c8cc`, `6e18c84`.

*Later:* I10 (a press on Expand held by this tab's job), `936ba28`.

*ernie-1*

### LOW

#### A20. Where CFG is the knob, the live negative prompt's caption still says '~2× slower'

**low** · every-session · Z-Anime and NetaYume · confirmed · `w2-controls`

**Observed.** Z-Anime at CFG 4 read '(needs CFG > 0 — ~2× slower)', NetaYume at 5 the same; their pipelines already run the unconditional pass with an empty negative at that CFG, so a typed one costs nothing.

**Recommendation.** Keep the greyed wording and True CFG's; a live negative where CFG is the knob says it runs at no extra cost.

**Fixed.** '(runs at CFG > N, no extra cost)', kept to one line beside the label; the grid case '(for grid cells above CFG N)'. `4573040`, `d6e1535`.

*Later:* I17 (the Refine dialog's negative caption worded the same), `936ba28`.

*zanime-8*

#### A21. The page names checkpoints by repository name, which hides the pinned version

**low** · every-session · Z-Anime and NetaYume · adjusted · `w2-jobs`

**Observed.** NetaYume v4 showed as 'NetaYume-Lumina-Image-2.0', Z-Anime Base as 'Z-Anime' and Anima Turbo v1.1 as 'Anima', in the topbar, the check's head and the facts.

**Recommendation.** `ModelStatus.model_label` from the start-up catalogue by exact id, shown where the running checkpoint is named.

**Fixed.** `model_label`; `checkpointName` names the running checkpoint in the topbar, the prompt check's head, the facts' Model row and the load card, with the id in their titles; stored ids are unchanged. `a4dfa2f`, `97e960a`.

*Later:* I13 (the Reuse notes, Expand's line and the Refine dialog too), `936ba28`.

*zanime-10*

#### A22. The page has no h1, and the settings panel is an unnamed landmark

**low** · every-session · screen reader · confirmed · `w2-a11y`

**Observed.** Idle, the tree held a banner, main, an unnamed complementary region and the heading 'Gallery (21)'; the brand was a div.

**Recommendation.** The brand as the h1; a named panel; a name for the results.

**Fixed.** The word mark is the h1, read as 'Local Diffusion'; the panel is a section named by a hidden 'Settings' heading; the viewer is a region named Results. `104e98a`, `0d42de2`, `e98aeff`.

*a11y-12*

#### A23. ERNIE's guide carries none of Baidu's recommended sizes, so every 1 MP preset reads as in the guide

**low** · every-session · ERNIE · adjusted · `w2-controls`

**Observed.** All 13 presets were marked, 12:5 titled 'within the size range this checkpoint is documented for', and Baidu's own 768x1376 typed in was not recognised.

**Recommendation.** `aspects` and `resolutions` from the card's table.

**Fixed.** Both ERNIE guides carry Baidu's seven sizes and their framings (the 9:16 preset standing for 768x1376); 12:5 reads 'not a recommended aspect', a typed size in a guided framing that is not one of them 'not one of the recommended sizes', and a typed size's framing is the nearest preset within 3%. `c039137`, `4573040`, `789519b`.

*ernie-10*

#### A24. The fake server claiming Anima or NewBie accepts LoRAs and keeps a Mu that the real backend refuses

**low** · every-session · Anima, NewBie · confirmed · `w2-controls`

**Observed.** /api/loras listed adapters and a Generate with an adapter and mu 0.8 recorded both, so no browser check saw the refusal and the fake's rows carried what the real server never records.

**Recommendation.** One id-based refusal that the backend and the fake share.

**Fixed.** `loading.lora_refusal(model_id)` from the loaders' own MODEL_IDs; the fake takes its refusal and the profile's `mu_supported`, and drops a Mu the claimed schedule cannot take. `8c3326d`.

*anima-11, newbie-10*

#### A25. The size line calls the authors' example sizes a documented bound, quotes bounds to one decimal, and Anima's tip leaves out its 1.5 MP tier

**low** · common · Z-Anime and NetaYume, NewBie, Anima · adjusted · `w2-controls`

**Observed.** NetaYume's 1440x1440 read 'above the 1.6 MP this checkpoint is documented for' beside a tip saying 1.57 MP is no maximum; `fmtMp` turned 1.05 into 1.0, 1.57 into 1.6 and 2.36 into 2.4. NewBie and WAI the same.

**Recommendation.** A `pixels_word` for example spans and the size line worded from it; two decimals where one changes a bound.

**Fixed.** `pixels_word` 'example' on NetaYume, NewBie and WAI, with `pixels_source` naming whose sizes they are ('the authors' example sizes', 'the author's example', 'the author's reference sizes'); bounds quoted to two decimals at most; Anima's size prose names its 1.5 MP tier. `c039137`, `4573040`, `789519b`.

*zanime-6, newbie-8, anima-8*

#### A26. The grid's Steps and CFG placeholders are Krea 2 Turbo's on every checkpoint

**low** · common · Z-Anime and NetaYume · confirmed · `w2-controls`

**Observed.** '6-12x2' and '0,0.5,1' on Z-Anime (40 steps, CFG 4) and NetaYume, where CFG 0 turns guidance and the negative off.

**Recommendation.** The profile's neighbourhood, as the grid's Mu placeholder already is.

**Fixed.** `gridExamples` gives round(.75s), s, round(1.25s) and c-1, c, c+1 (1, 2, 3 where the profile's CFG sits at its own threshold, and none where the CFG box never reaches the model, `cfg_ignored`: klein, schnell, Qwen-Image). `4573040`, `f4dbd43`, `4a0dd5d`.

*zanime-11*

#### A27. Every image's name and description both carry the whole prompt

**low** (finder: medium) · common · screen reader · adjusted · `w2-sheetlb`

**Observed.** An 81-word prompt's card image had a 463-character name and a 481-character description, so the prompt was read twice per image.

**Recommendation.** The description from the title's first line: the facts without the prompt.

**Fixed.** `titleImage` sets `aria-description` to the facts, leaving out what the name already says; the title under the pointer keeps the prompt. `033b6fc`, `20bb5df`.

*a11y-4*

#### A28. A gallery card is an unnamed div, so its tools are announced without the image they act on

**low** (finder: medium) · common · screen reader · adjusted · `w2-gallery`

**Observed.** Star, Reuse, Refine, Reference and Sheet named no image, and focus handed back to ⤢ from the Refine dialog named none.

**Recommendation.** role=group with the label '#N' on cards and viewer results.

**Fixed.** Each card and viewer result is a group named '#N'; a tombstone drops it. `22558e9`.

*a11y-6*

#### A29. Every prompt-check finding and critic issue is a tab stop, acting as a button with no button's role

**low** · common · screen reader · adjusted · `w2-a11y`

**Observed.** Selectable items read as list items titled 'Select these words'; the others were stops that did nothing.

**Recommendation.** A Show button for each selectable item and no list-item stops.

**Fixed.** A Show button after the item's words, described by the item and titled with the box it selects in; the items are not stops; a negative finding's Show is aria-disabled while the box is greyed; Show keeps to its line's height in the wide tier. `2a9cee6`, `6b838fb`, `bd44548`.

*a11y-13*

#### A30. The Refine dialog's scale buttons mark the chosen scale with `.active` only

**low** · common · screen reader · confirmed · `w2-refine`

**Observed.** No scale button had a pressed state, while the panel's size strips set `aria-pressed`.

**Recommendation.** `aria-pressed` beside `.active`; `disabled` past the guard stays the documented exception.

**Fixed.** `aria-pressed` beside `.active`. `72044fa`.

*a11y-11*

#### A31. Generate, Run grid and Run comparison held by an Expand or the critic keep their ordinary titles

**low** · common · ERNIE · confirmed · `w2-jobs`

**Observed.** The reason appeared only after a press, while Unload in the same state gave it as its title.

**Recommendation.** Unload's chain of reasons as their titles.

**Fixed.** `submitHold()` gives the reason (the server, Expand, the critic) as their title while held, and the plain titles return after. `97e960a`, `d1c11ee`.

*Later:* I8 (the Refine dialog's Refine follows the same hold), `936ba28`.

*ernie-11*

#### A32. At 1400x900 'Expanding…' widens Expand and wraps Reset onto a second caption line for the length of the request

**low** · common · ERNIE · adjusted · `w2-expand`

**Observed.** Expand went from 49 to 72 px and the prompt box down 19 px; reserving the width would have made the wrap permanent.

**Recommendation.** Label it 'Expand…' while running.

**Fixed.** 'Expand…' (57 px) while running, with 'Expanding…' in its title. `e92c8cc`.

*ernie-14*

#### A33. The check's summary counts each Han character as a word

**low** · common · ERNIE · confirmed · `w2-rules`

**Observed.** A 300-Han expansion read '300 words', a 12-character Chinese list '12 words'.

**Recommendation.** Send the CJK count beside the word count and say characters.

**Fixed.** `PromptAnalysis.characters` (`prompt_rules.cjk_count`); the summary says 'N characters', 'N words' or both, the larger first. `7d17925`.

*ernie-15*

#### A34. The `$character_2$` slot warning cites the authors' line that names `$character_1$`

**low** · common · NewBie · confirmed · `w2-rules`

**Observed.** One fixed cite was attached to every match of the placeholder rule.

**Recommendation.** Cite the matched slot's own line where it is verbatim in an example.

**Fixed.** `_slot_cite` cites each slot's own text, else the shortest example line holding it. `9961881`.

*newbie-12*

#### A35. With the check docked, Tab reaches the right-hand check before the image just made on its left

**low** (finder: medium) · common · screen reader · adjusted · `w2-a11y`

**Observed.** The image was the 41st stop from the prompt with the check open, the 26th without it.

**Recommendation.** `reading-flow` on the docked stage, and a host of its own for the check.

**Fixed.** The docked stage reads in source order with the check's host moved before the gallery (`reading-order`), the chips and the check each in a host of their own (`#tag-chips-wide`, `#tag-validation-wide`): Tab goes chips, the image and its tools, its note, `#recent`, the check, the gallery. `d3efdd0`, `0d42de2`, `6c98b34`, `4547521`.

*a11y-8*

#### A36. F stars an image with nothing spoken

**low** (finder: medium) · common · screen reader · adjusted · `w2-gallery`

**Observed.** On a card or sheet cell nothing changed for a screen reader; in the lightbox F rewrote the counter and the zoom readout with identical text.

**Recommendation.** One hidden polite line saying 'Starred #N'.

**Fixed.** `setFavorite(rec, value, { say })`, from the three F paths, writes 'Starred #N' or 'Unstarred #N' into a hidden polite line (`#star-said`, or the lightbox's own while it is up). `83639ba`, `0be449d`.

*a11y-7*

#### A37. The contact sheet exposes no table and names its cells only by #N and the shared prompt

**low** (finder: medium) · common · screen reader · adjusted · `w2-sheetlb`

**Observed.** On a steps × seeds sheet every cell was 'Open #16: a teapot…', the steps nowhere; opening it moved no focus and said nothing.

**Recommendation.** Name each cell by its axis values; table roles through `display: contents` rows.

**Fixed.** A matrix is role=table (rows `display: contents`, column and row headers), a wrapped row a list; cells are named 'Open #16 · Steps 2 · Seed 7601…' or by version; each sheet opened is said on a polite line. `1877979`, `20bb5df`.

*Later:* I6 (focus kept across a redraw), `936ba28`.

*a11y-5*

#### A38. The greyed negative prompt and the comparison's always-run Original box give no reason in the accessibility tree

**low** · common · screen reader · confirmed · `w2-a11y`

**Observed.** Both read as disabled with no description; the reason sat in a separate element.

**Recommendation.** `aria-describedby` and `aria-description`.

**Fixed.** The negative is described by `#neg-note`; the Original box reads 'Always runs: the other versions are compared against it'. `104e98a`.

*a11y-16*

#### A39. A sheet opened by a press is scrolled to, but focus stays on the control that asked, out of view

**low** · common · under 901 px · adjusted · `w2-sheetlb`

**Observed.** After Run comparison at 800x900 the next Tab scrolled the page from 2109 back to 348; at 2560x1305 to 0.

**Recommendation.** Move focus to the first cell with preventScroll where the asker is out of sight.

**Fixed.** `showSheetScope` takes the asker (for a job, what was pressed at submission) and, once the scroll settles, focuses the first cell where focus is still on an asker out of sight, in view meaning inside every clipping ancestor; a refine sweep hands its asker on. `1877979`, `20bb5df`.

*narrow-5*

#### A40. When the Refine dialog passes over the default 1.5× because it is past the measured line, it opens at 1× and says nothing

**low** · common · Z-Anime and NetaYume · adjusted · `w2-refine`

**Observed.** Every Z-Anime source from about 1.5 MP, the author's 1024x1536 included, opened at 1× with no word about why; only a remembered scale passed over was named.

**Recommendation.** Keep the cap; title the skipped button and add a clause to the size line.

**Fixed.** The skipped default's button is titled '(the usual opening size)', and while the size is the one it opened at the size line names the next scale up and the run it is past ('1.25× and larger go past the 1536×1536 (2.36 MP) measured on Z-Anime at fp8'). `72044fa`.

*zanime-5*

#### A41. Anima's one Example, cut to its first sentence, draws the check's own 'at least 2 sentences' note

**low** · common · Anima · confirmed · `w2-controls`

**Observed.** The card's caption has a second sentence that satisfies the rule verbatim.

**Recommendation.** Give the caption whole.

**Fixed.** The card's dragon caption, both sentences, rechecked against the card; a test holds every guide with a sentence rule to its examples. `c039137`.

*anima-6*

#### A42. Examples holding an author's template slot are offered with no note and generate the slot literally

**low** (finder: medium) · common · Z-Anime and NetaYume · adjusted · `w2-rules`

**Observed.** Z-Anime's '[character]' example had no note, left the caret at the end, and Ctrl+Enter rendered '[character]' without a word.

**Recommendation.** The guide's slot patterns to the page as data, a note in the menu, the first slot selected on a pick.

**Fixed.** `PromptGuide.slots` from the placeholder rule; the menu notes 'fill in [character]' (or the slots' count) and a pick selects the first slot. `9961881`.

*zanime-3*

#### A43. Z-Anime's check never notes a missing negative prompt, though its guide asks for one

**low** · common · Z-Anime and NetaYume · confirmed · `w2-rules`

**Observed.** The guide prose says 'Use a negative prompt for unwanted features'; its `analysis` had no entry.

**Recommendation.** An info entry citing that sentence.

**Fixed.** A `negative_prompt` info entry citing 'Use a negative prompt for unwanted features; Base supports it fully.' `9961881`.

*zanime-7*

#### A44. The artist note does not pluralise, and an index outage reads as 'not every tag'

**low** · common · Anima · adjusted · `w2-rules`

**Observed.** 'wlop, kantoku is an artist in Danbooru's index'; with the index unreachable the empty line and the note said one clause twice and understated that no tag was checked.

**Recommendation.** A joined list with a plural verb; word the outage from `categories` being null.

**Fixed.** 'wlop and kantoku are artists…', each named once; `categories_unread` gives 'No tag was checked for an artist name' and 'artists not checked'. `58db19f`, `c4d44c3`.

*anima-7*

#### A45. The tag summary says '1 prompt tags'

**low** · common · screen reader · confirmed · `w2-a11y`

**Observed.** Both forms of the live summary used the plural.

**Recommendation.** `plural()` in both.

**Fixed.** `plural()` in both forms. `2287ea9`.

*a11y-17*

#### A46. Below about 680 px a running job's Now line widens the one-column page

**low** (finder: medium) · common · under 901 px · adjusted · `w2-narrow`

**Observed.** At 600x900 the page's scrollWidth was 662 against 585, with Cancel off the screen, for as long as the job ran.

**Recommendation.** `minmax(0, 1fr)` for the one column, so the line takes its ellipsis.

**Fixed.** `main { grid-template-columns: minmax(0, 1fr) }` under 900 px. `0b237c8`.

*narrow-2*

#### A47. In the one-column lightbox the facts are an 84 px scroller, hiding Seed, Created and Run where the letterbox has room

**low** · common · under 901 px · adjusted · `w2-sheetlb`

**Observed.** At 600x900, 800x900 and 900x700 only Model and Size showed.

**Recommendation.** Size the facts from the letterbox the fitted picture leaves.

**Fixed.** Under 901 px the facts get `6em + --lb-room`, the stage height the fitted picture leaves, decided with the side-by-side split (`lbStacked`). `8a31d22`, `bf79894`.

*Later:* I14 (refitted when Use seed's note shrinks the stage), `936ba28`.

*narrow-8*

#### A48. The delete dialog is clamped against innerWidth and runs 7 px under the scrollbar

**low** · occasional · under 901 px · adjusted · `w2-gallery`

**Observed.** At every width, for a row's last card and the lightbox's Delete.

**Recommendation.** Clamp against documentElement's client box.

**Fixed.** `placeAsk` places it inside documentElement's client box. `5e720df`.

*narrow-9*

#### A49. The critic's polite line speaks at every job's start and end, and after a first critique repeats the first-use download sentence

**low** · occasional · screen reader · confirmed · `w2-jobs`

**Observed.** With the check open a job wrote 'An image job has the card…' at its start and 'First use downloads … once.' at its end.

**Recommendation.** Mark the critic ready on an answer; keep live only the lines that answer a press.

**Fixed.** `#critic-line` is a live half (Downloading…, Thinking…) and a quiet one (the cost, the job holding the card), each written on a change; any answer marks the critic ready at once. `97e960a`, `d1c11ee`.

*Later:* I1 (held while the server is offline or the model loads), `936ba28`.

*a11y-14*

#### A50. The lightbox's zoom readout, a live region, is rewritten at every layout

**low** · occasional · screen reader · confirmed · `w2-sheetlb`

**Observed.** Ten wheel notches made ten announcements, and F rewrote it with the same words.

**Recommendation.** Write on change; say a burst once it ends.

**Fixed.** The readout is aria-hidden and written on change, and a hidden copy is said 400 ms after a burst ends, at once for a keyed zoom; the counter likewise. `8a31d22`.

*a11y-15*

#### A51. Arrowing through the closed Runs picker opens each run's sheet in turn

**low** (finder: medium) · occasional · screen reader · adjusted · `w2-sheetlb`

**Observed.** Each ArrowDown fetched a whole run (limit=1024) and scrolled the page.

**Recommendation.** Note a browsing key's change and open it on Enter or blur.

**Fixed.** A browsing key's change is only noted and opens on Enter or as focus leaves; Escape puts back the run shown; a pointer pick opens at once. `bedb13f`, `20bb5df`.

*a11y-9*

#### A52. The tag suggestion lists and the Examples menu have no name

**low** · occasional · screen reader · confirmed · `w2-a11y`

**Observed.** And the option a chip editor's ArrowDown selected was not exposed.

**Recommendation.** Names for both, and `aria-activedescendant` with option ids.

**Fixed.** Both lists are named 'Tag suggestions' and the menu by its button; the editor's option ids and `aria-activedescendant` came with `f2cce6b` (wave 1) and a smoke step now holds them. `2287ea9`.

*a11y-10*

#### A53. A pass's 'Reuse pass…' is named 'Reuse this pass in the Refine dialog', which does not contain its visible words

**low** · occasional · screen reader · confirmed · `w2-gallery`

**Observed.** A voice command saying what it reads did not find it.

**Recommendation.** Start the name with the visible words.

**Fixed.** 'Reuse pass in the Refine dialog'. `22558e9`.

*a11y-19*

#### A54. Sheet column heads break 16-digit seeds mid-number below about 112 px cells

**low** · occasional · under 901 px · adjusted · `w2-sheetlb`

**Observed.** An 8-seed comparison at 1400x900 read '760152042044950' over '9' in every head.

**Recommendation.** Break only between groups of four digits.

**Fixed.** A seed goes to a line of its own whole; only a number wider than its column breaks, between groups of four from its end. `1877979`, `20bb5df`.

*narrow-7*

#### A55. On Z-Anime Compare 3 writes 0.05,0.15,0.2 instead of 0.1,0.15,0.25

**low** · occasional · Z-Anime and NetaYume · adjusted · `w2-refine`

**Observed.** Float error (0.5 × 0.15 / 0.05 = 1.4999999999999998) floored both ties, in the placeholder, the sweep line, the tip and the button.

**Recommendation.** Snap the quotient before rounding half up, in both copies.

**Fixed.** `compareSweep` and `model_defaults.strength_sweep` snap the quotient to six places before rounding half up. `72044fa`.

*zanime-4*

#### A56. Where Mu or LoRAs are unavailable, the grid's Mu and LoRA weight axes are greyed with no reason

**low** · occasional · Anima, NewBie · confirmed · `w2-controls`

**Observed.** The Mu tip still taught 'none,1.15'; the LoRA weight kept '0-1x0.25' and 'Needs an adapter selected above', which cannot be done there.

**Recommendation.** Give the reason; restore the syntax help where the axis applies.

**Fixed.** A dead axis reads 'unavailable' with its reason (the Mu tip's, /api/loras') in its badge and title; the syntax help returns where the axis applies. `4573040`, `f4dbd43`.

*anima-9, newbie-9*

#### A57. On NetaYume Check finds the system instruction only as the loader's exact string

**low** (finder: medium) · occasional · Z-Anime and NetaYume · adjusted · `w2-rules`

**Observed.** Without its full stop, Apply removed only `<Prompt Start>` and kept the instruction, and Check of the result said nothing.

**Recommendation.** Flag the text before the marker as one span where it reads as an instruction.

**Fixed.** The loader's instruction is matched in any case and spacing, its stop optional; another workflow's one-sentence 'You are a/an … assistant/model…' opener before the marker is cut with it as one span. `a954774`, `5627802`, `00c1df6`.

*zanime-1*

#### A58. On a prose guide a list parted by fullwidth commas or line breaks gets no style note

**low** (finder: medium) · occasional · ERNIE · adjusted · `w2-rules`

**Observed.** '小狗，雪地，松树林，阴天，冷色调' read as a tag list with no note, while the English comma list got one.

**Recommendation.** Count the pieces the style was read from.

**Fixed.** `_style` counts pieces split at ',', '，' and line breaks, lettering one piece. `7d17925`.

*ernie-7*

#### A59. The English translation is dropped while the Chinese prompt it describes stays

**low** · occasional · ERNIE · confirmed · `w2-expand`

**Observed.** At the click of a re-Expand that then failed, by Reset's Undo, and on a reload.

**Recommendation.** Clear it when a new expansion lands; keep it in the snapshots.

**Fixed.** Cleared only when another expansion lands or the prompt changes; carried in `panelState()` for Reset's and Reuse's Undo and a reload. `e92c8cc`, `6e18c84`.

*ernie-9*

#### A60. Typing during an Expand throws the finished expansion away

**low** (finder: medium) · occasional · ERNIE · adjusted · `w2-expand`

**Observed.** Only a note said so, and the sampled text could not be recovered.

**Recommendation.** Hold it under the prompt with Use it and Discard.

**Fixed.** 'Expansion ready - made from the prompt before your edits.' with Use it and Discard, under the prompt; such long notes take a line of the panel's flow. `e92c8cc`, `8edd609`.

*ernie-5*

#### A61. Expand's failures reach the page as bare library or pydantic text with the exception's class name

**low** · occasional · ERNIE · confirmed · `w2-expand`

**Observed.** 'width: Input should be greater than or equal to 256' alone, or 'RuntimeError: …', with no word that Expand failed or that the prompt is unchanged.

**Recommendation.** Wrap them; strip the class name, as the critic's route does.

**Fixed.** `main._expand_detail` strips the class name (the log keeps it); the page says `Expand did not finish: <reason>. The prompt is unchanged.`, and a size or length the server would refuse is said at the box. `4703441`, `e92c8cc`, `6e18c84`.

*ernie-13*

#### A62. A critic reorder that only moves fields inside an XML block is set aside as 'it changed words'

**low** · occasional · NewBie · confirmed · `w2-rules`

**Observed.** No word changed.

**Recommendation.** Keep the refusal, with a sentence of its own.

**Fixed.** `critic._not_a_reorder` says what changed: punctuation or spacing, words moved between blocks, words moved inside a block, or the text outside the blocks. `a5369e9`, `e66505e`.

*newbie-6*

#### A63. After a failed load is retried and succeeds, the finish line still points at a card that has gone

**low** · occasional · two tabs · adjusted · `w2-jobs`

**Observed.** 'why, and Retry load, are in the viewer' stayed, and another tab's red gated sentence stayed under a ready topbar.

**Recommendation.** Reword the line and take back the detail when the state leaves error.

**Fixed.** The 'Not made' sentence is reworded from the state in every tab, an unload since included, and the detail it posted to `#error` is taken back. `97e960a`, `d1c11ee`.

*twotabs-10*

#### A64. Reusing another checkpoint's pass says 'at strength 0.1' beside this checkpoint's 0.4

**low** · occasional · Anima · confirmed · `w2-refine`

**Observed.** The fold note named only steps, CFG and mu as swapped.

**Recommendation.** Say whose the strength was and what Strength starts at; count strength in the fold note.

**Fixed.** The reused line says whose the strength was and what Strength starts at (the profile's, the general starting strength, or the one last run), and the fold note counts strength among what is this checkpoint's (`refine_strength_profiled`). `72044fa`, `b5dda4f`.

*Later:* I16 (mu left out of that note where Mu is greyed), `936ba28`.

*anima-4*

#### A65. A comparison version that moves whole XML blocks is labelled with a long run of markup words

**low** · occasional · NewBie · confirmed · `w2-sheetlb`

**Observed.** 'same words, moved: character_2 … kasane_teto … pink_hair …'.

**Recommendation.** Name the block that moved.

**Fixed.** Moved blocks are named by tag (`same blocks, moved: <character_2> first`); loose text is compared gap by gap; the page's block pattern is held equal to the critic's by a test. `51b32f9`, `aba5f6e`.

*newbie-11*

#### A66. An Expand ending at a failed load repeats the viewer card's sentence in red, with literal backticks, and drops focus

**low** · occasional · ERNIE · confirmed · `w2-expand`

**Observed.** A Generate in the same state posted nothing there.

**Recommendation.** Point at the card; say the sentence once beside images.

**Fixed.** The route answers 503 with the status's sentence; beside the card a short pointer, beside images the sentence once with its commands in quotation marks. `4703441`, `6e18c84`.

*ernie-12*

#### A67. A grid variable set to none strips every line's indentation and doubled space

**low** (finder: medium) · occasional · NewBie · confirmed · `w2-narrow`

**Observed.** On an indented XML prompt the none cell differed from the others by about 20 Gemma tokens beyond the dropped value.

**Recommendation.** Close only the hole the empty value leaves.

**Fixed.** `grid.substitute` closes one adjacent space (`_close_hole`); a placeholder alone on its line empties that line. `2e72181`.

*newbie-4*

#### A68. The critic's reorder putting `<general_tags>` ahead of the character blocks is kept, against NewBie's documented order

**low** (finder: medium) · occasional · NewBie · adjusted · `w2-rules`

**Observed.** The page offered it under Use reordered and Compare, though order issues that reverse the order are already dropped.

**Recommendation.** Set it aside in `critic.validate`; optionally an info finding in Check.

**Partly fixed.** `critic.validate` sets such a reorder aside (`prompt_rules.scene_ahead`): 'it puts `<general_tags>` ahead of a character block, against the documented order'. The optional Check finding was not added, by decision: it would contradict Check's 'Not checked' for XML order. `a5369e9`, `e66505e`.

*newbie-5*

#### A69. A bracketed slot gets two findings on one span, and slots inside a clause get none

**low** · occasional · Z-Anime and NetaYume · adjusted · `w2-rules`

**Observed.** '[character]' drew the author's slot warning and the encoder's de-emphasis warning; '[location]' and '[time of day]' neither.

**Recommendation.** One finding per span; the card's landscape template as an example.

**Fixed.** An encoder weighting finding on exactly an author slot's span is dropped; the card's landscape template is a second Z-Anime Example, its four slots a second placeholder entry. `9961881`.

*zanime-9*

#### A70. On fixed-shift checkpoints a blank Mu is shift 6 (mu 1.79), yet the sheet sorts 'none' first and never says what it is

**low** · occasional · Z-Anime and NetaYume · adjusted · `w2-controls`

**Observed.** 'Mu 1,none,2.5' laid out 'Mu none | Mu 1 | Mu 2.5'.

**Recommendation.** The static shift's mu in the profile, shown in the placeholder and the sheet.

**Fixed.** Profiles carry `mu_static` (ln 6 on Z-Anime, NetaYume and NewBie; ln 3 on Z-Image); the Mu placeholder shows it, and the grid preview and the sheet read 'none (1.79)' and sort it there. `c039137`, `4573040`, `f4dbd43`, `d6e1535`.

*zanime-12*

#### A71. Reusing another checkpoint's record writes its Mu into Anima's disabled Mu box, and Generate sends it

**low** · occasional · Anima · adjusted · `w2-controls`

**Observed.** The fake recorded mu 0.8; the real `applied()` drops it.

**Recommendation.** Write Mu only where supported; send none from a greyed box.

**Fixed.** `fillPanel` writes Mu only where supported, `collect()` and the grid send none from a greyed box, and the Reuse note names only the boxes that can be checked. `4573040`.

*anima-2*

#### A72. A pass's Reuse on a checkpoint with no pass opens a dialog that can never run, still saying 'a load answers for certain'

**low** · occasional · Anima · confirmed · `w2-refine`

**Observed.** The model was loaded, so the capability was already final.

**Recommendation.** Word the line by state; the repo's own pipeline classes are final before a load.

**Fixed.** `refine_settled` (a loaded pipeline's answer, or the repo's own classes, Anima Turbo and NewBie): there the line is the card's refusal. `72044fa`.

*anima-3*

#### A73. When the critic's rewrite adds the @, the page still lists the artist finding as possibly applying

**low** · occasional · Anima · adjusted · `w2-rules`

**Observed.** The rewrite's lint ran with no categories.

**Recommendation.** Pass the analysis's categories to the lint and mark when every rewrite item was asked about.

**Fixed.** The rewrite's lint gets the analysis's artist categories; `rewrite_artists_checked` (`artists_read`) holds only where every item a Danbooru name could be was asked about and no artist stays without its @. `58db19f`, `c4d44c3`.

*anima-10*

#### A74. The Search tip runs past the window at 700 px and below, and the grid's LoRA weight tip past the left edge

**low** · occasional · under 901 px · confirmed · `w2-narrow`

**Observed.** At 600x900 the page's scrollWidth went from 585 to 709.

**Recommendation.** Clamp tips sideways.

**Fixed.** `placeTip` shifts a tip in by what it would overrun on either side, stopping 8 px short and rounding away from the edge. `0b237c8`, `4343213`.

*narrow-6*

#### A75. Below about 700 px a load's topbar wraps and cuts the job mirror's clock

**low** · occasional · under 901 px · confirmed · `w2-narrow`

**Observed.** 'loading weights · …' with the elapsed time hidden.

**Recommendation.** Give the mirror the free-memory reading's room there.

**Fixed.** Under 700 px the free-memory reading gives way while a job is mirrored. `0b237c8`.

*narrow-11*

#### A76. A job queued behind another tab's paused job reads 'waiting for the job ahead…' for the whole pause, and behind a critique with no job ahead

**low** (finder: medium) · occasional · two tabs · adjusted · `w2-jobs`

**Observed.** The paused tab showed nothing waiting behind it, and an Expand started then said nothing of its wait.

**Recommendation.** Keep who holds the generation gate and derive `waiting_on` from it.

**Fixed.** The server keeps the gate's holder; `waiting_on` reads paused, critic, expand or queue, or nothing with nothing ahead; every job waiting at the gate is relabelled as it changes hands; the paused tab lists what waits behind it. `a4dfa2f`, `7e3a2ff`, `5c11d48`, `97e960a`.

*Later:* K5 (Expand's own line names another tab's holder), `936ba28`; I9 (the waiting note kept across a queue redraw), `936ba28`.

*twotabs-2*

#### A77. A job queued from another tab and picked up after this tab's own is treated as this tab's

**low** (finder: medium) · occasional · two tabs · confirmed · `w2-jobs`

**Observed.** Its first frame cleared the image just made, and its finish line and Use seed described the other tab's job.

**Recommendation.** Tell this page's submissions apart from others'.

**Fixed.** `isForeign`: shown in the progress row only, named 'Another tab's job', ending 'Another tab's job finished: N images.' with no Use seed, and this tab's image stays. `97e960a`, `d1c11ee`.

*Later:* I4 (New image and this tab's finish line kept), `936ba28`.

*twotabs-5*

#### A78. A deleted reference is still sent with every Generate and refused each time

**low** · occasional · two tabs · confirmed · `w2-jobs`

**Observed.** Whether marked deleted here or deleted in another tab, until the chip was removed by hand.

**Recommendation.** Resolve it as Reuse does: drop it on the server's 'unknown image'.

**Fixed.** A chip whose picture fails asks for its record and drops it on 'unknown image'; a refusal naming a reference does the same and says so in the page's words. `97e960a`, `d1c11ee`.

*twotabs-6*

#### A79. A queued job that another tab cancelled stays in this tab's queue rows

**low** · occasional · two tabs · confirmed · `w2-jobs`

**Observed.** With its × and Cancel queued, until the watched job ended.

**Recommendation.** Resync the rows from the server while busy.

**Fixed.** `resyncQueue` makes the rows match the server's while busy; rows another tab submitted say so, and which are this tab's survives a reload in sessionStorage. `97e960a`, `d1c11ee`.

*twotabs-9*

#### A80. Re-render's refusal advises 'add model:Anima', which also matches Animagine

**low** · rare · Anima · confirmed · `w2-gallery`

**Observed.** Following it gave the same refusal; '1 of the selected images were made'.

**Recommendation.** Quote the whole id, or only 'Untick All models'.

**Fixed.** `main._foreign_refusal` offers the whole running id only where it is in none of the other ids selected, else 'Untick All models'; the verb agrees with the count. `2d8e7d5`.

*anima-5*

#### A81. On a low-step source Compare 3 writes two values under a label that says 3

**low** · rare · screen reader · adjusted · `w2-refine`

**Observed.** Its title still promised 'this checkpoint's 0.1 and either side of it'.

**Recommendation.** Build the title from what it writes.

**Fixed.** The button reads 'Compare 2' there, its title naming the values and the one left out and why ('0.05 leaves no step at 12 steps'). `72044fa`, `b5dda4f`.

*a11y-18*

#### A82. On a portrait one-column stage side by side always splits left and right

**low** · rare · under 901 px · confirmed · `w2-sheetlb`

**Observed.** 213 px halves where stacking gave about 309 px.

**Recommendation.** Split along the stage's longer side.

**Fixed.** `lbStacked` stacks the halves where that fits the picture at least an eighth larger, decided once per layout. `8a31d22`, `bf79894`.

*Later:* I18 (Side by side's title says which way before the press), `936ba28`.

*narrow-10*

#### A83. Starring an image deleted elsewhere says only 'unknown image' and leaves it live

**low** · rare · two tabs · confirmed · `w2-gallery`

**Observed.** In the panel's line and the lightbox's, while ⌦ in the same case says '#N was already deleted.' and tombstones it.

**Recommendation.** Do what ⌦ does.

**Fixed.** The star does what ⌦ does: `forgetImages`, '#N was already deleted.', and the Runs picker asked again. `83639ba`, `971d4f5`.

*twotabs-7*

#### A84. The Refine dialog refuses a source deleted elsewhere with the server's 'no gallery image with id 28'

**low** · rare · two tabs · confirmed · `w2-refine`

**Observed.** The dialog kept its title, source line and picture with no word that the source was gone.

**Recommendation.** Say it in the page's words and mark the source line.

**Fixed.** '#28 has been deleted, so it cannot be refined.', the source line marked '(deleted)', the dialog kept open with its edits. `72044fa`.

*twotabs-8*

#### A85. A Cancel pressed in another tab reads as this tab's own

**low** · rare · two tabs · confirmed · `w2-jobs`

**Observed.** 'Cancelled — kept 1 image.', titled 'Cancelled', in both tabs.

**Recommendation.** Record the cancels this page asked for.

**Fixed.** Others read 'Cancelled from another tab or window — …', titled 'Cancelled elsewhere'; a shutdown's stop is the server's (`stopped_by` 'shutdown'). `a4dfa2f`, `97e960a`, `d1c11ee`.

*twotabs-11*

## Integration review

After the merges, three reviewers read the integrated branch at `66bcf14`, each on fakes
of their own:

- the backend and its routes;
- the job and tab flow;
- the views.

They raised 18 findings, 6 medium and 12 low, ordered here the same way. All were fixed
in `936ba28`, together with five leftovers of the second wave (K1-K5, below), on 2,106
passing tests. All 34 smokes passed at their latest run.

Two writers then brought CLAUDE.md and AGENTS.md (`c4515b9`), and the design notes and
README (`f9de976`), up to the code. A checker raised 24 points against them. The first
was that both had been written on `66bcf14`, before `936ba28`. All 24 were fixed in
`a3ab988` and `564f94d`. The final pass took 8 agents.

#### I1. Ask the critic is the one editor action not held while the server is offline or the image model loads

**medium** · `936ba28`

**Observed.** Offline, a press read 'The critic did not answer: No answer from the Local Diffusion server — is it still running?.' and was never taken back; during a load the route's 409 came back as the answer.

**Fixed.** Held (`criticHold`) offline and during a load, with the reason as title and in the cost line; an offline failure reads 'the server did not answer' and the poll that answers takes it back; `critiqueFailureText` strips any closing stop.

*BK1*

#### I2. One comma item over 200 characters makes /api/tags/validate refuse the whole tray

**medium** · `936ba28`

**Observed.** Every chip turned 'Danbooru unreachable', guide tags went unchecked and the order captions were lost (Kohaku's hybrid prose).

**Fixed.** The request refuses only blank items or items over 4000 characters; an item longer than any Danbooru tag (`tags.TAG_NAME_MAX`, 200) is answered 'longer than any Danbooru tag' without asking the index.

*BK2*

#### I3. When another tab's job ends, an idle tab's open contact sheet is rebuilt and focus drops to `<body>`

**medium** · `936ba28`

**Observed.** The catch-up prepended the other job's images under the sheet.

**Fixed.** No catch-up while a sheet is open (Sheet off resets the gallery anyway), and the sheet keeps focus on a redraw (I6).

*F1*

#### I4. A job adopted from another tab withdraws New image for this tab's results and replaces this tab's finish line

**medium** · `936ba28`

**Observed.** Use seed went with it.

**Fixed.** New image is withdrawn when a job's first frame clears the strip, not at its start; another tab's job leaves `#finish-note` alone and is said on `#job-said` only.

*F2*

#### I5. Another tab's images never reach the gallery if this tab's own job, queued meanwhile, ends with no images

**medium** · `936ba28`

**Observed.** The quiet end of `elsewhereBusy` skipped the catch-up.

**Fixed.** `elsewhereCatchUp` records the owed catch-up, paid at this tab's next job end or the stream's giving up.

*F3*

#### I6. Redrawing the sheet drops keyboard focus to `<body>`

**medium** · `936ba28`

**Observed.** Two new triggers on this branch: F on a cell whose image another tab deleted, and another tab's job ending.

**Fixed.** `renderSheet` records the focused cell or control (`sheetFocusPlace`) and restores it, else the image now at that index, else the search box.

*V1*

#### I7. Past 4000 characters the chipless token line turns into 'Tokens: cut unknown' and takes back its spoken warning

**low** · `936ba28`

**Observed.** The measure route refused the text.

**Fixed.** The first 4000 code points are measured; past that the line leads with 'Over 4000 characters: Generate and Check refuse it', in amber, then the counted part's count and cut.

*BK3*

#### I8. The Refine dialog's Refine stays live while the critic or Expand holds Generate

**low** · `936ba28`

**Observed.** A press was then refused.

**Fixed.** `syncRefineRun` puts `submitHold()` first; a press says the reason in the dialog.

*BK4*

#### I9. Queueing a job, or any queue resync, wipes the waiting note under the bars

**low** · `936ba28`

**Observed.** The job ahead paused, or the first load.

**Fixed.** `showCell` keeps the note in its arguments, so every redraw writes it again.

*F4*

#### I10. A press on Expand held by this tab's own job overwrites a Reuse's Undo

**low** · `936ba28`

**Observed.** The press wrote the panel's line.

**Fixed.** It goes through `expandSay`, under the prompt while a Reuse's Undo holds the line.

*F5*

#### I11. Ask the critic held by this tab's job says why in red, and the line outlives the job

**low** · `936ba28`

**Observed.** Expand and Unload in the same state use a muted note that clears.

**Fixed.** A muted `showJobNote`, cleared at the job's end.

*F6*

#### I12. Reset leaves a Reuse's note in the panel line over the emptied panel

**low** · `936ba28`

**Observed.** 'Settings from #N loaded - your prompt was replaced' stayed.

**Fixed.** `resetPanel` clears it while the line still starts with that note's lead (`reuseSaid`).

*F7*

#### I13. The running checkpoint is named two ways

**low** · `936ba28`

**Observed.** The catalogue label in the topbar, facts and load card; the repository segment in the Reuse note, the ⚠ title and Expand's running line.

**Fixed.** `checkpointName` everywhere the running checkpoint is named, the Refine dialog's lines included.

*F8*

#### I14. Under 901 px the lightbox's Use seed note shrinks the stage without refitting the picture

**low** · `936ba28`

**Observed.** A portrait lost its bottom 30 px while the readout said 'fit'.

**Fixed.** A ResizeObserver on the stage runs `layoutLightbox` while the lightbox is open.

*V2*

#### I15. With Gallery beside at QHD the card-size S button changes nothing

**low** · `936ba28`

**Observed.** The column stayed three cards across at 209 px.

**Fixed.** S has its own least width in the column: four across at QHD.

*V3*

#### I16. The Refine dialog's note for another checkpoint's record lists mu where Mu is greyed

**low** · `936ba28`

**Observed.** Reuse's note for the same case had been changed to leave it out.

**Fixed.** The list is steps, CFG, mu only where supported, and strength for a pass.

*V4*

#### I17. On FLUX.1 the Refine dialog's CFG box is plain 'CFG' beside True CFG

**low** · `936ba28`

**Observed.** The panel and the grid call it 'CFG (distilled)'.

**Fixed.** `#refine-cfg-label` takes `hints.cfg_label`; the dialog's negative caption uses the panel's wording.

*V5*

#### I18. Before Side by side is pressed its title says 'The source left of the pass' where the press will stack them

**low** · `936ba28`

**Observed.** The title read the layout before the comparison was on.

**Fixed.** The split is decided with `lbStacked` for any pass and titles the button.

*V6*

**The second wave's leftovers, fixed in the same commit:**

- **K1.** Reset's note lies in the top right of the prompt box it has just emptied, at every width and never in the panel's flow, so Generate does not move as it comes or goes; smoke-narrow's workaround from `66bcf14` is gone.
- **K2.** The sentence-count test runs only over the examples of guides that have a sentence rule, with a guard that the list is not empty.
- **K3.** Every press on a held button says its reason as a sentence ending in a stop; titles stay without it.
- **K4.** A ? badge inside a box that clips its overflow (the comparison's Batch, the Refine dialog's Size and Strength) hangs its tip from the window (`tipClipper`, `floatTip`), follows a scroll, and is hidden while the badge is scrolled out.
- **K5.** The status names what holds the card (`card_holder`: job, critic or expand) and the ticket an editor action was sent with; Expand's running line says when it waits for another tab's Expand or critic.

## Found on the card

The 2026-10-04 follow-up's GPU list was run on the RTX 5080 on 2026-10-05. The page
checks ran the real server with headless Edge at 2560x1305, pressed with real mouse
events; the refine ladders ran one size per process. The rows are in
`docs/verified.md`: under Behaviour, dated 2026-10-05, and in the Refine pass's tables
(`fd4f009`, `d814fc8`, `7ff6eee`). They found three defects, each fixed the same day:

- **C1. A load started while the page's first adapter listing ran failed:**
  `Float8Tensor.__new__() got multiple values for argument 'qdata'`, 3 times out of 3 on
  WAI v17 (fp8). The listing builds the denoiser on the meta device under accelerate's
  `init_empty_weights()`, which patches `Module.register_parameter` for the whole
  process, so torchao's fp8 weights went to meta as a load registered them. A load now
  waits out a build, a build asked for during a load leaves the adapter unchecked
  (`LoadInProgress`), and builds go one at a time. `02cd1d9`. Rechecked on the card:
  three fresh processes with Load pressed 0.87 s after the page opened all loaded.
- **C2. A pass that landed after its source was deleted during its own cell was shown as
  live.** `refine_missing` was false, and its Reuse offered the Refine dialog on an image
  that was gone, until a reload. `save_image` now looks the source up in its own
  transaction, a job's replay marks such passes, and the page marks an arriving pass
  whose source it saw deleted. `2b89ed5`, checked on the fake and in pytest, not again
  on the card.
- **C3. With `HF_HUB_OFFLINE=1` on an uncached checkpoint, the load card said to connect
  to the network**, which was up. huggingface_hub raises `LocalEntryNotFoundError` from
  `OfflineModeIsEnabled`, and the outer class won. The offline class now wins wherever it
  is on the chain. `81781bf`, tested on the library's own chain, not again on the card.

The same checks also found smaller faults, fixed and checked on the fake:

- a fitted batch's lone preview sat centred and each image moved left as the next landed
  (`9377de1`);
- a short gallery's batch refitted by 3 px as the page's scrollbar went at the job's end.
  `fitViewer` now allows for that scrollbar (`7733f81`), after a stable gutter on the
  page, tried first (`2338697`), narrowed every vw size and was reverted (`3e143b7`);
- the budget line counted 4 of 6 seeds for a grid that made 5 (`6a7570d`);
- a sweep's Resume was refused, again and again, once its source had gone (`ad01fdc`);
- the time left was estimated from the first step, which carries the encode and the move
  to the card (`2a4d2cc`);
- the critic's clock wrapped to a line of its own at QHD (`b396e2e`);
- sizes computed in 1024s were labelled GB (`5ec8adb`).

## Validation

On the final tree, `uv run --group dev pytest -q -p no:cacheprovider` gives **2,114
passed and 4 skipped**. All 34 browser smokes pass, each on a fresh fake server, two of
them new with this work:

- `smoke-narrow`, on 7889: one column and narrow windows;
- `smoke-stage`, on 7888: Gallery beside.

The details are in `docs/review-2026-10-04-ui.md`, "Leftovers — 2026-10-05, second
pass".
