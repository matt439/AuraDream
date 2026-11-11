# AuraDream phone UI review (2026-11-05)

Scope: the phone tier (`(pointer: coarse)` with width under 700 px, or height of 500 px or less), measured at 393x852 (iPhone 15 Pro), 360x780, 320x700 and 852x393 landscape against the fake server. Eight reviewers covered Create information architecture, touch targets, Gallery, viewer and lightbox, dialogs, platform conventions, visual design and architecture. Every finding below was re-checked against the screenshots, `measure-create.json`, `measure-gallery.json` and the source. The screenshots (about 50 MB, named in the evidence below) are not committed: the tours in [`review-2026-11-05-phone/`](review-2026-11-05-phone/) take them again against a fake server (`tools/fake_server.py --port 7990/7991 --step 0.3 --data <scratch>`, then `AURADREAM_SCREENSHOT_DIR=<dir> node tools/browser.mjs <tour>`); `measure-*.json` were their logged small-target lists.

Measurement caveat: the capture ran with `mobile: false`, so Chrome drew a 15 px classic scrollbar. Widths and "limit 377/378 px" figures are about 15 px narrower than a real iPhone, which uses overlay scrollbars. No on-screen keyboard was drawn, so the keyboard findings are inferred from code and need a device check.

---

## 1. Verdict

The owner is right, and the cause is structural rather than a matter of sizes. The phone tier is about 327 lines of overrides in `web/styles/touch.css` and 90 lines in `web/touch.js`, laid over desktop components whose DOM order, markup and JS placement code were written for a two-column mouse layout. On the phone, the desktop's panel and stage are stacked into one column. The result is that Generate sits about 880 px down the page and is not on the first screen (`create-01-load-top.png`). The finished image and its live preview land about 1300 px down, under Notify, Grid search, Step cache and LoRA (`create-18-finished-full.png`). The running job's progress and queue sit mid-panel. The Gallery shows about 370 px of toolbar before the first picture (`gallery-01-top.png`), and the lightbox gives the picture about 40% of the screen (`gallery-20-lightbox-fit.png`). On top of that, the touch tokens deliberately keep a second tier of 32-36 px targets and never set a minimum width, so 40 of the 55 visible Create controls at load are under 44 px. The type scale is still the desktop's 12-13 px. There are no phone primitives: no bottom action bar, no sheet with a sticky footer, no action sheet, no history-backed Back. The phone smokes run with `mobile: false` and assert none of these properties. Several real breakages come straight from this override-only approach: the Inpaint, Outpaint and Edit footers end past the screen, the Inpaint and Outpaint settings sit off the right edge, the lightbox Delete is clipped, and the Examples menu scrolls the page sideways. Resizing alone will not fix it. The phone needs its own task order (compose, Generate, watch, judge, iterate), a thumb-zone action bar, its own viewer and its own sheet patterns. All of this can be built inside the phone media query while the desktop (and its 1400x900 smokes) and the tablet's two columns stay unchanged.

---

## 2. Target phone design

### 2.1 Principles

- Phone-only. Everything is keyed on `PHONE_MEDIA` (`state.js`, the same text as in `touch.css`). The desktop, the 1400x900 mouse smokes and the tablet's two columns are unchanged.
- Move nodes, never copy or reorder them with CSS. Where a control must leave the panel (a sheet, the action bar), `touch.js`/`phone.js` moves the live node into a phone host with the existing `moveBefore` pattern (as `placeTagValidation` does) and moves it back on a `matchMedia` change. Each id exists once, so `els`, `writeBox`, `setReferences` and `setQueue` keep working. Do not use `display: contents` on `.panel` or CSS `order` for anything whose focus order matters (WCAG 1.3.2/2.4.3; WebKit drops the region's landmark). `position: sticky` on `#btn-generate` is allowed, since it changes no order.
- Targets: at least 44x44 px with at least 8 px between neighbours. List and menu rows 48 px. The primary action 50-56 px. Inline citation links in prose are the only exception, and the audit helper lists them.
- Held buttons stay visible and `aria-disabled`, and a press says why (AGENTS.md). Nothing scrolls by itself. Nothing moves under the pointer: fixed bars reserve their height in `main`'s padding and `scroll-padding-bottom`.

### 2.2 Navigation model

Two tabs stay: Create and Gallery. The Create view gets a bottom action bar (Generate, a progress or result thumbnail, a queue badge) docked above the tab bar. Results do not get a third tab.

Reconciled: platform and architecture proposed a third Results tab, while create-ia and viewer proposed a Create action bar with a result dock. I chose the action bar with a dock. It keeps the compose, Generate, look loop on one screen, keeps "only a press switches the view" without a badge-driven tab, and adds one row of chrome instead of a destination the user has to keep switching to. The dock's thumbnail opens the full-screen result viewer, which is the lightbox in its phone layout, on a press only. The queue opens as a sheet from the badge. If the dock proves too small in use, a Results tab can be added later on the same `body[data-view]` mechanism, and the dock design does not block it.

Back: the lightbox, every sheet, the Models sheet, the context and action sheets and the run sheet push a history entry on the phone. Back (Android's button, iOS's edge swipe) closes the top layer first. Tabs use `replaceState`.

### 2.3 Create view (393x852 portrait)

```
+---------------------------------------+  0
| [● FLUX.1 Krea [dev] ▾]          [⋯] |  44  topbar: model chip opens Models,
+---------------------------------------+       ⋯ = Unload, Saved setups,
|  ┌─────────────────────────────────┐  |       Snippets, Load settings from PNG,
|  │ a fox walking in the snow,      │  |       Reset
|  │ overcast light                  │  |
|  │                                 │  |  prompt (16 px text)
|  └─────────────────────────────────┘  |
|  [Examples] [Check] [Tags]            |  44 px toolbar under the box
|                                       |
|  Shape [Square|Portrait 3:4|Lands 4:3]|  44 px segmented control
|  Size  [1 MP ▾]     Images [− 1 +]    |  44 px chip + stepper
|                                       |
|  More options                      ›  |  48 px row, live summary:
|  28 steps · CFG 3.5 · seed random  •  |  dot when anything differs
|                                       |
|  ┌─ Latest result ────────────────┐   |  result card, moved here
|  │            [image]             │   |  from under the panel
|  │ [☆ Star][↻ Reuse][⤢ Refine][⋯] │   |  labelled 44 px actions
|  └────────────────────────────────┘   |
|  Recent ...                           |
+---------------------------------------+ 732
| [▣ 48] [      Generate       ] [ 2 ]  |  64  action bar: thumbnail or progress
+---------------------------------------+ 796       ring "20/28", Generate 50 px,
|      ✦ Create        ▦ Gallery        |  56       queue badge opens the Queue sheet
+---------------------------------------+ 852  tab bar (+ safe area)
```

- Fixed chrome is 44 + 64 + 56 = 164 px (19%), against 114 px today. In exchange, Generate, the job's state and the newest result are always one tap away. While a box has focus (`body[data-typing]`), the action bar and tab bar step aside, as the tab bar does today. Once the visualViewport inset (P44) exists, the action bar rides above the keyboard instead.
- More options is a full-height sheet holding Sampling (Steps, CFG; Mu and True CFG under Advanced), Seed with a 44 px dice, Repeat, the negative prompt (inline under the prompt instead on checkpoints that use negatives), Custom dimensions, LoRA & references, Then refine, Notify, Step cache (still visible and inspectable) and Grid search (Run grid in the sheet's footer).
- The Queue sheet hosts the real `#progress` and `#queue-list` nodes: Pause, Skip and Cancel in a 4-column 44 px grid with Cancel last in danger colour, and queued jobs as 56 px rows with a ⋯ for Move to front and Cancel.
- Check opens as a sheet. A one-line "Check: N findings ›" chip under the prompt reopens it.
- The tablet (two columns) keeps today's layout.

### 2.4 Gallery

```
+---------------------------------------+
| [● FLUX.1 Krea ▾]                [⋯] |  44  app topbar
| Gallery 15         [Search] [Select][⋯]| 48  sticky head (not hide-on-scroll)
| (Starred)(All models +2)(Runs ▾)(Sheet)| 44  one sideways-scrolling chip row
+---------------------------------------+ ~140 first image
| [img ][img ][img ]                    |  3 columns (--thumb-min ~110 px), 4 px gap,
| [img ][img ][img ]                    |  Fit by default, Crop/size in ⋯;
| [img ][img ][img ]   no tools on tiles|  star badge only (44 px press)
| ...  sticky day header "5 Nov"        |
+---------------------------------------+
|      ✦ Create        ▦ Gallery        |  tab bar; in Select mode replaced by:
| [Cancel] 3 selected [☆][Refine][More..]|  selection bar (44+ px, labelled)
+---------------------------------------+
```

- Tap opens the viewer. Long press keeps the context menu (it is the only way to some actions on touch), now shown as a bottom action sheet, with a new "Select" item that enters select mode with that image picked.
- The head's ⋯ holds Saved searches…, Bundles…, Recycle bin, card size and Crop, and last, in danger colour, Delete unstarred (its confirm is kept).
- Runs opens a sheet of run rows (thumbnails, date, count, axes). Picking one shows a removable "Run 5 Nov 01:35 ×" chip. The `run:` text stays in the underlying search, so the server logic and smokes are unchanged.
- The recycle bin is its own screen: "‹ Gallery", Recycle bin (2), Select, Empty bin, with Restore and Delete in the selection bar.
- Pressing the already-selected tab scrolls that view to the top (the user's own press).

### 2.5 Viewer (lightbox, phone)

```
+---------------------------------------+
| [‹]          2 / 15           [Share] |  overlay top bar 44, 40% gradient
|                                       |
|                                       |
|          full-bleed image             |  100vw x 100dvh on #000,
|          (pinch, double-tap 1:1,      |  tap toggles chrome,
|           swipe ← → next, ↓ close,    |  swipe up = facts sheet
|           ↑ facts)                    |
|  ─── a lighthouse on a cliff… seed 11 |  facts peek (72 px)
+---------------------------------------+
| ☆Star | ↻Reuse | ⤢Refine | ✎Edit | ⋯More|  56 px bar + safe area
+---------------------------------------+
```

- Edit opens a small sheet: Inpaint…, Outpaint…, Edit by instruction…. More opens the shared action sheet: Use seed, More seeds…, Use as reference, Show run, Copy…, Download, then Recycle bin last in red. This brings back Inpaint and More seeds, which are hidden today at 900 px or less.
- Facts become 48 px rows (term, value, chevron). Tapping a row opens Search / Copy / Use seed / Open source. The inline 24 px ⌕ buttons and dotted links go on the phone.
- Fit, 1:1 and the readout leave the screen (gestures do these). The zoom readout shows briefly as a chip while zooming. Share uses `navigator.share` in a secure context and falls back to the download link on plain-http LAN.
- Reconciled close placement: the targets verifier said keep the close at the foot, where the thumb reaches. The viewer reviewer wanted a top-left Back. In the immersive layout the foot belongs to the action bar, so Back goes top-left at 44 px, and it is backed by the history entry (system Back) and swipe-down. The interim fix, before the immersive layout, keeps the close at the foot and makes it 44x44.
- Landscape: the same immersive layout, with the five actions as a 56 px rail on the right and facts as a side sheet.
- The result dock opens this same viewer on the newest result. While a job runs it shows the live preview full width.

### 2.6 Sheets and menus

Three patterns replace the 16 desktop dialogs and 3 popover menus:

1. Form sheet (Refine, Inpaint, Outpaint, Edit, More seeds, Swap tags, Mass refine, Then refine, Bundles, Saved setups/searches/snippets, imports, Check, More options, Queue, Models): full screen with no radius. A sticky header (title plus a 44 px ✕), one scrolling body, and a sticky footer holding the primary action and its summary ("1280×1280 · Outpaint"), padded by the safe area and by `--kb-inset` above the keyboard. No text-box autofocus on touch. Inpaint keeps its fixed-top rule: the canvas is anchored under the header and sized by the viewport, never by what is painted.
2. Action sheet (long-press menu, More ⋯, Examples, Picked ▾, the head ⋯): bottom-anchored, with a scrim, max 70dvh with its own scroll, 48 px rows, grouped with 8 px spacers, destructive rows last in `--danger`, a Cancel row, and a history entry. `contextMenuPressClick`'s swallowing of the lift's click stays.
3. Confirm: a short alert. Optionally bottom-anchored with stacked 48 px buttons. Empty bin and Delete permanently keep their confirmations; the per-image delete still never confirms.

List managers (setups, searches, snippets) use list-then-edit: 56 px rows that apply on tap, a trailing ⋯ (Edit, Rename, Delete), and an edit screen pushed inside the same sheet. The native `<select>` stays in the DOM, hidden on the phone, as the source of truth for the desktop smokes.

The Models sheet keeps its element and its light-dismiss logic. It becomes one scroller: a header with Done, the checkpoint list as card rows (the whole row selects; star and guide link at 44 px), Performance as a fold, "Made with…" collapsed below, and a sticky footer with Apply & save, Switch & load and Unload (moved from the topbar, still held with its reason while a job or load is live).

### 2.7 Sizing and type scale (phone `:root` in touch.css)

| Token | Today (touch) | Phone target | Use |
|---|---|---|---|
| `--h-xs` / `--h-sm` / `--h-tool` | 32 / 36 / 36 | 44 / 44 / 44 | every control |
| `--h-md` | 44 | 44 | fields, buttons |
| primary | 50 | 50-56 | Generate, sheet primary |
| list row | 36 (menus), 29 (Picked) | 48 | menus, sheets, disclosure rows |
| `--fs-xs` | 11 | 12 | badges only |
| `--fs-sm` | 12 | 13 | meta, timestamps |
| `--fs-label` | 12 (uppercase) | 14, sentence case, weight 600 | captions |
| `--fs-body-sm` | 13 | 15 | status, facts, queue, Check prose |
| `--fs-md` | 14 | 16 | body, buttons, inputs share a baseline |
| `--fs-lg` | - | 18 | view and sheet titles |
| spacing | 0-8 between controls | 4 inside / 8 between / 16 fields / 32 sections (`--sp-5` 24, `--sp-6` 32) | |
| page frame | gutter 12 + bordered panel padded 14 | gutter 16, panel border, background and padding dropped | 361 px of content |

The tablet can keep 36/32: split the token block so the phone values live in the phone query, not the shared coarse `:root`. Button tiers: Generate is the only solid accent shape. Selected presets and toggles use `--accent-soft` with an accent border. Frequent actions are tonal (`--panel-2`), rare ones are text-only, and destructive ones use `--danger` text. Pressed states use a 12% `color-mix` background on `:active` (background only, no scale).

---

## 3. Findings

Severity is after verification. Effort: S (under a day), M (1-3 days), L (more than 3 days). "Merged" names the reviewer dimensions a finding combines.

### A. Create

**P1 - Critical - Generate is below the fold and never in the thumb zone; no action bar.** Merged: create-ia, platform, architecture.
- Evidence: `create-01-load-top.png`: the 738 px band between the topbar (58) and the tab bar (56) ends at Sampling/Output, so Generate is not on screen. It is at page y ≈874-924 at 393x852 (`create-01-load-scroll-0700.png`), and further down at 360x780 and 320x700 (dice at pageTop 910). In landscape only the prompt and negative show (`create-23-landscape-top.png`). With Check or the folds open it moves to about 1900 px. `#btn-generate` (index.html:502) and `#progress` (:504) are mid-panel children; touch.css only makes the panel static.
- Fix: step 1 (CSS only): under the phone query, `#btn-generate { position: sticky; bottom: calc(var(--tabbar-h) + var(--safe-b)) }` inside the static panel (no panel scroller needed), on the tab bar's background with a hairline. Step 2: a `#phone-actions` host above the tab bar (thumbnail or progress ring, Generate at 50 px, queue badge), with `#btn-generate` moved in by `moveBefore` and restored on a media change. Reserve its height in `main` padding and `scroll-padding-bottom`, hide it under `body[data-typing]` until the keyboard inset (P44) exists, and keep it `aria-disabled` with its reason when held. Update the smoke-phone line 28 assertion (panel bottom against `__tabTop`) to include the bar.
- Files: web/styles/touch.css, web/touch.js, web/index.html, web/job-progress.js, web/styles/generation.css. Effort: M.

**P2 - Critical - The latest result and the live preview are about 1300 px below Generate.** Merged: viewer, create-ia, platform, architecture.
- Evidence: `create-18-finished-full.png`: the result card starts at ≈1310 CSS px, its tools at pageTop 1636 and `#viewer-more-seeds` at 1701, on an 1827 px page. `create-18-finished-top.png`: nothing of the result is visible. `create-15-job-running-at-generate.png`: mid-run, only a 2 px bar and "8/28" show. The live preview renders in the same `#viewer` (`.viewer .live img`). touch.css:200-202 makes Create the whole `.panel` followed by `.stage`.
- Fix: under `PHONE_MEDIA`, move `#viewer` and `#recent` (one node each) to sit right after the quick settings, and move them back on a media change. The action bar's left slot shows the live preview thumbnail with a progress ring, then the finished thumbnail. A press opens the phone viewer (P30) on the newest image, showing the live preview full width while the job runs. It never switches or scrolls by itself. Document the phone exception in rules.md.
- Files: web/touch.js, web/styles/touch.css, web/viewer.js, web/job-results.js, web/lightbox.js, web/index.html. Effort: L.

**P3 - High - Job progress and the queue sit mid-page, with Cancel 4 px from Skip and 21-24 px queue buttons.** Merged: create-ia, targets.
- Evidence: `create-16-queue-several.png`: the topbar reads "FLUX.… 20/28 …". The Now row and three queue rows sit between Generate and Load PNG. `#btn-queue-front` 36x36, `#btn-pause` 48.6x36, `#btn-skip` 40.3x36 and `#btn-cancel` 53.1x36 are 4 px apart. The queued × is 23.5x32 and ↑ is 21.2x32, 8 px apart (measure-create "job running + 3 queued"). The tab bar carries only an 8 px dot and a 2 px line (`create-15-tabbar-busy.png`).
- Fix: drive the action bar from job-progress state (step count over the preview, a queue badge). A press opens the Queue sheet hosting the real `#progress` and `#queue-list` (so `setQueue` stays their only writer): Pause, Skip and Cancel as a `repeat(4,1fr)` 44 px grid with an 8 px gap and Cancel last in danger colour, and queue rows of fixed 56 px height with a 44 px ⋯ (Move to front, Cancel). Keep `#topbar-line` and `#topbar-fill` in the DOM (`syncPhoneTabsBusy` observes them). Interim: give the existing buttons 44 px and 8-12 px gaps.
- Files: web/job-progress.js, web/job-queue.js, web/touch.js, web/styles/touch.css, web/styles/generation.css, web/index.html. Effort: L (interim S).

**P4 - High - No progressive disclosure: desktop tuning controls fill the first screen.** Merged: create-ia (disclosure and advanced folds), architecture.
- Evidence: `create-01-load-top.png`: the negative's caption, 14 aspect buttons, 4 tiers, Custom dimensions, a stray "1.0 MP" (`#size-hint`, index.html:411) and Steps/CFG/Mu/True CFG with four ? badges. Then Seed, Batch, Repeat, Load PNG, Notify, Then refine, Grid search, Step cache "Off" and LoRA (`create-01-load-scroll-0700.png`). 40 of 55 visible controls are under 44 px. With the folds open the page is 2405 px.
- Fix: the More options row and sheet (section 2.3). Move the nodes in (they are moved, not copied, and restored on a media change), show a dot when any hidden value differs from the checkpoint default, and keep Step cache inspectable inside the sheet. Load settings from PNG also goes in the topbar ⋯, since `#png-hint` is hidden on touch and this is the phone's only way in.
- Files: web/index.html, web/touch.js, web/styles/touch.css, web/panel.js, web/execution.js, web/grid-preview.js. Effort: L.

**P5 - High - The size picker spends three rows on 18 equal-weight buttons with joined pairs.** Merged: create-ia, targets, visual.
- Evidence: `create-04-aspect-presets.png`: `#presets` buttons are 37.5x36 with 0 px inside a pair (7:9|9:7) and 6 px between pairs; 9:16/16:9 are 44x36. `#tiers` adds a third row at 43-58x36. At 320 px there are four rows, about 330 of 700 px (`create-22-320x700-top.png`). The selected 1:1 and 1 MP use the same solid accent as Generate. touch.css:79 sets only height.
- Fix: stage 1 (CSS only): an even 44 px grid with 8 px between items, the tiers as a 4-segment row, and the selection drawn in `--accent-soft` with an accent border. Stage 2: a Square | Portrait | Landscape segmented control that remembers the last ratio per orientation (a second tap opens that orientation's ratios as 48 px rows), plus a "1 MP ▾" chip. It uses the existing `#btn-orientation` and the server's own pair members from `sizes.js`, so it is presentation only. Move "1.0 MP" into the Custom dimensions summary.
- Files: web/styles/touch.css, web/styles/generation.css, web/sizes.js, web/touch.js. Effort: M.

**P6 - High - The result's actions are five unlabelled 27-33 px glyphs 4 px apart, with Delete next to Star.** Merged: targets, viewer.
- Evidence: measure-create "after job finished": reuse 27x36, refine 27x36, star 28x36, inpaint 30x36, del 33x36, 4 px gaps (0 in landscape). gallery.css:115-134 names them only on hover; touch.css:163 removes that on touch (`create-18-viewer.png`).
- Fix: on the phone, a labelled action row under the image (icon plus word, `repeat(4,1fr)`, 44 px, 8 px gap): Star, Reuse, Refine, ⋯. Delete and Inpaint go into ⋯ (the action sheet). `resultFigure` is shared with the gallery cards, so use a phone CSS branch or phone-only label spans.
- Files: web/styles/touch.css, web/styles/gallery.css, web/viewer.js. Effort: M.

**P7 - Medium - After a run, the same actions appear three times.** Merged: create-ia, viewer.
- Evidence: `create-18-finished-full.png`: `#finish-note` has Use seed, More seeds… and a held Finish run (63.6/86.1/67.4 x 33.5, 8 px gaps). Under the result there is More seeds and a 47.6x18 dotted "Use seed" link. `#last-run` repeats Show run, More seeds… and Finish run at load.
- Fix: on the phone, keep `#finish-note` as a one-line status ("Done · 9 s") with no buttons, and keep `#job-said` for screen readers. Make the result card the home of the per-result actions (P6), with Use seed and More seeds in its ⋯, and replace the 18 px link. Hide `#last-run`'s buttons only while the viewer shows that same run. Document that a Finish run with nothing to finish is not drawn on the phone (the PC keeps it aria-disabled). Use phone-only hides so smoke-results and smoke-more-seeds keep their desktop paths.
- Files: web/job-results.js, web/viewer.js, web/generation.js, web/styles/touch.css. Effort: M.

**P8 - Medium - The "New image ↓" pill covers Examples and Check.** Merged: viewer, platform.
- Evidence: `create-19-new-image-pill.png`: a 95x36 pill fixed at topbar + 8 px (generation.css:382-386), sitting over `#btn-examples` with a 0 px gap. It appears when a job ends, so it can land under a finger heading for Check.
- Fix: once the dock exists (P1/P2), hide `#btn-new-image` on the phone; the dock shows the unseen result. Until then, move it to `bottom: calc(var(--tabbar-h) + var(--safe-b) + 8px)`. Revise the generation.css comment about the window's foot.
- Files: web/styles/generation.css, web/styles/touch.css, web/generation.js. Effort: S.

**P9 - Medium - Prompt Check expands inline and pushes Generate down about 1000 px.** Merged: create-ia, dialogs.
- Evidence: measure-create: scrollHeight 2493 with Check open against 1483 at load. `create-13-check-a..d.png`: a nested card about 300 px wide, a 32 px "Show" button mid-paragraph, small pills, and a 28 px ✕.
- Fix: a `PHONE_MEDIA` branch in `placeTagValidation` docks `#prompt-analysis` in a form sheet: a sticky head (summary chips, Check again, 44 px ✕), sections as 48 px folds, each finding as a card with its source chip and a trailing 44 px Show. Leave a "Check: N findings ›" chip under the prompt. `TWO_COLUMNS_MEDIA` keeps the tablet's docking. Do this after P1.
- Files: web/analysis.js, web/tag-chips.js, web/styles/analysis.css, web/styles/touch.css. Effort: M.

**P10 - Medium - The prompt caption crams 34 px Examples, Check and Reset into one row, 3.8 px apart.** Merged: create-ia, targets.
- Evidence: measure-create: `#btn-examples` 65.4x34 is 3.8 px from `#btn-analyze` 48.8x34; `#btn-reset` is 44.5x34; the ? badge is 14x14. On tag checkpoints, Tags and Expand join the row and it wraps.
- Fix: a non-wrapping 44 px toolbar under the textarea (Examples · Check · Tags/Expand) with 8 px gaps. Reset moves to the topbar ⋯ (optional; its Undo note must still appear under the editor). The Examples menu overflow is P38.
- Files: web/index.html, web/styles/touch.css, web/styles/fields.css, web/panel.js. Effort: S.

**P11 - Medium - The negative prompt is a full field even where the checkpoint ignores it.**
- Evidence: `create-01-load-top.png`: on FLUX.1 Krea, "(needs True CFG > 1 — ~2× slower)", Set True CFG (77.7x32) and a 2-row box take about 150 px of the first screen and are inert at defaults.
- Fix: when inactive, a one-line "Negative prompt — off (needs True CFG)" row inside More options. On checkpoints that use negatives, a collapsible field under the prompt with a first-words summary. A non-empty negative written by Reuse or Load PNG expands it. Writes still go through `writeBox`.
- Files: web/index.html, web/panel.js, web/styles/touch.css. Effort: S.

**P12 - Medium - The topbar spends 58 px on the wordmark, a truncated name and an equal-weight Unload.** Merged: create-ia, platform, visual.
- Evidence: `create-21-topbar.png` (378x58): "FLUX.1 Krea [d…" beside 44 px Models and Unload. At 320 px the name is "FL…" (`create-22-320x700-topbar.png`). During a job it reads "FLUX.… 20/28 …".
- Fix: a 44 px phone topbar: a model chip (dot, short name, ▾, at least 44 px tall) that opens Models, and one 44x44 ⋯. Drop `.brand` on the phone. Unload moves into the Models sheet's footer, staying held with its 409 reason. Step counts leave the topbar only once the action bar shows them; `#job-mirror` stays until then. Extend the landscape topbar rules rather than forking them. No hide-on-scroll in portrait.
- Files: web/index.html, web/styles/touch.css, web/style.css, web/status.js, web/model-switch.js. Effort: M.

**P13 - Medium - The Seed dice is 28x18 with a press zone over the Seed box; Batch and Repeat have no steppers.**
- Evidence: `#btn-dice` 28.5x18; touch.css:85-86 `::after inset -10px -8px` covers the top 2 px of `#seed`. The dialogs' dice are 41x44. Batch and Repeat are 62x44 number boxes for values of 1-4.
- Fix: a 44x44 trailing dice inside the Seed row; remove the caption button's `::after`. Optional: −/value/+ steppers with 44 px buttons for Batch, Repeat and Steps, wired through the existing input events.
- Files: web/styles/touch.css, web/styles/fields.css, web/touch.js. Effort: M.

**P14 - Medium - While typing, the sticky topbar eats the visible area, and the numeric keypads are generic.**
- Evidence: `create-08-prompt-long-focused.png`: the caption sits under the 58 px sticky bar. touch.css:263-267 hides only the tab bar. There is no visualViewport handling, no `enterkeyhint`, and no `inputmode` on `#cfg`, `#seed` or `#steps`.
- Fix: `body[data-typing] .topbar { position: static }` on the phone (landscape already does this). `enterkeyhint="search"` on `#search` (already `type=search`), `enterkeyhint="done"` on the sampling boxes, `inputmode="decimal"` on `#cfg`/`#true-cfg`/`#mu` and `inputmode="numeric"` on `#steps`/`#seed`/`#batch`/`#repeat`. The keyboard inset is P44.
- Files: web/styles/touch.css, web/index.html, web/touch.js. Effort: S.

**P15 - Low - Runs of more than 4 images (`.viewer.many`) are 20vh tiles with five 32 px tools always drawn over them.**
- Evidence: gallery.css:29 `--tile-h: 20vh`, :137-145 overlaid tools, touch.css:160. Batch 4 is `.multi` (34vh, tools below), so this applies to Batch above 4 or a grid.
- Fix: on the phone, `.many` becomes a 2-column grid of plain thumbnails; tap opens the viewer at that index, long press opens the action sheet. `.multi` gets P6's labelled tools.
- Files: web/styles/gallery.css, web/styles/touch.css, web/viewer.js. Effort: S.

### B. Gallery

**P16 - Critical - The gallery head takes 43% of the screen before the first image.** Merged: gallery, platform, visual (density, hierarchy), targets (head and footer).
- Evidence: `gallery-01-top.png`: the first card is at y≈370 of 852. Above it: topbar 58, Gallery plus Runs, full-width search, ?/Starred/All models/Sheet, "Delete unstarred" on its own row, and 8 footer buttons in two 36 px rows (Bin 34.6 wide; filters 20x20 boxes in 74x36 and 123x36 labels). Landscape leaves about 140 of 393 px for images (`gallery-52-landscape-top.png`). Nothing is sticky, so the filters scroll away.
- Fix: section 2.4. A sticky 48 px head (title and count, Search, Select, ⋯) and one chip row. Delete unstarred, Bin, Bundles, Saved searches and the card view go into ⋯ (an action sheet). The selection actions go to the selection bar (P18). No hide-on-scroll. The existing buttons stay the DOM sources and keep their aria-disabled holds. `#gallery-error` gets a line under the chips. Target: first image at y≈140 in portrait and ≈90 in landscape.
- Files: web/index.html, web/styles/touch.css, web/styles/gallery.css, web/gallery-head.js, web/gallery.js, web/gallery-select.js. Effort: L.

**P17 - High - Every card carries six 22-28 x 32 px glyph tools, Delete 4 px from Inpaint.** Merged: targets, gallery, visual, architecture.
- Evidence: measure-gallery "gallery top": reuse 22.2, refine 22.2, ref 23.7, run 24.9, inpaint 24.9, del 27.6 wide, all 32 tall; Inpaint ends at x 153 and Delete starts at 157. sheet.css:202 forces them visible on coarse pointers. About 105 targets on a 15-image page (`gallery-11-card-tools-closeup.png`). Delete only bins, so it is recoverable.
- Fix: phone only: `#gallery .card .tools { display: none }`. Keep the DOM, because context-menu.js `buttonItem` rows click these buttons (it checks `hidden`, not CSS display). Give the star badge a 44x44 press area. Optionally add one 44 px ⋯ per card that opens the same sheet. The tablet keeps its tools. Update rules.md ("a card's or a run tile's tools are 32 px tall") and smoke-phone's "card tools 32 px tall".
- Files: web/styles/touch.css, web/styles/sheet.css. Effort: S.

**P18 - High - Selection is a desktop modifier model: the count and actions scroll away and there is no bottom bar.**
- Evidence: gallery-select.js:9-13,186-200. `#btn-select` is one of eight 36 px footer buttons. With 3 picked (`gallery-15-select-mode-top.png`) the head reads "✓3 | Clear✕ | Picked ▾ …", and once scrolled nothing shows (`gallery-16`). There is no Select all; the rings are 24 px pseudo-elements.
- Fix: while select mode is on, a fixed selection bar replaces the tab bar: Cancel, "N selected", Star, Bin, Compare (from Picked ▾), Swap tags…, Refine…, More seeds…, each at least 44 px and labelled. Add "Select" to the long-press sheet and a cheap Select all. Drag-select is optional.
- Files: web/gallery-select.js, web/context-menu.js, web/index.html, web/styles/touch.css, web/styles/gallery.css, web/touch.js. Effort: L.

**P19 - Medium - Two letterboxed columns and no density control.**
- Evidence: `--thumb-min` 160 px gives 2 columns at 393 and 360 and 4 in landscape. Cards are `object-fit: contain`. `#card-view` (S/M, Fit/Crop) is `display: none` below 1800 px, so `gallery-09` is identical to `gallery-01`.
- Fix: phone `--thumb-min` ≈110 px (3 columns, 5-6 in landscape) with a 4 px gap. Expose `#card-view` in ⋯ with phone values. Fit stays the default (sheet.css:183-186 rationale). Note the phone density in gallery.css and rules.md.
- Files: web/styles/touch.css, web/styles/gallery.css, web/gallery.js. Effort: M.

**P20 - Medium - Runs is a truncated native select that writes `run:<id>` into the search.**
- Evidence: `gallery-12-runs-sheet-top.png`: "5 Nov, 01:35 pm · 4 …", and the search fills with `run:7615e4504d30`; the first image is at y≈640.
- Fix: a Runs chip opening a sheet of run rows (a 3-thumbnail strip, date, count, axes, checkpoint). A pick shows a removable run chip. Keep `#run-list` (hidden) and the `run:` search string.
- Files: web/index.html, web/gallery-head.js, web/gallery.js, web/styles/touch.css. Effort: M.

**P21 - Medium - The run sheet shows keyboard hints and three rows of desktop controls.**
- Evidence: "F stars the image in focus · Ctrl+wheel zooms, drag pans" (sheet-render.js:336-338, `span.sheet-hint`) shows on touch; touch.css:106 hides only `.lb-keys` and `.key-hint`. Columns/Rows/Fill width, −(26x36)/+(33x36)/1:1, Export, Swipe ×2, Finish run and More seeds take three rows, and the 2x2 grid starts at y≈640 (`gallery-14`).
- Fix: now, add `.sheet-hint` to the hidden list (this is a bug against a documented rule). Then one row: an axis chip "Steps × CFG ▾", a Swipe toggle and ⋯ (Export, Fill width, Finish run, More seeds). Drop −/+/1:1 in favour of pinch and double-tap; sticky axis labels.
- Files: web/styles/touch.css, web/sheet-render.js, web/sheet-zoom.js, web/styles/sheet.css. Effort: M (hint S).

**P22 - Medium - The recycle bin keeps every gallery control and uses small per-card Restore/Delete.**
- Evidence: `gallery-41-bin-view-top.png`: Runs, Starred, Sheet, Picked, Select, Swap, Refine, More seeds, Saved searches and Bundles remain, the bin's own controls are added, and the first card is at y≈425. Per-card ↶ is 24x32 and ⌦ 29x32.
- Fix: its own screen (section 2.4), with no per-cell tools; Restore and Delete (red, confirmed) in the selection bar; Back leaves the bin. Hide gallery-only controls while bin mode is on.
- Files: web/bin.js, web/styles/bin.css, web/styles/touch.css, web/index.html. Effort: M.

**P23 - Medium - A tapped card keeps its "#14 · 1024×1024 · seed 5" label.**
- Evidence: `gallery-15` (right column, second card). sheet.css:256 shows `.tools::before` on `:focus-within`; touch.css:163-164 drops only the tool names.
- Fix: `.card:not(:has(:focus-visible)) .tools::before { display: none }` in the touch tier. This becomes moot under P17 on the phone, but the tablet needs it.
- Files: web/styles/touch.css. Effort: S.

**P24 - Medium - Pressing the active tab does nothing; "Top" is a 54x36 pill over the cards.**
- Evidence: touch.js:68-71 `setPhoneView` is a no-op when the view is unchanged. `#btn-gallery-top` 54.2x36 sits bottom-right over the cards' ⌦ position.
- Fix: a press on the selected tab scrolls that view to the top (reuse `goToGalleryTop`, which honours reduced motion). Keep the pill at 44 px, bottom-centre, shown after 3 screens. Add the re-press to rules.md's list of presses.
- Files: web/touch.js, web/styles/touch.css, web/gallery.js. Effort: S.

**P25 - Low - Filters are checkbox labels and search help is a two-thirds-screen tip.**
- Evidence: `#fav-only` and `#all-models` 20x20 in 74x36 and 123x36 labels. `gallery-07-search-help-tip.png` covers y 215-790 with 25 non-tappable lines.
- Fix: the labels become 44 px toggle chips (ids kept). A tappable token sheet with recent and saved searches is optional.
- Files: web/styles/touch.css, web/index.html. Effort: S.

**P26 - Low - The scope of bulk buttons is stated only in a tooltip.**
- Evidence: with nothing picked, More seeds… acts on the filter, and the title never shows on touch. Each action opens a preview dialog stating scope and count before anything queues.
- Fix: labels carry the scope on the phone ("More seeds of all…", "of 3…"), or filter-scoped actions sit under a scoped heading in ⋯. Never hide held buttons without a documented exception.
- Files: web/gallery-select.js, web/more-seeds.js, web/tag-swap.js, web/refine-batch.js. Effort: S.

**P27 - Low - The empty-gallery Examples button opens a menu in the hidden Create view.**
- Evidence: gallery.js:191-198 → `viewerExamples` (viewer.js:643-648) → `showExamples(true)` with no `showCreateView()`; the panel is `display: none` in Gallery (touch.css:200). Found in code, not in a screenshot.
- Fix: call `showCreateView()` first and add this press to rules.md's list. Optionally a proper empty state with a "Create your first image" button.
- Files: web/viewer.js. Effort: S.

**P28 - Low - Infinite scroll gives no sense of position.**
- Evidence: `.gallery-break` day lines are not sticky, there is no scrubber, and the count scrolls away.
- Fix: sticky day headers under the sticky head (they overlay, so nothing moves). A date fast-scroll thumb is optional. A 44 px full-width Load more retry.
- Files: web/styles/touch.css, web/styles/gallery.css, web/gallery.js. Effort: M.

### C. Viewer and lightbox

**P29 - High - The lightbox action row runs off screen, Inpaint and More seeds are hidden, and the JS uses the wrong breakpoint.** Merged: viewer, targets, visual, architecture.
- Evidence: measure-gallery: `#lb-delete` at x 348-433 on 393 in every portrait state (`gallery-20-lightbox-fit.png`). The row does scroll (`gallery-24`), but with no cue (`scrollbar-width: none`). Root cause: `.lb-actions` is a flex item with no `min-width: 0`, so it grows to 369 px inside a 354 px column. lightbox.css:318-323 hides `#lb-inpaint` and `#lb-more-seeds` at 900 px or less. lightbox.js:126 `lbNarrow` is `(max-width: 900px)`, true on an 852x393 landscape phone where touch.css lays the lightbox out as a row, and lightbox-compare.js:203 does the same.
- Fix: interim: `.lb-actions { flex: 1 1 100%; min-width: 0; flex-wrap: wrap }` with 44 px buttons, and stop hiding Inpaint and More seeds on the phone. Add `LB_STACKED_MEDIA` to state.js and use the same text in lightbox.js, lightbox-compare.js, lightbox.css and touch.css. Full fix: the 5-slot bar from section 2.5 (P30). Update rules.md's "a row that scrolls sideways".
- Files: web/lightbox.js, web/lightbox-compare.js, web/state.js, web/styles/lightbox.css, web/styles/touch.css. Effort: M.

**P30 - High - The picture gets about 40% of a portrait screen.**
- Evidence: `gallery-20-lightbox-fit.png`: a 1280² image drawn ≈340x340 (y 14-383). Below it: the prompt, 8 fact rows, the Compare row, the Fit/1:1/Download row, the action row and the counter. A 2:3 portrait would be about 245 px wide. lightbox.css:296-311; touch.css:113-118 gutters.
- Fix: the immersive layout from section 2.5: full-bleed on #000, overlay top and bottom bars, a tap toggles the chrome (instead of a backdrop tap closing), swipe up for facts. `lbFactsRoom`/`--lb-room` get a phone branch. Update the rules.md gesture text.
- Files: web/styles/lightbox.css, web/styles/touch.css, web/lightbox.js, web/index.html. Effort: L.

**P31 - High - Leaving the lightbox: a 26x26 ✕ in the bottom-right corner, and Back leaves the app.** Merged: viewer, targets (history itself is P64).
- Evidence: `#lb-close` 26x26 at (355, 814), 12 px from `#lb-delete`, over the ghosted tab bar. lightbox.css:225-231 `.close` padding 2px 4px; :302 orders `.lb-head` last. There is no pushState or popstate anywhere in web/*.js. Swipe-down (90 px) works but nothing hints at it.
- Fix: now, a 44x44 `.close` in the touch tier (`min-width`/`min-height: var(--h-md); display: inline-grid; place-items: center; padding: 0`), which also fixes the dialogs' ✕ (P47). With P30, a 44 px ‹ Back top-left, Back through history (P64), and swipe-down kept.
- Files: web/styles/touch.css, web/styles/lightbox.css, web/lightbox.js. Effort: S, then part of P30.

**P32 - Medium - Facts are a desktop list: seven 24 px ⌕ buttons touching 15-50 px links.**
- Evidence: measure-gallery: `button.lb-search` 24x24 with 0-8.9 px gaps on rows 22 px apart (so neighbouring targets overlap). "#3" is 14.7x24; "Use seed" 51.6x24 is 0 px from its ⌕ (lightbox.css:116-127, the 24 px WCAG floor).
- Fix: a facts pull-up sheet with 48 px rows; a tap opens Search / Copy / Use seed / Open source. This is a phone branch only, since `factsList` is shared with the desktop.
- Files: web/lightbox.js, web/styles/lightbox.css, web/styles/touch.css. Effort: M.

**P33 - Medium - Fit, 1:1, the readout and Download take a row of mixed heights for what gestures already do.**
- Evidence: Fit 39x44, 1:1 42x44, Download 72x36 (`gallery-20`). Pinch and double-tap already zoom.
- Fix: hide `.lb-zoom-group` on the phone and show the readout as a transient chip. Share via `navigator.share({files})` where `canShare` allows it, falling back to the download link on plain-http LAN.
- Files: web/styles/touch.css, web/lightbox.js, web/index.html. Effort: S.

**P34 - Medium - Swipe is the only way between images, with no hint, and the counter is in the corner.**
- Evidence: touch.css:256 `.lb-nav { display: none }`; "2/15 · #16" bottom-right beside the ✕.
- Fix: the counter centred in the top bar; a one-time 24 px nudge on the first open (skipped under reduced motion); an optional 44 px filmstrip.
- Files: web/styles/touch.css, web/lightbox.js. Effort: M.

**P35 - Medium - Landscape lightbox: the actions sit below the fold of a 300 px column.**
- Evidence: `gallery-29-lightbox-landscape.png`: `#lb-fit` and `#lb-download` are at y 424-428 of 393 (not in view); `.lb-side` is `min(300px, 42vw)` with its own scroll (touch.css:302-312); `#lb-close` is 26x26.
- Fix: the immersive landscape layout: an action rail on the right, facts as a side sheet.
- Files: web/styles/touch.css, web/styles/lightbox.css, web/lightbox.js. Effort: M.

**P36 - Low - The page shows through the lightbox.**
- Evidence: `gallery-20` and `gallery-26`: tab labels, card glyphs and "#16 1280×1280 · seed 11" show through. lightbox.css:23 is 0.97 alpha with a transparent `::backdrop`.
- Fix: phone only: `.lightbox, .lightbox::backdrop { background: #000 }`. Nothing else needs hiding, since the dialog is in the top layer.
- Files: web/styles/touch.css. Effort: S.

**P37 - Low - The key letter "S" shows on Compare with source (a specificity bug).**
- Evidence: lightbox.css:187 `:is(#lb-compare-on[aria-keyshortcuts="S"], …)::after` at (1,1,0) beats touch.css's `#lb-compare-on::after { content: none }` at (1,0,0). Visible in `gallery-20`.
- Fix: `#lb-compare-on[aria-keyshortcuts]::after, #lb-compare-cfg[aria-keyshortcuts]::after { content: none }` in touch.css.
- Files: web/styles/touch.css. Effort: S.

### D. Menus

**P38 - High - The Examples and Picked menus run off screen or use 29 px rows.** Merged: dialogs, create-ia, targets, architecture.
- Evidence: measure-create "examples menu open": scrollWidth 413 against innerWidth 393, so the page scrolls sideways (`create-10-examples-menu.png`; fields.css:298-304 `position: absolute; left: 0; width: var(--menu-w)`). Sample text is cut and max-height is 320 px. Picked ▾ rows are 159.5x28.8 (`gallery-18`); touch.css's row rule covers only `.context-menu li`.
- Fix: stopgap: `.menu { position: fixed; left: 8px; right: 8px; width: auto; bottom: calc(var(--tabbar-h) + var(--safe-b) + 8px); top: auto } .menu li { min-height: 44px }`. Then the shared action sheet, with Saved setups… pinned first in Examples.
- Files: web/styles/fields.css, web/styles/touch.css, web/prompt-examples.js, web/gallery-select.js. Effort: S.

**P39 - Medium - The long-press menu is a 17-row, 658 px desktop popover anchored at the finger.** Merged: gallery, targets, viewer, dialogs, platform, architecture.
- Evidence: `gallery-31-card-context-menu.png`: 17 rows of 190x36, no scrim, Move to recycle bin directly under Open PNG. context-menu.js:292-294 places it at the finger; touch.css:131 sets 36 px rows. In the lightbox (`gallery-26`) a leading orphan separator appears where Open is hidden. This is today's documented design and it works, so it is a design change rather than a breakage.
- Fix: an early return in the placement code under `isPhone()`, and the action sheet (section 2.6): a header naming the image, 48 px rows, groups, Bin last in red, Cancel, no "Open" (a tap opens), and a Select item (P18). Keep `contextMenuPressClick`. Fix the orphan separator now. Raising rows to 44 px is a cheap first step.
- Files: web/context-menu.js, web/styles/context-menu.css, web/styles/touch.css. Effort: M.

### E. Dialogs and sheets

**P40 - Critical - Inpaint's mask settings sit off the right edge, and its sliders are 16 px tall.** Merged: dialogs, targets.
- Evidence: measure-gallery "dialog inpaint": "Only the masked area" at x 364-512, `#inpaint-crop-pad` at 562-631 and Show mask at 655-748, past a 377 px limit, with no sideways scroll. These cannot be used (`gallery-dlg-inpaint-open.png`). `#inpaint-size` is 77.7x16 and `#inpaint-feather` 59.1x16 with the UA thumb (no range styling anywhere). Brush and Eraser are 47.7/49.9x36 at a 4 px gap. Cause: inpaint.css:64-82 `.ip-tools` has two fixed-height `nowrap` rows.
- Fix: step 1 (S): in the coarse block, `.ip-tools { height: auto }` and `.ip-row { flex-wrap: wrap; height: auto; white-space: normal; row-gap: var(--sp-2) }`, 44 px tools, and each slider on its own full-width row (44 px track, 28 px thumb via `::-webkit-slider-thumb`/`::-moz-range-thumb`). The wrap depends only on viewport width, so the canvas still never resizes with what is painted (inpaint.css:34-39). Brush|Eraser becomes a 2-segment control. Step 2: the strength paragraph goes behind its ?. Step 3 (optional): canvas pinch-zoom, with mask-coordinate tests.
- Files: web/styles/inpaint.css, web/styles/touch.css. Effort: S (step 1), L in total.

**P41 - High - Inpaint, Outpaint and Edit end 18 px past the bottom of the screen.** Merged: dialogs, platform, architecture.
- Evidence: measure-gallery `modalPastViewportBottom` is true for all three (top 34 + height 836 = 870 on 852). In `gallery-dlg-inpaint-bottom.png` Cancel and Inpaint are cut at the foot (still pressable). Cause: touch.css:270-273 `dialog:not(.lightbox):not(.context-menu)` (0,2,1) replaces `.inpaint-dialog`'s max-height but keeps inpaint.css:15 `margin-top: var(--ip-top)` (34 px).
- Fix: one rule at matching specificity, for example `dialog.inpaint-dialog:not(.lightbox):not(.context-menu) { max-height: calc(100dvh - var(--ip-top) - var(--safe-t) - var(--safe-b) - var(--sp-2)) }`. Add a smoke assertion that each dialog's rect is inside the viewport.
- Files: web/styles/touch.css. Effort: S.

**P42 - High - Outpaint's cross layout overflows 393 px, so Right padding cannot be set.**
- Evidence: measure-gallery: `#outpaint-right` at x 546-624 (entirely off screen), `#outpaint-bottom` 307-385, crop pad 355-424. Number boxes are 77.6x36 and presets 48-99x36 (`gallery-dlg-outpaint-grown-25.png`: the frame is cut on the right).
- Fix: the frame preview at full width, then Top/Right/Bottom/Left as a 2x2 grid of 48 px fields with 44 px steppers (`min-width: 0`). The presets become a scrolling 44 px chip row that applies on tap, using the server's own preset ids. Band, "Only the masked area" and Pad go under Advanced. A sticky footer reads "1280×1280 · Outpaint".
- Files: web/styles/outpaint.css, web/styles/touch.css, web/index.html, web/outpaint.js. Effort: M.

**P43 - High - There is no phone dialog pattern: the only rule is a max-size clamp.** Merged: dialogs, platform, architecture.
- Evidence: touch.css:269-273 is the only touch dialog rule. The 16 dialogs are rounded cards inset 8 px with 18 px padding (about 341 px of content) and actions at the end of the scroll. Closing varies: a 26 px ✕, a 47x36 "Close", or only a bottom Cancel. There are 11 different action-row classes and none is sticky. 13 of 17 dialogs fit; the umbrella problem is the missing pattern.
- Fix: step 1 (CSS only): under `(pointer: coarse) and (max-width: 600px)`, `inset: 0`, no radius, a sticky existing head and a sticky existing action row (list `.rf-actions, .swap-actions, .rb-actions, .ms-actions, .confirm-actions, …`), and `overscroll-behavior: contain`. Keep inpaint's top at 0 under the sheet. Step 2: a shared `.dialog-actions` hook and a uniform 44 px header ✕ (index.html). Record the pattern in rules.md.
- Files: web/styles/touch.css, web/index.html, web/styles/refine.css, web/styles/inpaint.css, web/styles/bundles.css. Effort: M.

**P44 - High - The keyboard can cover dialog actions: several dialogs focus a text box on open.**
- Simulator follow-up (2026-11-07): Refine now opens without the keyboard on the phone, but focusing Strength can still leave Cancel and Refine behind it on an iPhone 15 Pro in portrait and on iPad Air 11-inch (M2). Native rectangles and screenshots confirm the overlap. That pre-fix failure is retained in the [Safari Simulator check](#6-safari-simulator-check-2026-11-07); Refine's correction is verified in [section 7](#7-refine-keyboard-correction-2026-11-07). The other edit dialogs' scoped native checks are recorded in [section 10](#10-edit-dialog-keyboard-acceptance-2026-11-07); its physical-device and caret gates remain open.
- Original review evidence (2026-11-05): autofocus at index.html:1280 (`#refine-strength`), :2038, :2078, :2123; `.focus()` calls at refine.js:302, tag-swap.js:145, more-seeds.js:331. Dialogs are sized in dvh, which iOS does not shrink for the keyboard. Then inferred, not captured; the Simulator follow-up above now supplies native evidence.
- Fix: on coarse pointers, no autofocus on text boxes (move focus to the sheet's header or ✕); the desktop keeps its reasons. A visualViewport listener in touch.js sets `--kb-inset` while a sheet is open, and the sticky footer uses `bottom: var(--kb-inset, 0)`. The same inset lets the Create action bar ride above the keyboard (P1).
- Files: web/touch.js, web/styles/touch.css, web/index.html, web/refine.js, web/tag-swap.js, web/more-seeds.js. Effort: M.

**P45 - High - The Models sheet is a squeezed desktop table in a 439 px inner scroll.** Merged: dialogs, platform, targets, visual, architecture.
- Evidence: `create-20-models-sheet*.png`: 4198 px of content in a 439 px box. The header "Model / Style Refine / Size Speed" wraps to three lines. Four dotted links per row; star 36x36; link 32x21; radio and Group-by-type 20x20; 128 of 153 targets small. "Made with…" and "Load automatically" take about 260 px. It is non-modal (`topModal` null), but light dismiss works (app.js:286-291), so a tap outside closes it.
- Fix: section 2.6. One scroller, a Done header, card rows whose whole row is the radio, chips instead of dotted links (the verbatim sources reachable from one ⓘ), a sticky footer with Apply & save, Switch & load and Unload. `inert` on main while it is open, on the phone only. Keep the element (not a dialog) because of the focusout-close logic and app.js:729/761 guards. Keep the table cells the desktop smokes read.
- Files: web/styles/model-picker.css, web/styles/touch.css, web/model-switch.js, web/model-picker.js, web/index.html. Effort: L.

**P46 - Medium - Refine's Seed box and dice run off the right edge.**
- Evidence: measure-gallery: `#refine-seed` 246-392 and `#refine-dice` 400-442 against the limit. fields.css:101 `min-width: min(calc(16ch + 26px), 45%)`, plus Compare widening Strength. `#refine-compare-scales` 92x32 sits at a 0 px gap.
- Fix: in the coarse block, `.rf-strength .row { flex-wrap: wrap }` with Strength at 100%, Versions and Seed plus the dice sharing the next row, and the min-width overridden. The scale pickers become a 44 px 4-segment control.
- Files: web/styles/touch.css, web/styles/refine.css, web/styles/refine-batch.css. Effort: S.

**P47 - Medium - Dialog ✕ buttons are 26x26 glyphs.**
- Evidence: `#refine-close`, `#inpaint-close`, `#outpaint-close` and `#edit-close` are 26x26; `#analysis-close` is 28.3x28; `.close` has no touch override (lightbox.css:225-232).
- Fix: the same one rule as P31.
- Files: web/styles/touch.css. Effort: S.

**P48 - Medium - Saved setups, searches and snippets use native list boxes and a crowded one-screen editor, with two text bugs.** Merged: dialogs, visual.
- Evidence: `create-12-saved-setups-dialog.png`: a native `<select size>` inside two inner scrolls (three scrollbars); `#setups-all` is 13x20. Names are cut with "…". The snippets dialog's captions above New/Save/Delete are clipped to a sliver. "Export./Import." is literal in index.html:2066-2067. iOS draws a multi-row select as a one-line picker.
- Fix: now: correct "Export…" and "Import…" and give the caption row auto height. Then list-then-edit (section 2.6), with the `<select>` kept hidden for smoke-generation-presets, smoke-saved-searches and smoke-snippets. Restore, Undo and import rules are unchanged.
- Files: web/index.html, web/styles/snippets.css, web/styles/setups.css, web/styles/saved-searches.css, web/styles/touch.css, web/generation-presets.js, web/saved-searches.js, web/prompt-snippets.js. Effort: L (text fix S).

**P49 - Medium - Swap tags opens with the keyboard up, small row controls and an empty results box.**
- Evidence: tag-swap.js:145 focuses Find. `.swap-at` 62x32, `.swap-remove` 24x32, radios 20x20, a 100-word paragraph and an empty ≈320 px box push Swap below the fold (`gallery-dlg-swap-tags-open.png`).
- Fix: each row as a card (Find over Replace at 48 px, a 44 px ✕), a Prompt | Negative segmented control, the explanation behind a disclosure, a results box with no min-height, and the sticky "Swap in 3 images" footer.
- Files: web/styles/touch.css, web/index.html, web/tag-swap.js. Effort: S.

**P50 - Medium - ? help badges: 14 px drawn, and their 38 px press zone covers the field below.** Merged: targets, dialogs.
- Evidence: fields.css:204-210 draws 14x14; touch.css:96 `.help::before { inset: -12px }`. The Steps/CFG/Mu/True CFG badges sit 6 px above their boxes, so the zone covers about the top 6 px of each box. Outpaint's strength ? is 4 px from a label. About 25 badges in all.
- Fix: `inset: -12px -12px 0` so the zone never reaches the field below. Optionally draw 20-24 px. Inside sheets, show the tip inline below its label (`position: static`), and put the long strength paragraphs behind their ?.
- Files: web/styles/touch.css, web/styles/fields.css, web/styles/inpaint.css. Effort: S.

**P51 - Low - Confirm dialogs put Cancel and the destructive action about 190 px apart.**
- Evidence: `gallery-dlg-empty-bin-confirm-open.png`. The 16 px off-centre is the emulator's scrollbar, not the CSS.
- Fix: optionally bottom-anchored with stacked 48 px buttons. The confirmations themselves stay.
- Files: web/styles/touch.css. Effort: S.

**P52 - Low - Bundles, Mass refine and Then refine use 20 px radios and checkboxes and 36 px scale buttons.**
- Evidence: radios 20x20 in 88-106x23 labels, `#bundles-close` 47x36, a 335x21 native file input, scale buttons 37-52x36 at 6 px gaps.
- Fix: 44 px label rows, segmented controls, a 48 px "Choose bundle…" button, and Then refine as a switch row. Holds stay aria-disabled.
- Files: web/styles/touch.css, web/styles/bundles.css, web/styles/generation.css, web/index.html. Effort: S.

**P53 - Low - Explanatory prose dominates dialogs and Check.**
- Evidence: `gallery-dlg-inpaint-open.png`: a 5-line paragraph about 0.95 (≈300 device px) before the folds; Check's notes run 2-4 lines each.
- Fix: one line inline plus a "Why?" that holds the verbatim text, keeping the result-bearing number and the 16 GB card qualifier (`card_words`). The sticky footer (P43) keeps the actions visible.
- Files: web/styles/inpaint.css, web/styles/outpaint.css, web/styles/instruction-edit.css, web/styles/analysis.css, web/styles/touch.css. Effort: M.

### F. Visual system and touch targets

**P54 - High - The touch tokens bake in sub-44 sizes and never set widths.** (targets, systemic)
- Evidence: touch.css:30-33 `--h-xs` 32, `--h-sm` 36, `--h-tool` 36. The only touch min-width rule is `#btn-saved-searches { min-width: 0 }`. 40 of 55 Create controls are under 44 at load, 54 of 84 with the folds open, and about 120 per Gallery screen.
- Fix: in the phone query's `:root` (not the shared coarse one, so the tablet is untouched), set all three to 44 px. Add `min-width: 44px` per component (icons, presets, queue, `.close`), not globally, and exclude `.lb-link` and `.analysis-link`. Land it with P17, otherwise the card row overflows (6x44 > 180 px), and check `.viewer .tile > .tools`' fixed height (gallery.css:94). Add the audit check (P69). Update rules.md ~6582 and smoke-phone's 36/32 px assertions.
- Files: web/styles/touch.css, tools/smoke-phone.json, docs/rules.md. Effort: M.

**P55 - High - The phone keeps the desktop type scale: most text is 12-13 px next to 16 px inputs.**
- Evidence: touch.css:37-38 changes only `--fs-xs` (11) and `--fs-sm` (12). `--fs-label` 12 is used by 93 rules and `--fs-body-sm` 13 by 61. 10 px literals at fields.css:209, generation.css:69 and analysis.css:223. Uppercase 12 px captions (`create-01-load-top.png`); queue and status text at 12-13 px (`create-16`); facts at 13 px.
- Fix: the scale in section 2.7 on the phone `:root`, `--lh-tight` 1.25, the literals replaced by a token, and no uppercase or letter-spacing on `.field > span` and `.section > summary` on the phone. Update the style.css token comment and rules.md.
- Files: web/styles/touch.css, web/styles/fields.css, web/styles/generation.css, web/styles/analysis.css. Effort: M.

**P56 - Medium - No button hierarchy: selected presets share Generate's solid accent and everything else is the same outline.**
- Evidence: 1:1 and 1 MP are solid `--accent-strong`, like Generate (`create-01`). Examples, Check, Reset, Set True CFG, Show run, More seeds and Finish run are identical ghosts. Delete unstarred has its own top row (`gallery-01`). Disabled buttons are visibly dimmer, so that part was overstated.
- Fix: section 2.7 tiers. Selected toggles in `--accent-soft` with an accent border; tonal, text and danger tiers. Held buttons stay visible.
- Files: web/styles/touch.css, web/styles/generation.css, web/styles/gallery.css. Effort: M.

**P57 - Medium - No spacing rhythm, and nested frames waste 52 px of width.**
- Evidence: aspect pairs at 0 px; ragged row ends (≈305 and ≈287 of 351); a ≈45 px hole before the orphan "1.0 MP"; section captions as close to the previous section as to their own fields; footer buttons 4-8 px apart; about 80 px of dead space in Snippets; `--gutter` 12 plus a bordered panel padded 14.
- Fix: the rhythm in section 2.7 with `--sp-5` and `--sp-6`. On the phone `.panel` has no border, background or padding, and the gutter is 16.
- Files: web/styles/touch.css, web/styles/generation.css, web/styles/fields.css, web/styles/snippets.css. Effort: M.

**P58 - Medium - Rows break badly when they wrap.**
- Evidence: Set True CFG orphaned on its own line (`create-01`); More seeds…/Finish run on an indented second line (`create-18-finished-full`); "Hi…" (Hidden tabs only) at 320 px; the topbar's "FLUX.… 20/28 …"; Check's mid-paragraph Show; inpaint's "nothing masked yet - paint the p…"; "Run 7615… nothing in th…".
- Fix: caption rows hold a label and at most one trailing action, with notes on their own line. Status sentences go full width with actions on a row below, starting at the left edge. Instructions wrap to 2 lines with a "more" toggle, never ellipsize. Reserved heights are kept.
- Files: web/styles/touch.css, web/styles/fields.css, web/styles/generation.css, web/styles/analysis.css, web/styles/inpaint.css, web/styles/sheet.css. Effort: M.

**P59 - Medium - Folds and section headings use four different styles.**
- Evidence: "GRID SEARCH" uppercase, muted, with a 10 px ▸ at 0.7 opacity (generation.css:56-69); "Custom dimensions" and "LoRA" bold with a large ▶ (fields.css:14); Step cache an inline label; dialog summaries uppercase. The fold summaries are 18 px drawn, reaching about 36 px only by a pseudo-element.
- Fix: one phone disclosure row for every `<details>`: 48 px, a hairline above, the label at `--fs-md` 600, the value right-aligned (up to 2 lines), a 16 px rotating chevron. Non-folding section headings get a distinct style. Step cache stacks its label. This replaces the pseudo-element reach.
- Files: web/styles/touch.css, web/styles/generation.css, web/styles/fields.css, web/styles/model-picker.css, web/styles/inpaint.css. Effort: S.

**P60 - Medium - Checkboxes and radios are fixed at 20x20.**
- Evidence: touch.css:76. `#notify-done`, `#grid-resume`, `#model-group-by-type`, 22 model radios, `#refine-lock` and the bundle and swap radios are all 20x20; bare labels 23 px; `#setups-all` 13x20. `label.check` rows are 36 px. Accent-color is already set (style.css:113).
- Fix: booleans as 51x31 switches in 44-48 px full-width rows; 2-3 option radio groups as 44 px segmented controls; `.check, label:has(> input) { min-height: 44px }`. Keep the native inputs underneath.
- Files: web/styles/touch.css, web/styles/setups.css, web/styles/swap.css, web/styles/bundles.css, web/styles/model-picker.css. Effort: M.

**P61 - Medium - There is no pressed state anywhere.**
- Evidence: no `:active` in style.css or styles/*.css. Hover fills are removed on touch; only a 25% tap highlight remains.
- Fix: background-only `:active` state layers from tokens on buttons, presets, tabs and menu rows (the primary darkened with color-mix). Then a transparent tap highlight, and a passive no-op `touchstart` listener so iOS applies `:active`. No scale transform.
- Files: web/styles/touch.css, web/touch.js. Effort: S.

**P62 - Low - Text-link actions: "Then refine: off" is 80x18 and fold summaries are 18 px.**
- Evidence: `#then-refine-open` 80x18 (pseudo-element to ≈38); summaries 324x18 (≈36); finish-note buttons 33.5 px at 8 px gaps (not touching, as first claimed).
- Fix: real 44-48 px rows on the phone (P59, P60); finish-note buttons at 44 px or moved (P7). Citation links stay as the documented exception.
- Files: web/styles/touch.css, web/styles/generation.css. Effort: S.

**P63 - Low - Tab bar: weak selected state, an 8 px busy dot, and translucency.**
- Evidence: `create-21-tabbar.png`: only the text and glyph colour change; an 8x8 dot; 22 px glyphs; a 94% background with blur, through which the presets read in landscape.
- Fix: a 56x30 active-indicator pill, 24-26 px glyphs, an opaque `var(--panel)` background, and the busy state carried by the action bar (P1/P3). The badge count goes in the accessible name.
- Files: web/styles/touch.css, web/touch.js, web/index.html. Effort: S.

### G. Platform

**P64 - High - Back is not integrated: there are no history entries for overlays.** Merged: platform, viewer.
- Evidence: no pushState, replaceState, popstate or location.hash in web/*.js. Views switch by sessionStorage only. Chrome's CloseWatcher closes showModal dialogs on Android Back, but not the Models section, the context menu or the run sheet, and iOS's edge swipe leaves the page from any layer.
- Fix: a phone-only history layer in touch.js. `pushState({layer})` on opening the lightbox, a sheet, Models, an action sheet or the run sheet; on `popstate`, close the top layer with its own close function; on any other close (✕, Escape, swipe-down), call `history.back()` only if its own state is on top, so no orphan entries pile up. Tabs use `replaceState`. Guarded by `isPhone()`. Add a smoke step that calls `history.back()`.
- Files: web/touch.js, web/lightbox.js, web/model-switch.js, web/context-menu.js, web/sheet.js. Effort: M.

**P65 - Medium - Phone landscape is the portrait column stretched to 852 px.** Merged: visual (high), create-ia (low), platform (low).
- Evidence: `create-23-landscape-top.png`: one 790 px column, only the prompt and negative, and presets showing through the 44 px tab bar. `create-23-landscape-scroll-2.png`: Generate is 790x50, Then refine is flung 640 px right, and Step cache is 715 px wide. The Gallery leaves about 140 px for images.
- Fix: after the portrait work. Now: an opaque tab bar, the action bar (P1) in landscape, `max-inline-size: 560px` centred for single-column content, and the gallery head in one row under the existing landscape query. Later, a two-pane Create that scrolls as the page (no inner scroller on the panel). No navigation rail. The tablet query and the one-column exclusions are unchanged.
- Files: web/styles/touch.css, web/styles/generation.css, web/styles/gallery.css. Effort: M (two panes L).

**P66 - Low - Pull-to-refresh is unmanaged, and the prompt draft does not survive a reload.**
- Evidence: no `overscroll-behavior` on html or body; no draft persistence (the inner scrollers already contain their overscroll).
- Fix: `html { overscroll-behavior-y: contain }` on the phone, plus `overscroll-behavior: contain` on sheets. Optional: a per-tab draft in sessionStorage (try/catch), restored through `writeBox` only into empty boxes before any Reuse or Load PNG.
- Files: web/styles/touch.css, web/touch.js, web/panel.js. Effort: S.

**P67 - Low - The home-screen PWA is a browser tab by design.**
- Evidence: the manifest's `display: "browser"`; index.html:5-12 explains omitting `apple-mobile-web-app-capable` (downloads need Safari around them).
- Fix: keep the default. A standalone mode only for HTTPS setups, routing Download through the share sheet (`navigator.share` needs a secure context). Update index.html and rules.md together; sw.js still caches only the offline page.
- Files: web/manifest.webmanifest, web/index.html, web/lightbox.js. Effort: M.

### H. Architecture and tests

**P68 - High - The phone tier only overrides desktop components, so it cannot give the phone its own structure (the root cause).**
- Evidence: touch.css is 327 lines of tokens, min-heights, pseudo-elements and `display: none !important`; touch.js (90 lines) only switches `body[data-view]`. The DOM order is the desktop's: the panel at index.html:157, Generate at :502 mid-panel, grid, step cache and LoRA after it, the stage and viewer at :795/:815. Note: `.panel` is already a flex column (style.css:395-403), so sticky positioning works without moves.
- Fix: stage it. Use CSS where no order changes (sticky Generate, collapsed folds). Then phone hosts (action bar, More options, Queue and Check sheets), filled by a `phone.js` that moves nodes with `moveBefore` and restores them on a media change. Amend AGENTS.md ("touch.css and styles/phone-*.css, linked last"), rules.md ("nothing is removed, only not drawn" becomes "moved, never copied, on the phone"), docs/maintainer/source-audit.md, and add phone.js to the classic-script order guard.
- Files: web/touch.js (+ web/phone.js), web/styles/touch.css, web/index.html, AGENTS.md, docs/rules.md, docs/maintainer/source-audit.md. Effort: L (spread across phases).

**P69 - Medium - The phone smokes emulate a desktop window and assert none of the complaints.**
- Evidence: tools/browser.mjs:468-469 sends `mobile: false`, so there is a 15 px scrollbar (scrollWidth 378 against 393). smoke-phone checks overflow, the panel bottom and one wrap. It opens no dialog and checks no target size. P29, P38, P40 and P41 all pass today.
- Fix: an opt-in `"mobile": true` on the viewport step (the default stays false). A page-side audit helper that fails on visible controls whose press box is under 44x44 (with a listed exception set), controls past the viewport, and dialogs outside 0..innerHeight. New rows: smoke-phone-create, smoke-phone-gallery, smoke-phone-dialogs and smoke-phone-landscape on ports 7960-7963 with fresh scratch `--data`, plus their costs in run_smoke_costs.json. tools/ changes are not CI-definition changes.
- Files: tools/browser.mjs, tools/smoke-phone.json, tools/smoke-phone-*.json, tools/run_smoke_costs.json, AGENTS.md (smoke table), docs/rules.md (Tests). Effort: M.

**P70 - Medium - JS computes the visible area as topbar to innerHeight and ignores the bottom chrome.**
- Evidence: job-results.js:521-523 (`viewerOutOfSight`, `innerHeight - 48` while the tab bar is 56 + safe-b), app.js:706, reuse.js:350-351/364, gallery-search.js:19, gallery-select.js:549-550, panel.js:555-602, sheet-render.js:917-918, tag-chips.js:89.
- Fix: a `viewBand()` helper in state.js that subtracts a `--phone-bottom-chrome` custom property (tab bar plus action bar) and returns `innerHeight` when it is 0, so the desktop is identical. Adopt it in the reveal and visibility checks first. Prerequisite for P1.
- Files: web/state.js, web/job-results.js, web/reuse.js, web/app.js, web/panel.js, web/gallery-select.js. Effort: M.

**P71 - Low (prerequisite) - The tier never re-syncs on a media change.**
- Evidence: touch.js reads `isPhone()` on demand; there is no `change` listener on `PHONE_MEDIA` (app.js:197 does listen on `BESIDE_MEDIA`).
- Fix: one listener that moves nodes in and out of the phone hosts (keeping focus), closes phone sheets and restores the per-view scroll. smoke-tablet steps resize 678→820→678 and check that Generate and the viewer are shown and focusable.
- Files: web/touch.js, tools/smoke-tablet.json. Effort: S.

**P72 - Low - touch.css is one cascade-order monolith held together by `!important` and specificity.**
- Evidence: `!important` at touch.css:72, :141 and :200-202; the dialog clamp winning by accident (P41); width-only narrow rules that phones also hit (style.css:325/335/358/423, gallery.css:382, model-picker.css:136/144, …); coarse rules scattered in sheet.css:202/308 and gallery.css:144.
- Fix: phone rules stay unlayered and linked last (no `@layer`, which would lose to the unlayered desktop CSS). Add a `body[data-tier="phone"]` hook set from `PHONE_MEDIA`, plus `:where()` for predictable specificity. Split into `styles/phone-*.css` as the work lands. List in rules.md the width-only rules that phones also hit.
- Files: web/styles/touch.css, web/index.html, docs/rules.md, docs/maintainer/source-audit.md. Effort: M.

---

## 4. Implementation plan

Constant across phases: every rule sits under `PHONE_MEDIA` (or the coarse tier where the tablet should share it). The 1400x900 mouse smokes and the tablet's two columns do not change. Per AGENTS.md, each phase rewrites the parts of `docs/rules.md` "Phones and tablets" it changes (and Tests for new smokes), in the same commit as the code. New smoke rows go into the AGENTS.md table. A `web/` change needs no server restart. GPU-free: this is all browser work against the fake server.

| Phase | Findings | Size | Rules and smokes touched |
|---|---|---|---|
| **0. Enablers** | P69 (mobile flag, audit helper, four new smoke rows; start with a baseline that records today's failures as known), P71 (media-change sync), P70 (`viewBand`), P72 (`data-tier` hook, phone-*.css split), P64 (history layer), the sheet and action-sheet CSS primitives (P43 step 1, P39 base) | M-L (≈1 week) | AGENTS.md (where phone CSS lives, the smoke table), rules.md Tests and Phones and tablets ("moved, never copied"), docs/maintainer/source-audit.md, smoke-tablet (resize steps), the classic-script order guard (phone.js) |
| **1. Stop the breakage (quick wins)** | P41, P40 step 1, P29 interim (`min-width: 0`, wrap, unhide Inpaint and More seeds, `LB_STACKED_MEDIA`), P38 stopgap, P46, P31/P47 (44 px `.close`), P37, P21 hint, P23, P27, P36, P50 inset, P48 text bug and captions, P8 interim, P66, P14 | S each, ≈2-3 days in all | smoke-phone (dialog fit, lightbox action row, Examples overflow), rules.md (lightbox row, keyboard hints, data-typing, the phone dialog rule), the smoke-phone-dialogs row starts passing |
| **2. Visual foundation** | P54 together with P17 (tokens to 44 and card tools hidden in one change), P55, P57, P56, P59, P60, P61, P62, P58, P63 | M-L (≈1 week) | rules.md (heights, type tokens, card tools, pseudo-element reach, fold summaries), style.css token comment, smoke-phone ("card tools 32 px", "36 px rows"), smoke-tablet (unchanged values confirmed) |
| **3. Create structure** | P1 (sticky first, then the action bar), P2, P3, P8, P4, P5, P11, P12, P10, P6, P7, P13, P9, P15, P68 (hosts) | L (≈2 weeks) | rules.md (one column with a tab bar → action bar, data-typing for the bar, New image pill phone exception, Finish run not drawn, placeTagValidation phone branch, topbar, Unload in the sheet), smoke-phone-create, smoke-phone (Generate position, line 28), smoke-size-crop and smoke-results (desktop paths unchanged) |
| **4. Gallery** | P16, P18, P19, P20, P22, P24, P21 rest, P25, P26, P28 | L (≈1.5 weeks) | rules.md (footer at full names, long press adds Select, tab bar replaced while selecting, card view below 1800 px, tab re-press, bin screen), index.html placement comments, smoke-phone-gallery, smoke-phone, smoke-select and smoke-gallery (desktop unchanged) |
| **5. Viewer** | P30, P31, P29 full bar, P32, P33, P34, P35, P39 (action sheet) | L (≈1.5 weeks) | rules.md (lightbox gestures: tap toggles chrome, Back; context menu "16 px under the finger" and "36 px rows" → bottom sheet; the menu holding Inpaint and More seeds), lightbox.css header comments, smoke-phone-gallery, smoke-context-menu (phone branch), smoke-lightbox (desktop unchanged) |
| **6. Sheets and dialogs** | P43 step 2, P44, P45, P42, P40 steps 2-3, P49, P48 list-then-edit, P52, P51, P53, P9 sheet if not already done | L (≈2 weeks) | rules.md (the Models picker "between the topbar and the bar", the phone sheet pattern, the keyboard inset), smoke-phone-dialogs, smoke-generation-presets, smoke-saved-searches and smoke-snippets (their `<select>` kept), smoke-model-* (desktop unchanged) |
| **7. Landscape and platform polish** | P65, P35 if deferred, P67 (only if HTTPS is wanted), the remaining P63 | M | rules.md landscape paragraph, smoke-phone-landscape |

**Status (2026-11-06).** Phases 0 and 1 (2026-11-05), 2, 3 and 4 (2026-11-06) are built (PR #54, `phone/review-and-phase-0-1`), phase 5 (PR #55), phase 6 (PR #56) and phase 7 on `phone/phase-7-landscape`. Phases 2-3 at a glance (each step below has the detail):
- Done: P54, P17, P55, P57 (its size pairs and "1.0 MP" line went with P5's chip and Shape row), P59, P61, P62, P63 (the bar's picture rings a running job; the tab's dot stays), P52's leftovers, P1 step 2, P2, P3, P5, P6, P8, P10, P11, P15, P68 (the hosts in `phone.js`/`phone-compose.js`).
- Partly: P56 (Delete unstarred's row is P19/P24's, the red bin row P39's), P58 (Check's mid-paragraph Show waits for P9's cards), P60 (24 px ticks, no drawn switches or segmented radios), P4 (no Advanced split for Mu and True CFG), P7 ("Done · 9 s" not reworded), P9 (the sheet only; its cards and folds are phase 6's), P12 (Unload stays in the ⋯, not the Models sheet's footer), P13 (no Steps stepper), P72 (`phone.js`, `phone-compose.js`, `styles/phone.css`; no phone-*.css split).
- Skipped: P17's optional ⋯ per card (a node on every desktop card too; section 2.4 draws none); P23 no longer applies on the phone (the caption went with the card tools; the tablet keeps the fix).
- Done: P70, P71, P64 (the whole touch tier, the tablet too), P41, P46, P47, P37, P36, P23, P27, P14; and the parts the plan gave these phases: P29 interim, P40 step 1, P8 interim, P1 step 1 (sticky Generate; a refused press says why beside it), P21 hint, P50 inset (and the Grid search rows 8 px apart), P48 text bug and captions, P38 stopgap, P43 step 1, P39 base (docked, 48 px rows, no orphan line).
- Done ahead of phase 6: P42 by CSS alone (Left and Right beside the frame); the 2x2 steppers and chip row stay there.
- Partly done: P69 (the `mobile` flag, `inject` and `tools/phone_audit.js`; two new rows, `smoke-phone-dialogs` and `smoke-phone-gallery`, both on 7960 because the port table is full; smoke-phone-create and -landscape wait for their phases; under-44 px counts are logged, not asserted); P72 (the `data-tier` hook only); P31 (the 44 px close; the top-left Back is P30); P52 (24 px ticks on 44 px rows in dialogs; Swap tags' radios, scale buttons and switch rows left); P66 (pull-to-refresh stopped).
- Skipped: P66's sessionStorage draft (`persistPanel`/`restorePanel` already keep the prompt across a reload, which smoke-phone checks); P72's phone-*.css split and `phone.js` (nothing phone-only needed a file of its own yet).
- Phase 2, first step: P54 and P17 together. The phone's `--h-xs`, `--h-sm` and `--h-tool` are 44 (the tablet keeps 32/36 and its cards' tools), with 44 px least widths on small and tiny ghost buttons, the size pills, a close and the lightbox's Fit and 1:1; a card shows its star alone (a 44 px square), its tools and a long run's tile tools not drawn but kept for the long-press menu, the bin's two tools kept at 44; a result's tools 44 tall, 44 wide where the tile has room. `smoke-phone` asserts no control under 44 px on the Gallery view and Create's first screen (the whole Create view but Then refine's opener and the Grid search summary, P59/P62). Left of these: a ⋯ per card (optional in P17) and the 2x2 batch's tools, which share a 165 px tile until the result card (P5/P6).
- Phase 2, second step: P55, P57, P59 and P58. The phone's type scale is on its `:root` (13, 13, 14, 15, 16, 18, 20 px; no text under 13, which `__phoneText` in `tools/phone_audit.js` checks on Create and Gallery at 393 and 320 px), captions in sentence case at 14 px bold, `?` badges drawn 20 px; the panel frameless with a 16 px gutter, fields 16 apart, the results 32 below (`--sp-5`, `--sp-6`, `--lh-tight` are page tokens); every fold a 48 px row with its value right-aligned and a turning chevron, Step cache stacked; the negative caption, Refine's Size caption, the finished job's line and the Notify row wrap cleanly on the tablet too, and Inpaint's and the run sheet's state lines are two lines instead of an ellipsis. Left of P58: the topbar's job mirror (P12) and the check's mid-paragraph Show (P9); of P57: the size pairs' 0 px joins and the "1.0 MP" orphan (P5).
- Phase 2, third step: P56, P60, P61, P62 and P63, and P52's leftovers. Generate is the one solid accent on Create (a sheet's primary in its sheet); a chosen size, scale or toggle is tonal (`--accent-soft`, a 2 px accent edge, the text's colour); Run grid is secondary; `.ghost.danger`, the lightbox's and a result's Delete are `--danger` at rest. Every tick and radio on the phone is 24 px on a 44 px row (Swap tags' choices among them, 16 px apart), a sheet's scale buttons 8 px apart, Bundles' file picker 44 px. A press is a background-image state layer on the touch tier (the tablet too): 12% of the text's colour, black 20% over a primary, nothing over a held control, no tap flash, a passive `touchstart` for iOS. Then refine's opener is a 44 px box. The tab bar is opaque, the chosen tab's 24 px glyph in a 56x28 pill, a job's dot 12 px on Create's pill with "a job running" in the tab's name. The phone audit now asserts no control under 44 px on the whole Create view (no exception) and in every sheet `smoke-phone-dialogs` opens. Skipped: P60's drawn 51x31 switches (one tick look across the phone; a trailing switch clashes with the Notify row's grid and Then refine's opener) and its segmented controls for two- and three-choice radios (44 px rows already; the size pills are the segmented control); P56's "Delete unstarred has its own top row" is the gallery head's (P19/P24); the action sheet's red bin row is P39's.
- Phase 3, the action bar: P1 step 2, P2 and P8's remainder. The Create view has `#phone-actions` docked over the tab bar (66 px; 53 on its side, its row at most 560 px across the middle): the newest result's picture (the live preview ringed with the job's fraction while one runs; a press opens the newest image in the lightbox, or brings the live preview's tile into view), Generate moved into it by `phone.js` (`moveBefore`, home again off the phone, `body[data-tier]`'s sync), and the queue's count, whose press opens a plain Queue sheet holding the real `#progress` and `#queue-note` while it is open. Both held with their reasons; a held press says why above the bars. The viewer, its note and `#recent` are moved into `#phone-results`, Generate's place after Output, so the image just made starts about 950 px down rather than 1480; the idle one-line viewer is not drawn there. The bars step aside under `data-typing`; `--phone-bottom-chrome`, `main`'s padding, the scroll padding and the Models sheet count the action bar in the Create view. New image's pill is not offered on the phone. `smoke-phone-create` (7956, beside `smoke-phone`) holds it; `styles/phone.css` and `phone.js` are the phone's first files of their own (P72). Left for P3: the Queue sheet's own design (the 4-column job buttons, 56 px rows with ⋯); for P30: a live preview in the phone's own viewer, after which the picture's press during a job can open it rather than scroll.
- Phase 3, the queue and the result: P3, P6, P7 and P15. The Queue sheet holds the job and the panel's line (`#error`, so a held press there says why in the sheet) as well as the queue; the panel draws no progress or queue of its own on the phone. The job's name over its bars, then ↑ Front, Pause, Skip and Cancel as four equal 44 px columns 8 px apart, Cancel last in `--danger`; each queued job a 56 px row with one 44 px ⋯ whose docked menu (the image menu's element, `showContextMenu`, moved into the sheet) has Move to front and Cancel this job, the latter in `--danger`. A result's row is Reuse, Refine and Star with their words and a 44 px ⋯ for the image's own menu, Delete last in it in `--danger`; the markup's order is kept (Reuse, Refine, Star), not the sketch's Star first, so the focus order is the order drawn; under 270 px of tile (a batch of two stacked, 259 px) each word under its glyph, under 200 (a 2x2 batch) Star and ⋯ only. The finished line's Use seed and More seeds… and a held Finish run are not drawn, nor a lone result's seed line and More seeds; a batch's More seeds stays. A long run is a grid of plain pictures, two across. Skipped: P7's "Done · 9 s" rewording (the sentence stays as the PC's), and hiding `#last-run`'s buttons while the viewer shows that run, which cannot happen (it shows only at a load with nothing running). Left: the action sheet's scrim and Cancel row (P39); a ⋯ toggle (a second press reopens the menu rather than closing it).
- Phase 3, More options: P4, P11 and P13. Create's first screen is the prompt, the size, Batch, a More options row and the newest result (about 600 px down at 393x852, where it was 950). The row, a 56 px tonal card, says what its sheet holds ("28 steps · CFG 3.5 · seed random", then whatever else is set) with a dot while anything there is off the checkpoint's defaults; a press opens a full-screen sheet (`#more-sheet`, inside the panel so its events still reach the panel's listeners; its own history entry, Done in its title row) holding the panel's own Sampling, Seed, Repeat, Custom dimensions, LoRA & references, Notify and Then refine, Step cache (its box, inspectable) and Grid search, moved in by `phone.js` and home off the phone, with Run grid in its sticky foot while the grid fold is open; opening that fold brings its row up under the title, clear of the foot. The negative prompt is under the prompt only while it runs and first in the sheet otherwise, moved as the sheet closes or the status changes, never while the sheet is open or the box has the focus. Under the prompt the live negative is a 48 px fold row, shut, with its first words ("none" while empty); a press opens it to its note and box, and a write of words into it (Reuse, Load settings from PNG, a setup, the tag helper, a snippet, a finding) or Check's selection opens it first, through `replaceBoxText`; on SDXL at 393x852 More options and the results' top are on the first screen again. Batch and Repeat are rows of −, the box and + (44 px, held with their reason at their bounds; 400 px at most on its side; the boxes named by `aria-label` while the buttons are in), Seed's dice a 44 px square after its box. A refusal that focuses a box in the closed sheet opens it, its words in the sheet's foot. The tier's scroll restore now waits for the page to grow to its phone height (a shorter Create view showed the old clamp). Skipped: Sampling's Advanced split for Mu and True CFG (the four boxes fit a 361 px row, and two to a row under 360); a Steps stepper (optional; it shares Sampling's row); Load settings from PNG… in the topbar ⋯, which waits for P12 (it stays in the panel under the results). Still open: on a tag checkpoint a typed prompt's tag chips (about 200 px for seven tags) push More options under the bars again, and at 360x780 the size's "1.0 MP" line (P5) and the 5-row prompt box leave More options under the bars on SDXL.
- Phase 3, the compose controls: P5, P10, P12 and P9's sheet (`phone-compose.js`, the phone's second script of its own). The topbar is 52 px: the model chip (`#btn-models` with `#status` moved into it - dot, name and state, a chevron; the word Models visually hidden) and a 44 px ⋯ whose docked menu holds Load or Unload, Saved setups…, Personal snippets…, Load settings from PNG… and Reset (last, `--danger`), each pressing the page's own button, a held press said above the bars; the word mark, Unload, Load, the free space and the job's mirror are not drawn (the action bar says the job). Examples, Check, Tags and Expand move from the caption into a 44 px toolbar under the box, 8 px apart; the suggestions still hang from the box's foot, over it. The size is a Size chip on the caption's line (megapixels and pixels; amber-edged while the size line warns) and a 50 px Shape row - Square, Portrait, Landscape, each with the ratio it takes (the last chosen in it, else the guide's first of 3:4, 2:3, 7:9, 9:16, 4:7, 5:12) - pressing the strips' own buttons, which stay undrawn; the chosen Portrait or Landscape opens its ratios, the chip the four sizes and Custom dimensions…, as the docked menu's rows with the one in force ticked. The size line is drawn only while it warns. Check opens as a sheet (`#check-sheet`, the check's own head its title row); shut by its ✕, Escape or Back, the check is kept and a "Check: N findings ›" line under the toolbar shows it again; a finding's Show leaves the sheet for the box. The refusal line over the bars is now `position: fixed` (sticky, it stayed at its place now the panel ends short of the bars). Measured at 393x852: More options' foot 542 px down (625 before), 607 on SDXL (689); at 360x780 it is above the bars on FLUX.1 and on SDXL too (607 of 658 px, its negative's row shut), which the More options step left open. Left of P9 (phase 6): the findings as cards with a trailing 44 px Show, the sections as 48 px folds. Skipped: Unload in the Models sheet's footer (P12 puts it there; this slice's brief and the sketch's ⋯ hold it, and the Models sheet's footer is P43/phase 6's); Batch on the Size chip's row ("Images [− 1 +]" in the sketch: at 360 px the chip, the caption and the stepper do not fit one row).
- Phase 4, slice A (2026-11-06): P16, P25 and P26, the held buttons' look and the drawn ticks (`phone-gallery.js`, `styles/phone-gallery.css`). The gallery head is one sticky row - the search (its placeholder the count, "Search 9 images"), Filters, Select and ⋯ - the first card at 129 px (360x780 too; 123 on its side) where it was 450. Filters is a sheet from the foot holding Starred and All models as 44 px chips, Runs, Sheet and the search's terms as a fold; ⋯ a sheet holding the picks' count, Clear and Picked (its rows inside the sheet), Re-render, Swap tags…, Refine…, More seeds…, Saved searches…, Bundles…, Recycle bin, the bin's own buttons (first in the bin, Recycle bin at their head) and Delete unstarred last - each the head's own node, moved, held ones listed with their reason. What the four bulk actions take is said once over them ("On the 3 selected images", or why none can), each row a line of what it does or a reason of its own; a held press says why in the sheet's foot. A held `.ghost` is the muted colour at full opacity on no fill (a held primary solid, as Generate's); every checkbox and radio on the phone is drawn (`appearance: none`). Skipped: the chip row of section 2.4 (the head is one row with a Filters sheet instead, as this slice's brief asks); the head's card view and Crop (P19); the run rows sheet (P20); the bin as its own screen (P22: its title row is drawn, its gallery-only rows stay listed and held). Left for slice B: the selection bar (P18) - the picks' count, Clear, Picked, Swap tags…, Refine… and More seeds… are in ⋯ until then.
- Phase 4, slice B (2026-11-06): P18 and P19 (`phone-select.js`; the cards and the bar in `styles/phone-gallery.css`). The cards are three across upright and six on the side, 4 px apart and each filling its cell (two and four at Large); Card size (Small, Large; `auradream.gallery.phone.v1`) and `#card-view`'s own Fit and Crop pair, moved in, are in ⋯, Fit the default. While Select is on in the Gallery view a selection bar is in the tab bar's place (117 px; one 61 px row on the side): Done (the picks kept), the count, Clear and Select all over Star or Unstar and Bin (the Picked items, pressed), Refine…, More seeds… and Swap tags… (the footer's buttons, moved in from ⋯) and More (⋯, for Compare, Unstar, Re-render at a tier and Seeds to grid); in the bin Restore picked and Delete picked permanently. Held ones stay drawn and say why over the bar (the gallery's line moved there). Back leaves Select (one history entry). A card's ring is 28 px at its foot's right corner (a bin card's at the top), the stars of unstarred cards not drawn in Select and none taking a press. A long press's menu gains Select after Open (the tablet's too). Skipped: Download of the picks (there is none: Bundles… backs up picked images), Compare as a bar button (no room for seven 44 px columns at 360 px; it is in More's Picked rows), drag-select (optional), and a Select row in ⋯ (Select is the head's own button beside ⋯ since slice A).
- Phase 4, slice C (2026-11-06): P20, P21's rest and P22 (`phone-runs.js`; the rows, chip, foot and bin screen in `styles/phone-gallery.css`). Filters' Runs is a 56 px row opening a sheet of run rows - the run's first image (`/api/runs` now answers each run's `cover`, its oldest image outside the bin), when it ran and its count over what it varied and its prompt - built from the picker's own options, which stay what a pick sets (the select kept in the sheet, not drawn; the `run:` search unchanged). The run shown is a chip in the search's place ("Run 3:48 am", the rest where it has the room), its press the runs' sheet, its ✕ the gallery as before the run. A run's sheet has Columns and Rows side by side as 44 px selects and its run actions in a 61 px foot over the tab bar - Finish run and More seeds…, moved in, and a ⋯ pressing its swipes, Fill width and exports - its zoom buttons not drawn. The recycle bin is a screen of its own: a ‹ before its name, Back leaving it (one history entry, the tablet's too), its cards no tools but their age ("today", "3 days ago"), the sheets only its own rows (no Starred, Runs, Sheet, Picked, bulk actions, Saved searches…, Bundles… or Recycle bin row); Restore picked and Delete picked permanently (red, confirmed) in Select's bar. Skipped: P21's axis chip "Steps × CFG ▾" (two selects side by side fit one row and keep the sheet's own pickers), its sticky axis labels, and pinch or double-tap zoom on the sheet (its zoom buttons are not drawn; a cell opens the lightbox, which pinches); the 3-thumbnail strip (one 44 px picture: three did not leave the words room at 360 px); Empty bin in the bin's title row (it stays in ⋯ beside which images it takes).
- Phase 4, slice D (2026-11-06): P24, P28 and P27's empty state (`phone-gallery.js`, `touch.js`; `styles/phone-gallery.css`). A press of the tab already chosen takes its view to its top (`phoneViewToTop`; smooth unless reduced motion, the focus kept on the tab, no history entry). Back to top showed nowhere on the phone (it waited for the head to be a window above the view, and the head is sticky there): it is offered three screens down and takes the page to 0, a 44 px round ↑ at the screen's foot on the left over the bars, where no card's star ever is (the star is a card's top right; a card is 93 px across at 320 px). While the page scrolls a 32 px pill hangs under the head across its middle with the top card's day and its place, "Yesterday · 52 of 160" (the bin's place alone), gone 1.5 s after the scroll stops; `aria-hidden`, a press goes through it. The empty gallery's first visit is a screen: No images yet, Create an image the one solid button (the Create view, the prompt focused), Start from an example, and Show all models while images wait on other checkpoints. Skipped: P28's sticky day lines (in one grid every stuck line stays stuck to its end, so they pile at one place - a longer one's tail past a shorter one's, or an opaque band over the cards' stars; the place pill says the day instead), its date fast-scroll thumb (optional) and its 44 px Load more retry (already 44 px across the gallery). Left: at 360 px a three-digit count cuts the search's placeholder ("Search 160 image").
- Phase 4 at a glance (the four slices above have the detail). Done: P16, P19, P20, P22, P24, P25, P26 and P27's empty state. Partly: P18 (no drag-select, which is optional; Compare in the bar's More, not a bar button; no Download of the picks, which the app has no action for), P21 (its hint, Columns and Rows on one row, the run actions in a foot with a ⋯; no axis chip, sticky axis labels or pinch and double-tap zoom on the sheet), P28 (the place pill; no sticky day lines or date thumb; Load more was already 44 px). Skipped: section 2.4's chip row (a Filters sheet instead), the run rows' 3-thumbnail strip, Empty bin in the bin's title row. Seen in the final 393/360 tour, left open: the place pill hangs over the first day line just under it ("Today · 1 of 16" across "Yesterday · …"); the bin's day lines are the images' days over cards labelled with their binned age ("Yesterday" over "today"); a run of no images (all cancelled) is a Runs row with a blank picture, and its sheet's empty line says "the filters above the sheet"; two ⋯ one over the other (topbar and gallery head); the newest group of a day has no day line of its own; and on Create, after a job cancelled from elsewhere, an empty viewer frame about 370 px tall with the action bar's picture back to its placeholder (phase 3's, seen here).
- Phase 5 (2026-11-06, `phone/phase-5-viewer`): P29's full bar, P30, P31, P32, P33, P34, P35 and P39 (`phone-viewer.js`, `styles/phone-viewer.css`). The lightbox on a phone is the viewer of section 2.5: the stage the whole black screen (the picture fitted to all of it), a top bar of ‹ (the cross, moved in; 44 px), the place in the walk across its middle and Share (`navigator.share` with the PNG in a secure context, else Download), and a foot of five 44 px columns - Star, Reuse and Refine moved in, Edit and More - with the lightbox's line over it; Restore takes Edit's place in the bin. Edit is an action sheet of Inpaint…, Outpaint… and Edit by instruction…; More the image's menu less those, with Download, the bin last in red. Fit, 1:1, the arrows, the keys and the action column are not drawn; the zoom shows as a chip while it changes. The facts are a peek (the prompt's first line and the seed, "Details") that a press or a swipe up raises as a sheet with its own scroll and history entry, each fact a 48 px row with a 44 px ⌕ at its end. A tap on the picture lowers the facts or hides and shows the bars; it never closes (Back, ‹ and the swipe down do). The first walk nudges the picture once unless motion is reduced. On its side the bar is a 72 px rail at the right and the raised facts a side sheet. `LB_STACKED_MEDIA` leaves the phone out either way up, so `lbNarrow` and the facts' letterbox no longer apply there. Every docked menu (P39) is an action sheet: a scrim under it whose press shuts it alone (a second ⋯ press among them), its name at its head, no line under the name, a Cancel row last. `smoke-phone-gallery` holds the viewer (portrait and on its side), `smoke-phone`, `smoke-phone-dialogs` and `smoke-phone-create` the sheet's head and Cancel. Skipped or partly: P32's tap-a-row menu (each fact keeps its own ⌕ and links at 44 px instead, no Copy per row: More has Copy prompt and seed); P34's filmstrip (optional); the result dock opening the viewer on the live preview while a job runs (section 2.5's last line); P35's facts as a side sheet only once raised, the peek staying at the foot. Share's fetch can outlive iOS's press, and is then a download: the device check.
- Phase 6 (2026-11-06, `phone/phase-6-sheets`): P43 step 2, P44, P45, P48, P49, P51, P53 and P9's cards (`phone-sheets.js`, `styles/phone-sheets.css`). Every sheet opens with the focus on its head's ✕ rather than a text box (an `autofocus`, the opener's `.focus()`, or one within two seconds of the opening with no press since), so the keyboard does not come up over it; a refusal that opens More options on a box keeps that box. Swap tags, Then refine, Mass refine and More seeds gained a ✕ in their head. `--kb-inset` (the keyboard's height, from `visualViewport`) lifts the sticky action rows and the delete question above the keyboard. The delete question stacks Delete over Cancel, 48 px each. Swap tags' rows are cards in the sheet's own scroll, its explanation, Mass refine's and Inpaint's and Outpaint's strength notes in shut "How it works" folds, and the three result lists as tall as their rows (none drawn while empty). The check's findings and critic issues are cards with Show at their end and 44 px source chips. Saved setups', Saved searches' and Prompt snippets' lists are 56 px rows drawn from their hidden selects. The Models sheet is one scroller, each checkpoint a card (the chosen one's notes in words), with a sticky foot of its note, Apply & save, Switch & load and Load or Unload (P12's Unload). `smoke-phone-dialogs` holds them. Skipped or partly: P48's edit screen pushed inside the sheet and a row's ⋯ (a row chooses, as a click in the select does; Restore, Apply and Insert stay the foot's); P45's chips in place of the facts' notes and `inert` on the page behind (light dismiss already closes it); P42's 2x2 steppers and preset chip row (the phase-1 frame stands); P40 step 3 (pinch on the canvas, optional); P9's sections as folds; P53's Edit and Refine notes (two lines each); P44's keyboard on a real iPhone (the device check).
- Phase 7 (2026-11-06, `phone/phase-7-landscape`): P65 and the rest of P63. On its side the portrait column is centred rather than stretched: Create's panel at most 600 px across the middle (the action bar's row was 560 already), every sheet's words a centred 640 px column under a title row and an action row across the screen, the delete question 560 px at the foot's middle, the Models sheet in the same column. Found on the way: a phone on its side over 900 px wide (a Pro Max's 932) had the PC's two columns - Create and the Gallery each a third of the screen - and the short window's 37 px fields; style.css's one-column query, fields.css's and generation.css's one-column rules and `TWO_COLUMNS_MEDIA` now take in `(pointer: coarse) and (max-height: 500px)`. The Models sheet's cards keep their narrow areas on its side (they were 500 px tall) and its pictures are 56 px. `smoke-phone-landscape` (7957, beside `smoke-tablet`) holds it at 852x393 and 932x430. P63 needed nothing more (the opaque bar, the 56x28 pill, the 24 px glyphs and the job in the tab's name are phase 2's). P35 was phase 5's. Skipped: P65's two-pane Create (later, in the review's words; the centred column fits the panel, and a second pane would need the results beside the settings with no inner scroller); P67's standalone mode, by the review's own advice (the manifest's `display: "browser"` stays, downloads keep Safari round them).
- Left: the device check (a real iPhone for the keyboard inset, Share's timing and P36; Android for Back) and the partly done findings above.

Notes:
- Order within phase 3: sticky Generate (CSS) first, which removes most of the reach problem on its own. Then the result card moved up, then More options with the size control, then the Queue and Check sheets. Re-measure after each step.
- P54 must land with P17, otherwise the six 44 px card tools overflow a 180 px card.
- Device check before closing phase 6: a real iPhone for P44 (keyboard over sheets), P1's keyboard behaviour and P36's ghosting; Android for P64's Back.
- Rough total: 8-10 weeks of focused work. Phase 1 alone fixes every control that is out of reach today.

---

## 5. Refuted or narrowed claims (what was checked and dropped)

The verification pass refuted no finding outright. These claims were corrected or dropped:

- CSS `order` or `display: contents` on `.panel` to reorder Create: dropped. Visual order would no longer match focus order (WCAG 1.3.2/2.4.3), WebKit drops the region's landmark, and the panel loses its padding. Nodes are moved instead.
- "CSS overrides cannot give the phone a hierarchy": overstated. `.panel` is already a flex column, so sticky Generate needs no DOM move.
- `@layer phone` for the phone rules: wrong. A layered rule loses to unlayered desktop CSS.
- Long press repurposed to select: dropped. Long press is the only way to Use seed, Copy and Open PNG on touch. It keeps the menu, which gains a Select item.
- Lightbox Delete "unreachable" and "the row did not scroll": wrong. `gallery-24` shows it reachable by a sideways scroll with no cue. It is a discoverability problem, high rather than critical.
- Card ⌦ as critical data loss: lowered. It bins, which is recoverable.
- "Inpaint, Outpaint and Edit primary buttons cannot be reached at all": overstated. About 18 px is past the screen and they are still pressable. Only 3-4 of 17 dialogs fail.
- Lightbox long-press menu "first row scrolled away": wrong. It is an orphan separator where Open is hidden. The menu starts inside the screen.
- Empty-bin confirm off-centre: an emulator scrollbar artefact, not a CSS bug.
- "No accent-color on checkboxes": false (style.css:113).
- Models picker "must press Models again to close": wrong. It light-dismisses on any outside press.
- Topbar costing 117 px: true only in standalone mode, which the app does not use. The real cost is 58 px.
- Finish-note buttons touching at 0 px: wrong. They are 33.5 px tall with 8 px gaps.
- Batch 4 drawn as 20vh `.many` tiles: wrong. Batch 4 is `.multi`; `.many` starts above 4.
- Bulk buttons queueing large GPU work in one tap: wrong. Each opens a preview dialog that states the scope and count.
- iOS `type=number` lacking a decimal point: wrong. `inputmode="decimal"` is a nicer keypad, not a missing capability. `#search` is already `type=search`.
- Hiding held buttons on the phone: dropped. The aria-disabled-with-reason rule applies; they move to the selection bar instead.
- `cover` crop as the gallery default: dropped. Fit stays (it shows lettering and orientation), with Crop one tap away.
- A separate Grid search sheet, and hiding Step cache: dropped. Both go into More options, and Step cache stays inspectable.
- Hiding `#last-run`'s buttons outright: narrowed to "only while the viewer shows that run".
- Hiding `.phone-tabs` under open dialogs: unnecessary once the lightbox is opaque.
- Hide-on-scroll headers in portrait: dropped. They move controls the user is not touching.
- A navigation rail in landscape: dropped for now as a large change for little gain.
- `transform: scale()` press feedback: dropped in favour of background-only state layers.
- Standalone PWA on plain-http LAN: blocked, since `navigator.share` needs a secure context. "Two stacked bottom bars" is overstated because Safari's toolbar collapses on scroll.
- Pull-to-refresh "scroll chaining from the Models list": already contained (model-picker.css:25).
- Optional extras kept as later work, not fixes: pinch-to-change columns, drag-select, a date scrubber, a search-token sheet, canvas pinch-zoom, swipe-to-dismiss sheets, a lightbox filmstrip.
- A third Results tab: not chosen. The action bar with a result dock covers the need with less chrome (see 2.2). It stays possible later.
- Severity adjustments applied: Help badges high → medium; Check inline high → medium; the context menu high → medium; Models sheet (platform) high → medium, kept high here as a merged core task; the lightbox backdrop to low; the stuck-label and Delete unstarred findings to medium.
- Physical-device acceptance remains for P44 and P36. The [initial Safari Simulator check](#6-safari-simulator-check-2026-11-07) recorded P44's Refine keyboard overlap; the [Refine correction](#7-refine-keyboard-correction-2026-11-07) now passes its focused native Safari checks.

## 6. Safari Simulator check (2026-11-07)

The [connection and rerun notes](ios-simulator.md) describe the prepared
machines, tunnel, fixture, retained harness and native-input pitfalls.

Tested application revision `4e1127c194871acf437637679e39cf349c2756cf` through
DeckLab's prepared MacBook SSH/Appium setup: Xcode 27.2 (27B52), iOS 27.0
(24A434), Appium 3.8.0 and XCUITest 12.22.7. The Windows fake server served the
actual checkout over a reverse SSH loopback tunnel, with a fresh synthetic
gallery. No weights, GPU work or household gallery were used. Documentation
changed during the iPad rerun; application assets remained unchanged.

**Result: 35 passed, 3 failed, 2 not applicable.** Each failed scenario is
Refine's software-keyboard check. The two not-applicable checks are the
phone-only image Share control on iPad.

| Simulator | Orientation | Initial viewport (CSS px) | Passed | Failed | N/A |
|---|---|---|---|---|---|
| iPhone 15 Pro | Portrait | 393 x 695 | 9 | 1 | 0 |
| iPhone 15 Pro | Landscape | 852 x 329 | 10 | 0 | 0 |
| iPad Air 11-inch (M2) | Portrait | 820 x 1048 | 8 | 1 | 1 |
| iPad Air 11-inch (M2) | Landscape | 1180 x 688 | 8 | 1 | 1 |

The [JSON evidence](review-2026-11-05-phone/simulator-2026-11-07.json)
records every scenario, source and harness hashes, raw report receipts, the
native control/keyboard rectangles and cleanup verification. The cases cover
Create, prompt entry, Generate/Queue, Models, More options (its absence on
iPad), gallery selection, viewer/swipe/Details/Back, viewer actions, Refine's
keyboard and phone Share.

Native taps and software-key presses exercised composition and Generate.
Native touch swipes advanced the viewer exactly one image. The phone's secure
loopback Share opened the native PNG activity UI. History Back closed the
tested overlay layers; this used `history.back()`, not the iOS edge gesture.
The phone's Refine dialog opened on its close button without raising the
keyboard, but explicitly focusing Strength exposed the remaining P44 defect.

**At this tested revision, P44 remained open on iPhone 15 Pro in portrait and on iPad Air 11-inch (M2).**
After generating an image, open it in the viewer, choose Refine, scroll
Strength into view and tap it. Cancel and Refine can sit behind the software
keyboard. On the iPhone, the native buttons
span y=565..615, while the keyboard starts at y=567: 48 px of each button is
covered. The [native screenshot](review-2026-11-05-phone/simulator-portrait-refine-keyboard.png)
and retained accessibility XML confirm the overlap.

On iPad in portrait, Cancel spans y=918..969 while the keyboard starts at
y=898, leaving the entire button underneath it. The [iPad screenshot](review-2026-11-05-phone/simulator-ipad-portrait-refine-keyboard.png)
shows the Refine footer hidden below the keyboard too. The current keyboard
inset adjustment is phone-only, so tablet dialogs need their own validation
when P44 is corrected.

On iPad in landscape, Cancel and Refine span y=809..860 and are marked
`visible=false` by native accessibility; the keyboard starts at y=453.
The [landscape screenshot](review-2026-11-05-phone/simulator-ipad-landscape-refine-keyboard.png)
confirms the hidden footer. The automated case failed while finding the
native button; the screenshot and XML distinguish this product problem from
a selector-only failure. iPhone landscape passed the native footer check.

On iPhone portrait, the observed values were `innerHeight=564`, `visualViewport.height=391`,
`offsetTop=131` and `--kb-inset=0px`. `syncKeyboardInset`
(`web/phone-sheets.js`) calculates 42 px and discards it under its 80 px
threshold; `phone-sheets.css` consequently applies no footer lift. Safari's
resize/pan interaction is the likely mechanism. Changing that calculation
still needs implementation and validation across both orientations. A zero
inset alone is not a defect: the pilot also captured a state where Safari
resized the viewport and the footer was visible without an inset.

Accepted rows come from the completed phone configurations of
`results-20261107T003614Z` and the corrected iPad run
`results-20261107T004658Z`. The original iPad attempts are excluded: Safari
displayed Start Page while WebDriver inspected a background AuraDream tab.
The corrected harness opens each URL through native Safari, requires that
exact document to be visible, and uses measured native taps on iPad. The
earlier pilot's nonzero-inset and Share-selector assumptions were corrected
before the accepted phone run. Raw reports, command logs, harness snapshots,
screenshots and native XML are retained under `work/safari-20261107/` locally.

Independent cleanup verified that all eight test-created Simulators across
the three runs were removed, all three owned Appium ports were closed, and
no runner remained. The Windows fake server and its port 17961 were stopped.

This is Simulator evidence. Physical iPhone acceptance, Android Back, native
iOS edge-swipe Back, iPad Split View, long press/pinch, other edit dialogs,
completed sharing delivery and GPU/model acceptance remain separate. No
product code was changed by that initial testing run.

## 7. Refine keyboard correction (2026-11-07)

**Refine passes on iPhone 15 Pro and iPad Air 11-inch (M2), both orientations.**
Tested application revision `68695a25d422fb343d1c064d5a32c200b189cdb1`, with
unchanged committed application assets throughout `results-20261107T020921Z`.
The final retained harness is `2026-11-07.10-native-refine`. This is a focused
Refine run: four passed configurations, eight passed field scenarios, no failures.
It does not repeat the ten general flows from section 6.

| Simulator | Orientation | Initial viewport (CSS px) | Smallest keyboard viewport height | Strength / Seed | Evidence |
|---|---|---|---|---|---|
| iPhone 15 Pro | Portrait | 393 x 695 | 391 | Passed / passed | [Native screenshot](review-2026-11-05-phone/simulator-p44-iphone-portrait.png) |
| iPhone 15 Pro | Landscape | 852 x 283 | 47.9844 | Passed / passed | [Native screenshot](review-2026-11-05-phone/simulator-p44-iphone-landscape.png) |
| iPad Air 11-inch (M2) | Portrait | 820 x 1048 | 711 | Passed / passed | [Native screenshot](review-2026-11-05-phone/simulator-p44-ipad-portrait.png) |
| iPad Air 11-inch (M2) | Landscape | 1180 x 688 | 266 | Passed / passed | [Native screenshot](review-2026-11-05-phone/simulator-p44-ipad-landscape.png) |

Each field must be fully visible inside the native window and above the real
software keyboard, with five interior DOM hit tests clear of the sticky rows.
A native keyboard digit changes its value. The actual native Cancel then closes
Refine alone, dismisses the keyboard and restores the visible viewport. Both
native action rectangles must be clear of the keyboard in consecutive stable
samples. Screenshots were inspected as well as these geometry checks.

Phone form sheets now use the visible viewport's top and height directly.
Tablet Refine keeps its width and top gap, with a visible-height cap and sticky
actions. A touch Refine field covered by the header or actions is revealed by
scrolling only the dialog, including after a late preview or restored-draft line.
Already clear fields stay in place; an oversized textarea keeps Safari's caret
pan. The small landscape state scrolls the header away. On a wide phone its
44 px Cancel and Refine buttons sit beside the field: Safari can leave only
48 px, too little for a full-width input and footer stacked vertically.

The [accepted JSON receipt](review-2026-11-05-phone/simulator-p44-2026-11-07.json)
records the source trees, harness/report hashes, field values before and after
native typing, native and DOM rectangles, hit tests, screenshots and cleanup.
The raw report SHA256 is
`e70e398f0abe22e1ad6ec36d2c1a1ac42932ac41205cd597a92c6d1cf50bf4db`.
Full reports, command logs, XML, pictures and frozen harness snapshots remain in
`work/safari-p44-20261107/` and the corresponding Mac directory in the
[rerun notes](ios-simulator.md#repeat-the-p44-focused-follow-up).

The [five development attempts](review-2026-11-05-phone/simulator-p44-pilots-2026-11-07.json)
remain separate, including failures and interrupted coverage. Stronger input
checks exposed late field displacement in portrait and the 48 px landscape
case. Appium's web-tap calibration also stalled on Generate and on reopening
Refine; the retained harness waits for keyboard dismissal before Generate and
opens Refine through its unique visible native button, with observed bounds.
The interrupted run on the final application source passed landscape Strength
but did not finish Seed; it is not counted as a passed configuration.

Automated checks on the resulting application source: 36 focused classic-script
and documentation-map tests, and seven browser smokes (`smoke-phone-dialogs`,
`smoke-phone-gallery`, `smoke-phone-landscape`, `smoke-tablet`, `smoke-phone`,
`smoke-phone-create`, `smoke-refine`). The dialog smoke exercises supplied 391 px
and 94/48 px viewport heights, a late 28 px content line, focus changes, page
pinch reset and restoration. These mechanism checks complement native Safari.

Independent cleanup found all 16 Simulators created across the six runs absent,
all six Appium ports closed, no owned runner/Appium process and no reverse
fixture tunnel. Windows fixture ports 17961 and 7960 are closed; the synthetic
scratch galleries and raw evidence are retained. No weights, GPU work or real
gallery were used.

Remaining simulator work includes the other edit dialogs' keyboards and iPad
Split View. Physical iPhone acceptance, Android and native edge-swipe Back,
completed sharing delivery, long press/pinch and GPU/model acceptance remain
separate. `UI#40` stays open for those checks and the review's partial findings.

## 8. Residual phone corrections (2026-11-07)

Four defects from phase 4's final tour are corrected:

- The newest Today group has a day heading on the phone. Same-day prepends
  reuse it; different-day prepends keep the old heading as its group's
  boundary, including a page left open across midnight and historical
  images arriving with their own heading. A mouse keeps the gallery's
  original first-group presentation.
- The floating place pill hides immediately while a day heading crosses
  its box, so the two dates do not overlap, and returns after it passes.
- Bin headings explicitly say "Created" before the creation day, separate
  from each card's binned age. Grouping, order and date searches still use
  creation dates.
- Cancelling after the first preview, before an image finishes, leaves no
  empty result frame on Create. The outcome remains above the action bar,
  and its picture still opens the newest earlier result from the session.
  Single-image, two-image and six-image cancellations cover the ordinary,
  fitted and long-strip layouts.

The regressions extend `smoke-phone-create` and `smoke-phone-gallery` on
isolated fake galleries. The cancellation assertion failed against the old
CSS before the fix. Retained screenshots: [cancelled Create](review-2026-11-05-phone/residual-cancelled-create.png),
[day-heading crossing](review-2026-11-05-phone/residual-day-crossing.png) and
[bin dates](review-2026-11-05-phone/residual-bin-dates.png).

Validation passed: 36 classic-script and documentation-map tests, Ruff, and
eight browser smokes (`smoke-phone-create`, `smoke-phone-gallery`,
`smoke-phone`, `smoke-phone-dialogs`, `smoke-phone-landscape`,
`smoke-tablet`, `smoke-gallery`, `smoke-bin`). The gallery smoke includes
same-day, midnight, historical and both mixed-day prepend orders through
the real `joinFinished`, plus the mouse roundtrip. These Chromium checks
use only synthetic galleries and do not extend the native Safari result.

The remaining tour defects and the implementable partial findings are
addressed in section 9. The Refine-only Simulator result in section 7 does
not qualify other dialogs. No model or GPU acceptance is claimed.

## 9. Remaining implementation findings (2026-11-07)

- Empty runs use an intentional ∅ cover marker beside their image count.
  Their sheet says that the run has no images; a filter miss asks for a
  change of search or filters without referring to controls above a sheet.
  Finish run and More seeds retain their existing availability.
- Gallery actions uses a list icon, distinct from the app topbar's ⋯.
  The 44 px button keeps its actions and says Gallery actions or Bin
  actions, including the selected count where applicable.
- The phone search uses a compact count at narrow widths:
  `Search 160` is 77.27 px wide, fitting the 126 px available at
  360 px and 86 px at 320 px. Its accessible name remains Search gallery;
  empty and bin wording are unchanged. The gallery smoke checks both sizes.
- Check's sections are 48 px disclosures (P9), Findings open initially and
  the other sections closed. Choices survive redraws and tier changes.
  Content nodes, hidden states and the critic's existing selectors stay in
  place; the mouse gets its plain headings back.
- The result dock opens live frames across the phone (P30), without
  scrolling Create. The preview follows only the job opened by the press.
  Its finished image replaces the frame, with an explicit Open finished
  image handoff to the normal viewer. A job ending without a saved image
  keeps the last frame labelled Preview ended; a partly completed batch
  keeps its latest finished image. Back, Escape and close restore dock focus, with
  one history entry through the handoff. Frames never become records.
  A finished image removed elsewhere clears its pixels and Open action,
  showing its bin/deleted status without falling back to an older image.
  Restore recovers that same cached image. Leaving the phone closes the
  preview and gives focus to the visible Generate button.
- The keyboard viewport, sticky-action and focused-field protections now
  cover Inpaint, Outpaint and Edit by instruction as well as Refine (P44).
  Tablet widths and top gaps stay intact, and the short phone landscape
  actions fit beside the field. Inpaint's fixed-top canvas rule is retained.
- More options and the library forms had the same tiny-keyboard problem:
  their 61 px sticky heads and multi-action footers covered a 48 px viewport.
  On the phone those rows now scroll normally in the short state; dismissing
  the keyboard restores their usual sticky positions. A separate reveal
  keeps the focused field clear after late content without changing drafts,
  holds, nodes or page scroll. Count confirmations use their rendered
  bottom-anchored box and a full 48 px cap, preserving count validation and
  Cancel. The four image-pass dialogs retain their existing calculation.

The earlier phases' explicitly skipped alternatives and optional extras
remain skipped: the list manager edit sub-screen, fact-row menus, different
Outpaint controls, sheet axis chips/zoom, switches, drag-select, the date
scrubber and standalone mode. Refine and Edit retain their brief two-line
context notes (P53); landscape facts keep their peek at the foot until raised
as a side sheet (P35). These retained presentations are the chosen alternatives,
not remaining implementation defects.

Validation uses synthetic galleries with no weights or GPU work. The new
`smoke-phone-keyboard` supplies viewport dimensions for 88 field/late-content
states: 70 image-pass cases across phone portrait, 48 px landscape, both
tablet orientations and a 678 px Split View emulation; 16 other phone-form
cases at 391/48 px; and two count-confirmation cases at 48 px. Baseline
checks reproduced the sticky-row and count-field clipping before their fixes.
That emulation does not qualify actual iPad
multitasking or a software keyboard. The existing Create, dialogs and Gallery
smokes exercise the new preview, disclosures, cancelled zero-image run and
menu semantics. All twelve relevant browser smokes passed, alongside 36
classic-script/documentation-map tests and Ruff. Native follow-up is a
separate acceptance step.

CI follow-up (2026-11-07): the new empty-run marker, ∅ (U+2205), was
missing from the smoke fonts' measured characters. A Windows Edge 156 run
of `tools/font-widths.json` measured its advance at 627.93 px for weight
400 and 647.95 px for weights 600/700, at 1000 px and DPR 1. The character
and all three widths are now retained in the measurement fixture and
`tools/segoe-widths.json`; every earlier width matched the existing table.
The 15 smoke-font tests, 36 classic-script/documentation-map tests and Ruff
passed after the correction. CI also caught the desktop contact-sheet
smoke's old empty-filter wording. Its assertion now checks the complete
current message, keeping the empty-grid and height checks; the corrected
contact-sheet smoke passed locally.

Still requiring device acceptance: physical iPhone keyboard behaviour,
Share's timing and completed delivery, Android/system and native iOS edge
Back, long press/pinch, and actual iPad Split View. `UI#40` remains indexed
for that acceptance; the earlier skipped design choices do not block it.

## 10. Edit-dialog keyboard acceptance (2026-11-07)

**23 native field checks passed across four Simulator configurations.**
These cover Inpaint, Outpaint and Edit by instruction; section 7 retains
Refine's separate acceptance. The phone landscape Instruction textarea is
larger than the available keyboard viewport, so its caret behaviour remains
unqualified. It is not counted as a passed field.

| Simulator | Orientation | Passed fields | Accepted evidence |
|---|---|---|---|
| iPhone 15 Pro | Portrait | 6 | `results-20261107T033550Z` |
| iPhone 15 Pro | Landscape | 5 | `results-20261107T033550Z`; Instruction caret unqualified |
| iPad Air 11-inch (M2) | Portrait | 6 | Five numeric fields in `results-20261107T040316Z`; Instruction in `results-20261107T043039Z` |
| iPad Air 11-inch (M2) | Landscape | 6 | `results-20261107T043039Z` |

Each field has native typing that changes its value, five clear interior
DOM hit tests, two stable samples of action rectangles above the real
software keyboard, and an actual native Cancel that closes the dialog and
keyboard and restores the viewport. Dialogs open through their application
callback on a seeded synthetic row; opener gestures and edit submission are
outside this keyboard scope. The editor is held by the fake's checkpoint,
so this proves neither model loading nor GPU/edit output.

This is per-field acceptance across retained runs, not one all-green full
matrix run. Sixteen accepted fields were measured on `da14e75efc7f6039e2f446645790c9842275ebe8`;
the seven missing cases passed on `14f6b04c062411ddc42f66d2e06f981fea0d1220`.
Portrait Edit Seed has its own completed field receipt before the later
Instruction selector failure; that enclosing failed flow remains recorded.
That later change adds other phone-form/count protection while retaining
the image-pass arithmetic and CSS. Application assets stayed unchanged
through each run. The later gallery placeholder wording change leaves
those image-pass paths untouched. The [receipt](review-2026-11-05-phone/simulator-edit-keyboards-2026-11-07.json)
records each accepted field's source, harness and raw evidence hashes,
alongside every excluded setup/selector attempt and cleanup.

The attempts exposed harness defects: treating Inpaint's valid canvas focus
as keyboard autofocus; probing obsolete Safari contexts before their newer
replacement; using the shared initial navigation gate before AuraDream's
visible-document routine; and expecting sentence-case Instruction on the
tablet's uppercase native caption. The final harness filters the installed
driver's context URL metadata before probing JS, still requires the exact
visible AuraDream document, and matches the observed unique native label
without case sensitivity. After the retained phone acceptance, retries
selected only missing iPad cases. Earlier pilots are excluded, and the
setup/selector failures remain recorded; they are not silently combined
into passed configurations.

Retained native screenshots: [iPhone portrait](review-2026-11-05-phone/simulator-edit-iphone-portrait.png),
[iPhone landscape](review-2026-11-05-phone/simulator-edit-iphone-landscape.png),
[iPad portrait](review-2026-11-05-phone/simulator-edit-ipad-portrait.png) and
[iPad landscape](review-2026-11-05-phone/simulator-edit-ipad-landscape.png).
Full reports, command logs, XML and frozen harnesses are retained under
`work/safari-edit-keyboard-20261107/` and the corresponding Mac directory in
the [rerun notes](ios-simulator.md#other-image-pass-keyboard-checks).

Independent cleanup verified all ten created Simulators absent, all five
Appium ports closed, no owned runner/Appium/tunnel, and Windows fixture
port 17961 closed. Raw evidence and synthetic scratch galleries are retained.

`UI#40` is now `USER`. Remaining acceptance is:

- Physical iPhone/iPad keyboard use, including the other phone-form fallback,
  count confirmation and landscape Instruction caret behaviour.
- Actual iPad Split View/Slide Over transitions and their keyboards.
- Android/system Back and native iOS edge Back through stacked overlays.
- Physical viewer opacity, long press, pinch/double-tap and swipes.
- Share timing and completed file delivery, including the LAN Download fallback.

Other phone forms have the supplied-viewport evidence in section 9, not
native keyboard acceptance. No weights, GPU work or real gallery were used.

## 11. Check-sheet close focus order (2026-11-11)

PR #69's [first hosted run](https://github.com/matt439/AuraDream/actions/runs/42974741989)
failed the phone Create smoke's combined closed-sheet/retained-check/history/
Check-focus assertion after reopening the check from its chip and pressing
✕. Its automatic rerun passed alone, so the first failure remained a race
to repair. All CPU jobs and the other browser shards passed.

A controlled native-method probe reproduced the failed focus handoff:
`closeAnalysis` hid the answer, then tried to focus Check while its sheet
was still modal. Focus stayed on `analysis-close`; the hidden-state
MutationObserver closed the sheet only later, and the close handler's
body/no-focus repair did not cover that hidden descendant. A 250 ms bounded
tail around the actual Check focus call delayed observer delivery without
replacing native dialog closing or focusing. The original complete assertion
failed on the old source and passed after the change.

`closeAnalysis` now calls `syncPhoneCheck` immediately after hiding the
answer, before its existing Check-or-prompt focus handoff. No timer or new
lifecycle state is needed. Finding Show keeps its separate dismiss-and-
select path. The durable smoke explicitly focuses the chip before reopening
and records whether the real Check focus call happens while the sheet is
modal. It keeps the original full assertion and requires that recorded
modal state to be false, then restores the native focus method.

The strengthened Linux run then exposed a separate fixture race after the
fake's step delay became zero: the batch-layout sections polled for a
temporary `busy` state that could finish between polls. Those three native
Generate presses now capture the prior viewer ids and wait for submission
and rendering to finish with exactly 2, 4 or 8 fresh ids. A no-op with the
same count cannot pass. Their original layout/tools assertions and the
earlier timed live/queue/cancel coverage remain; no fake delay or polling
timeout was increased.

The separate `work/phone-batch-settle-probe/` control observed a native-tap
step-zero job already complete before polling, then proved that a same-count
no-op press fails the fresh-id condition. Actual Generate presses then
passed for 2, 4 and 8 new image ids.

The scratch probe, fixed trace and repeatable old-handler negative control
are retained in `work/phone-check-close-probe/`. Three fresh fake-browser
smokes (`smoke-phone-create`, `smoke-check-dismissal`, `smoke-phone-dialogs`)
passed together at two workers on Windows (57 s) and on Linux (55 s).
The Linux run used the CI versions, Chrome 156.0.8078.79 and Node 24.22.2,
with the private font configuration. The fixed close screenshot was inspected. The root
classic-script/module-map run passed 60 tests; the agent's classic-script,
browser-input and step-container run passed 132 tests. No GPU, model load,
real-gallery or physical-device acceptance is claimed.

After the every-focus-attempt recorder and fresh-batch-id synchronization
were final, the affected Linux phone Create smoke passed again (48 s browser,
54 s total). Its receipt is under `work/ci-pr69-linux-final/`.
