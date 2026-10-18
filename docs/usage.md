# Using Local Diffusion

For installation, launch options, sizes and environment variables, start with
the [README](../README.md). The [anime model guide](anime-model-guide.md) helps
choose a checkpoint; [verified.md](verified.md) records measured performance.

## Load and switch models

**Load** loads the selected checkpoint; **Generate** loads it automatically when
needed. First use may download weights or convert a checkpoint, taking several
minutes. **Unload** releases the image weights and prompt helpers.

The server starts without asking questions in the terminal. **Models** opens
automatically when the browser first connects to an idle server with no model
loaded, or a failed load. It stays closed if the server is already loading,
ready or busy; closing it keeps it closed for that page visit. Choose **Models**,
select a checkpoint, quantization and offload, and press **Switch & load** to
apply them without restarting the server or closing the page. Compact
facts show **Style**, **Refine** support, **Speed** and the upper recommended size in
megapixels. The size label distinguishes a recommendation from a documented or
native maximum, reference size or largest example. The tooltip explains its
source and limits; a published example is not a guaranteed maximum, and equal
MP does not guarantee equal quality for every aspect ratio. The model-page link opens the source card.
Finish or cancel the
image queue first, including paused jobs, and wait for any current load, prompt
check, critic or Expand request to finish. The old weights are released before
the selected checkpoint loads. While it loads, Check, tag lookup and adapters
work, and Generate queues behind the load. A failed load shows its reason and
**Retry load**; you can also select another checkpoint.

Prompts and saved images stay available. Sampling starts from that checkpoint's
defaults or its remembered settings, and the guide, supported controls, adapters
and gallery scope update. **All models** shows images from the other checkpoints.
Reopen an existing Refine dialog after a switch to use the new schedule. Finish run for an earlier submission requires switching back to its checkpoint.
Other open tabs follow the server's selection too.

Press the star beside a model to add or remove it from **Favourites** at the
top of the list. This is saved immediately in this browser and survives page
reloads and server restarts; it does not change or load the selected model.
The ordinary and Nunchaku versions of a checkpoint share its star.

**Made with** shows the newest images the chosen checkpoint has made in this
gallery and how many there are, so its style can be judged from its own
pictures: on a window 1280 px or wider in a column on the picker's left, at
the size of the gallery's small cards, and on a narrower one in a row under
the table. It follows the row you choose and works while nothing
is loaded. Click a picture (or Tab to the pictures, use the arrow keys and
press Space) to open it in the viewer; closing the viewer goes back to the
picker. **See all** closes the picker and shows every image made with that
checkpoint in the gallery (it ticks **All models**, clears **Starred** and
searches for `model:=<id>`). A checkpoint with no images says so instead.

**Quantization** trades memory for precision: FP8 and INT8 keep 8-bit weights
(the same size; INT8's output is not yet compared with FP8's on the card),
BF16 is full precision and the largest, and Nunchaku
4-bit builds are the smallest but run only on Linux or WSL. **Offload** trades
memory for speed: Model offload keeps one component on the GPU at a time,
Sequential moves smaller pieces and is much slower, and No offload is fastest
but needs 24 GB or more for FLUX-size models. FP8 with Model offload is
recommended for a 16 GB card; each choice's note says the same under its menu.

The server's checkpoint says
**Loaded**, **Loading…**, **Load failed** or **Current** under its name. A dotted
label has a note: hover it, or a screen reader reads it with the row. Only the
chosen row's star and model page are tab stops (Shift+Tab and Tab from its
radio), so Tab reaches the actions without passing every row. Opening the
picker focuses the chosen row and scrolls the table to it. If a quantization
or format has no build of the chosen model, no row is chosen; the model is
chosen again when a list has it. A field you changed keeps your value until
you apply it; the others follow changes saved in another tab. A switch made
in another tab drops your unsaved changes, and the line under the buttons
says which choice was dropped.

**Speed** uses recorded RTX 5080 samples around 1 MP (WAI's portrait is 1.38 MP):
**Fast** is up to 15 seconds, **Medium** over 15 through 45 seconds, and **Slow**
over 45 seconds. The sample time and area appear below the category. A
category appears only when the sample used your chosen quantization, offload
and platform. Otherwise the word is **Reference**, or **Unmeasured** for BF16
or sequential offload, and "timed with FP8" or "timed on 4-bit Linux build"
says how the sample differs; these read dimmer than a category. **First run, busy PC** marks the busy-CPU first
images, which are **Reference** too: a first image is not a warm timing. The tooltip gives steps, settings and date. **Unmeasured** models
without a sample show default steps instead of an invented timing. These are
previous runs, not estimates for your current prompt, size or steps; loading
time is separate.

Some checkpoints are gated on Hugging Face. FLUX.1 Krea [dev] and FLUX.1 [dev]
need their licence accepted on the model page; Krea 2 Turbo and Raw need the
authors' approval, which can take days. Then sign in once with
`uv run hf auth login`. The 4-bit lite-infer builds are not gated. Each
row says **Open**, **Accept licence** or **Needs approval** under the name.
Qwen-Image's row also says **Over 16 GB**: only its 4-bit Nunchaku build (WSL)
fits this card, and choosing it under FP8, INT8 or BF16 says so under the
buttons before you press one.
Until access is granted, loading a gated model fails and says why; a page
opened after that failure shows the picker with the reason. Choose an open
model in **Models**, or press **Retry load** once access is granted.

**Apply & save** applies changed settings without loading their weights: a
loaded model whose checkpoint or loading options change is unloaded at once,
and its button's tooltip says so. Saving only the startup preference leaves
an already loaded model in place. Both buttons remember the model and loading
options on the server for next time. Check **Load automatically on startup**
(beside the buttons) to load them when the server starts; leave it unchecked
to load only when you press Load, Switch & load or Generate. The checkbox
takes effect after pressing either save button.

The line under the buttons says what the server has loaded, what a press did
and what went wrong: a held button's reason when pressed, a failed load or
switch, and a load in progress. Changes not yet applied read **Unsaved
changes**, with the running settings and **Revert** to put them back; they
stay while the picker is closed. If the server was started with flags or
variables that beat the saved choice, the saved message names them.

The list follows the chosen quantization. Nunchaku offers its mapped checkpoints
and an nvfp4/int4 weight format on Linux/WSL. Explicit launch flags and environment
variables override remembered settings. Starting with a different model
(`--model` or `LOCAL_DIFFUSION_MODEL_ID`) ignores the whole saved choice,
since its quantization, offload and auto-load were picked for its own model.
`--defaults` ignores the saved file and `--menu` opens the optional terminal
picker, with the saved choice as its defaults; picking another model there
does not load automatically at startup either. If a switch cannot release the running model, that model stays
selected and loaded and the saved settings are left as they were. The preference file is
`local-diffusion.settings.json`, beside the gallery database.

## Generate and manage jobs

Enter a prompt, choose a size and press **Generate**. The checkpoint supplies
its starting Steps, CFG and Mu. The **?** badges explain each control and
which settings the model uses. A blank Seed draws a new seed; a fixed Seed
lets you compare settings. Reproduction also depends on the checkpoint,
quantizer, size and other inputs. Nunchaku is not bitwise reproducible.

- **Batch** makes several images together, using more VRAM.
- **Repeat** makes that many batches sequentially, with consecutive seeds.
  Batch 2 and Repeat 3 make six images, holding only one batch at a time.
  Use Repeat for many large images; the batch pixel limit does not guarantee
  that a particular checkpoint fits in VRAM.
- **Generate** and **Run grid** can queue work behind a running job. Queued
  jobs have individual cancel buttons. Jobs survive closing or reloading the
  browser, but a server restart ends them.
- **Cancel** stops after the current denoising step and keeps saved images.
- **Pause** waits for the current cell to finish. It needs a job with another
  cell to run; a single batch has no such boundary. Resume continues it.
  Pressed on a queued job, it holds that job before its first image once its
  turn comes, keeping its place in the queue.
  The model remains loaded, so Unload is unavailable while paused. A pause
  past a minute says how long it has lasted, in the tab's title and, in a
  tab whose job waits behind it, the note under the bars.

The browser remembers prompts and controls; sampling settings are remembered
per checkpoint. **Reset** restores defaults. Resume is never remembered as
ticked. **Reuse** on an image restores its settings; check the warning when
reusing one from another checkpoint. Prompt replacements support Ctrl+Z.

### Load settings from a PNG

Drop a PNG anywhere on the page, or press **Load settings from PNG…** under
**Generate**, to fill the panel from a file - a PNG from the
other gallery (Windows and WSL keep separate ones), from a backup, or from
another tool. It works like Reuse: the note offers **Undo**, Ctrl+Z gives
back the prompt boxes, and a PNG made on another checkpoint says so.

- A **Local Diffusion PNG** gives back every setting it was made with:
  prompt, negative prompt, size, steps, CFG, Mu, seed and adapter. Its
  reference images come along only if this gallery holds the very same
  pictures under the same numbers; a refine pass fills the panel as a
  normal render at the panel's own size.
- An **A1111, Forge or Civitai PNG** (its `parameters` text) gives the
  prompt, negative prompt and size. Its steps, sampler, CFG scale, seed and
  checkpoint do not carry over: another tool's sampler and checkpoint do not
  make the same picture from the same numbers, so the panel keeps its own,
  and the note lists them. Whether its negative prompt runs is judged at
  the panel's own CFG.
- A JPEG, a ComfyUI PNG (its workflow is not settings this app reads) or a
  PNG without settings is refused with a reason.

Only the PNG's text is read and sent to the server; the image itself is not
uploaded, not added to the gallery and not used as a reference. Dragging a
gallery image onto the panel does nothing: use its **Reuse** instead.

## Write and check prompts

**Examples** supplies the authors' prompts; templates select the first slot
to fill. **Tags**, where available, supplies the checkpoint's own vocabulary
and recommended negative prompt. Its comma chips can be reordered or removed.
On Anima Turbo, **Start with masterpiece, best quality, score_7, safe**
writes the card's recommended prefix ahead of the prompt: only the tags it
has not got, never `safe` beside `sensitive`, `nsfw` or `explicit`, and
Ctrl+Z takes it back. Anima Aesthetic offers no prefix or score tags, since
its card advises against `score_*` tags.
The negative prompt is inactive when the current guidance settings do not
use it; the caption explains how to enable it. On a checkpoint with a
**True CFG** box (Qwen-Image and the FLUX.1 family), while True CFG is
above 1, a negative box holding only spaces sends a single space, which
still runs True CFG - Qwen-Image's guide asks for one when there is nothing
to remove - and the image's facts list it as "a single space". An empty box
sends none, and elsewhere blank always means no negative prompt.

**Check** measures the prompt with the checkpoint's own tokenizers and checks
its documented guidance. It runs on the CPU without loading image weights,
including while images generate. The first use may download tokenizer files
(megabytes); after a 30-second wait, any unfinished download continues for
the next check. Encoder rows show token counts and where reading stops.
Order checks apply to supported comma-tag prompts; prose ordering advice
does not imply every sentence has been checked. On Anima a sentence may
stand anywhere among the tags, as its card allows, and is never moved by
**Use reordered**; an `@` artist goes in the Artist step. Recheck after
editing.

Tag suggestions and relevant checks query Danbooru's index with tag text;
on Anima, a prose prompt's comma clauses (never a whole sentence) are sent
too, since its order and artist checks read them.
Images and generation settings are not sent. Network failures leave tags
unchecked, and tag advice never blocks generation. Ordinary prompt checking
and the critic run locally. Prompt weighting syntax such as `(word:1.2)`
is literal text here, not a weighting control.

**Ask the critic** uses local Qwen3-4B through llama.cpp to suggest issues,
a rewrite and sometimes a reordering, citing the supplied author sources.
Validation removes unsupported citations and invalid reorders, but suggestions
can still misread a source or add unwanted details. Review the rewrite before
using it. **Use rewrite**, **Use reordered** and **Apply** replace the prompt;
Ctrl+Z restores it.

First use downloads about 0.6 GB of llama.cpp and 4.3 GB of Qwen3-4B Q8_0
into the Hugging Face cache, verified against pinned hashes. Later uses start
from disk; measured critiques take roughly 10–20 seconds and 5.5 GiB VRAM.
The critic is unavailable during an image load, queued/running jobs or Expand.
While it runs, generation submissions are held, including its initial
download and any prompt recheck. Its process stops after answering.
Supported platforms and custom binaries are described in
[Settings](../README.md#settings); `LOCAL_DIFFUSION_CRITIC=false` disables it.

**Compare on the same seeds** renders the selected original, reordered,
rewritten and **Your version** prompts on four or eight shared seeds.
Two versions on four seeds make eight images. Comparison **Batch** changes
how many run together; the main panel's Batch and Repeat do not apply.
Only a pure reordering isolates word order; a rewrite changes other things.
Four seeds are a first look, not strong evidence. The results open as a
contact sheet with one row per version, scoped to that submission.

On ERNIE, **Expand** rewrites a short prompt into detailed Chinese for review.
First use downloads the separate 7.7 GB rewriter and 312 MB translator.
The English translation is a reading aid; the Chinese text is the prompt
used for generation and saved with the image. Generation does not rewrite
silently. Edits made during Expand
are preserved, with its result offered separately. Expand also holds
generation submissions while it runs. **Stop** beside its running note
abandons the answer and preserves your prompt and Undo. A download or model
call already underway may finish; the server keeps its GPU gate until that
stage ends. Generate can then queue behind it, and another Expand has its
own ticket. Unload explains when an abandoned expansion is still finishing.

**Notify when done** below progress is off until you opt in. Tick it to
request browser notification permission. It notifies when this tab is hidden
and its own watched job ends, with distinct messages for failure, cancellation
and budget stops. Visible or replayed finishes do not alert again. The
browser preference is separate from generation settings; blocked or
unsupported notifications show a reason beside the checkbox. Desktop
permission and delivery depend on your browser and operating system.

## Use references and adapters

The Models table also shows the full image-model download reference and
whether all required files are **On disk**, **Part cached**, **Not cached**
or **Unknown**. This is an offline file check, not a download or integrity
test. The size is the full sourced cost, not the amount still missing;
cached files can reduce it. Unknown means the app cannot establish a
complete reference, including some gated models and custom checkpoints.
An On disk model still takes time to load or convert. Facts refresh after
successful or failed loads without moving the chosen row or its controls.

FLUX.2 [klein] accepts up to four gallery images as references. Add them with
the card's **Reference** button or `--reference 12,15` in the CLI. Their order
defines “image 1”, “image 2” in the prompt; drag chips or use Alt+Left/Right
to reorder them. Reference import from personal files is not implemented.
Deleted references are removed after verification; submit again to run
without them. An image in the recycle bin cannot be used as a reference or
a refine source until it is restored.

Place `.safetensors` LoRAs in `loras/`, then select an adapter and weight.
Weights can be swept without reloading the checkpoint. Incompatible adapters
are hidden/refused; unreadable compatibility information is reported as
unchecked. LoRAs are unavailable with Nunchaku, both Anima variants (Turbo
and Aesthetic) and NewBie.
The app records references, adapter and weight for Reuse and Resume.

## Refine an existing image

Open **Refine** on a gallery image. Its dialog starts from the source's prompt,
sampling settings and seed, independently of the main panel. Choose a scale
and **Strength**; lower strengths retain more of the source, higher strengths
redraw more. Start with the checkpoint's default: the same strength means
different things on different schedules. **From panel** copies the panel's
prompt pair when you want to use a revised prompt.
For a source from another checkpoint, the dialog keeps its prompt and seed
but uses the running checkpoint's sampling defaults.

**Versions** makes one, two or four passes on consecutive seeds, one image at
a time. Enter `0.05,0.1,0.2` or `0.05-0.2x0.05` in Strength for a sweep;
**Compare 3** supplies nearby values that leave enough denoising steps.
Changing aspect ratio crops the source instead of stretching it. The size
guard permits up to any 2 MP preset at 2× (about 8.4 MP), but the dialog's
measured-size warning is the better guide to VRAM use.

Availability depends on the pipeline and is confirmed after loading.
Refine uses the loaded weights and a plain resize, not an external upscaler.
Direct rendering at a larger size changes the noise shape and composition
even with the same seed. In the lightbox, **Compare with source** shows a
divider or side-by-side comparison. Deleting the source leaves its passes
in the gallery, but they can no longer be reproduced from that source; a
source in the recycle bin is named as such, and restoring it makes its
passes reproducible again.

## Inpaint part of an image

**Inpaint…** repaints only the part of a gallery image you paint over and
keeps the rest exactly. Open it from the lightbox, a card's or a result's
**✎**, or the right-click menu. It runs on the SDXL checkpoints (Animagine,
Illustrious, Kohaku, WAI, RouWei, NoobAI) and the Z-Image family (Z-Image
Turbo, Z-Anime); elsewhere the button is held and its tooltip says why.
FLUX.1, FLUX.2 [klein] and Qwen-Image are not enabled yet.

Paint the area to redraw over the picture. With the picture focused:
B brush, E eraser, `[` and `]` (or Alt+wheel) size the brush, I inverts,
Ctrl+Z and Ctrl+Shift+Z/Ctrl+Y undo and redo, and the arrow keys move the
brush with Space painting. These keys act only while the picture has focus,
so typing in the prompt never paints. **Feather** softens the mask's edge;
**Show mask** hides the overlay. **Clear** empties the mask, and Undo brings it back.

The dialog starts from the source's prompt, seed and sampling settings, with
**Strength**, **Versions** and **Seed** working as in Refine; the prompt and
sampling boxes are under **Prompt and sampling**. Strength says how much of
the masked area is redrawn. It starts at the installed diffusers inpaint
class's own default (about 1 on both families), since no checkpoint author
gives one; the note under it says where the card measured that default as the
right start (WAI v17, Animagine XL 4.0 and Z-Anime) or that this checkpoint was
not measured. On Z-Image checkpoints the preview line also says how much noise
the pass starts at, as Refine's does. Several comma-separated values run side
by side and open as a sheet. Ctrl+Enter runs.

An inpaint always runs at the source's size. Afterwards the result is laid
over the source through the mask, so every unmasked pixel is the source's
own and a feathered edge blends. The mask is stored by its hash and also
embedded in the result's PNG, so the image alone can be reproduced (Reuse,
`--from-image`, a bundle or reindex). Closing without running keeps your strokes
for the next time you open that image, until the page reloads. Re-render,
More seeds and Swap tags rerun an inpaint with its mask. A Re-render at
another size refuses it (unpick it, or search `inpaint:no`), and Mass
refine drops the mask. Search `inpaint:yes`
for inpaints, `mask:<hash prefix>` for one mask's.

On the card, an inpaint costs about what a refine pass of the same size
costs: SDXL runs the guard's 8.4 MP top with room to spare, while Z-Anime
fills the card at 2.4 MP. Lower strengths keep the masked area's old colours,
so the default is the place to start; see [verified.md](verified.md#inpaint).

## Outpaint an image

**Outpaint…** extends a gallery image's canvas past its edges and paints the
new area, on the same checkpoints as Inpaint. Open it from the lightbox, a
fresh result's **⬚ Outpaint…**, the right-click menu, or Reuse of an
outpainted image.

Type how far to extend each side (**Top**, **Left**, **Right**, **Bottom**),
in steps of 16 px; the frame shows the new canvas with the original in
place and the new area hatched. **+25% all round**, **to 16:9** and
**to 2:3** replace all four sides. **To shape…** picks another aspect,
and **Apply** applies it. A preset already met says so and stays held. The server
fills the new area from the source's stretched and blurred edge pixels, then
builds the mask: the new border plus a feathered **Band** reaching that far
into the original (0-256 px in steps of 8). The starting 32 joined best of
0, 32, 64 and 128, judged by eye on WAI v17, Animagine XL 4.0 and Z-Anime:
0 leaves a hard cut at the original's edge, 64 a faint double image across
the band, and 128 a plain one. You never paint it. A wider band blends the seam over
more of the original. Inward of the band, the original is kept pixel for pixel.

Strength, Versions, Seed and the prompt work as in Inpaint. The new area
starts as the server's rough fill, so expect a high strength there. The
pass runs at the source's size plus the padding, within the size guard.
Search `outpaint:yes` for outpaints; `inpaint:yes outpaint:no` gives the
painted inpaints alone. Lower strengths let the blurred fill show through;
see [verified.md](verified.md#outpaint).

## Sweep settings with Grid search

Blank axes use the main panel's value; populated axes cross with one another.
The preview shows resolved prompts, image count, existing cells and, when
previous renders provide measurements, time and disk estimates.

| Axis | Syntax | Example |
| --- | --- | --- |
| Steps, CFG | Single value, comma list, range with optional step | `6-12x2`, `0,0.5,1` |
| Mu | The same, plus `none` for the checkpoint's default shift | `none,1.15` |
| Seeds | Count of consecutive seeds crossed with all settings | `4` |
| Seed list | Explicit seeds instead of Seeds and the panel's Seed | `4471,8823`, `100-103` |
| LoRA weight | Numeric axis with an adapter selected | `0-1x0.25` |
| Budget | Minutes before starting no further cell | `420` |

A range without a step counts in ones; write `0.8-1.2x0.1` for decimals.
Mu `none` preserves the checkpoint's schedule, including any fixed shift;
it differs from explicit `0`. Unavailable controls stay unavailable in grids.
Each axis is limited to 64 values and the default whole-grid limit is 1,024
images. Review warnings about duplicate effective settings before running.

Prompt variables use `{name}` placeholders and pipe-separated values:

```text
Prompt: {size} dog sitting on {ground}

size = big | small | none
ground = grass | dirt
```

This makes six prompts. `none` removes that fragment; commas remain part of
a value. Unused variables and placeholders without values are refused.
Use **Run grid** to substitute variables, not Generate. Seeds are the outer
loop: each seed is tried across all settings before moving to the next.

To continue, restore the same grid, size, checkpoint, quantizer and starting
seed, then tick **Resume**. A Seed list needs no separate starting seed.
The panel's **Resume** tick skips settings the gallery already holds,
including matching images from other jobs and live queued work. An entirely
completed grid is refused. A hand-built Resume still needs the original
prompt, axes and base seed in the panel.

**Finish run** on a finish line, last-run banner or run sheet uses the
server's stored cells instead. It works after a tab reload or server restart,
including stopped Repeat/comparison jobs and runs that made no image. Your
current composer stays as it is. Its preview states missing images/cells,
redraws of permanently deleted successful cells, and the original budget.
Binned images count as present. Model/quantization changes, unreadable older
cells, missing sources/masks/LoRAs and already queued work give a held reason.
Older/imported runs without stored cells still use the panel's Resume tick.

Budget includes loading and queued waiting, but not time spent paused; a
cell started before expiry finishes. Saved images remain after cancellation,
a failure or a restart. Finish uses the same budget again and pins the
checked missing set before queueing it. It never adds new seeds; **More
seeds** remains the separate action for that.

## Browse, compare and keep results

The gallery initially shows the running checkpoint. Tick **All models** for
the rest, or **Starred** for favorites. **Runs** opens a run and its sheet.
`run:` includes all resumptions of a grid; `job:` selects one submission.
Search combines these terms with free text matching the prompt:

| Search | Matches |
| --- | --- |
| `run:ab12cd`, `job:ab12cd` | Run or submission |
| `id:412`, `id:400-420`, `seed:42` | Image ids or seed |
| `steps:6-12`, `cfg:<=1`, `tcfg:1.5` | Sampling values; `tcfg:none` means no active negative guidance |
| `mu:none`, `quant:fp8`, `model:flux` | Default shift, quantization, model substring |
| `model:=circlestone-labs/Anima` | One checkpoint's whole id exactly (a leading `=` on any text key) |
| `refine:412`, `refine:none`, `strength:0.3-0.5` | Passes from a source, direct renders, strength |
| `inpaint:yes`, `outpaint:no`, `mask:3f9a` | Inpaints, outpaints, one mask's by hash prefix (`mask:none` is no inpaint) |
| `lora:ink`, `lora:none` | LoRA name substring, or images made without one (`lora:=none` is an adapter named `none`) |
| `hash:3f9a` | Identical pixels by hash prefix |
| `size:1024x768` | Width and height exactly |
| `date:today`, `date:yesterday`, `date:2026-10-07` | A day in the server's local timezone |

**Re-render** renders the filtered images at the panel's size with their
recorded settings and skips completed or already queued cells - or, with cards picked with
**Select**, exactly those, whatever the filter ("Re-render 2 selected at
1024×1024"). A picked image moved to the recycle bin or deleted since is
left out and named. It refuses images from another checkpoint (picked ones
by number, to unpick). Star favorites at a smaller size, filter by Starred
or pick the keepers, and choose a larger size; expect new compositions for
direct renders. Where the row has no room for both, **Delete unstarred**
gives Re-render its place while cards are picked, so a pick never moves the
gallery; Clear brings it back (once the pointer has left the gallery, where
bringing it back would move the row under it).

**Sheet** lays out varying settings as rows, columns and selectors. Prompt
comparison versions take the rows; refine sweeps include their source.
**Fill width** enlarges cells, **Export PNG** saves the current sheet with
labels and settings, and **Export all** includes every pinned slice, splitting
large exports. For a prompt comparison, use its `job:` scope so version labels
retain their meaning. Star images directly on the sheet or in the lightbox.

In the lightbox, **Download** beside the zoom controls saves the full-size
PNG with its settings intact, named by image id, short model name and seed.

**Select** (first in the row under the search box) picks cards for an
action on several images: in select mode a click or Space picks a card,
Shift picks a range from the last card picked, and Enter still opens the
lightbox. Without Select, Ctrl+click picks a card and Shift+click (or
Ctrl+Shift+click) picks the range from it. The selection survives searches, filters and paging; its count
says how many picked images the current view does not show. **Clear** or
Esc empties it, and a second Esc leaves select mode. **Picked → Seeds to grid**
opens the grid and fills its Seed list with distinct picked seeds, in picked
order. Other axes and composer fields stay as they are. Unsafe browser
numbers are held with a reason; the normal server preview reports grid
limits and Batch duplicates. It never trims your picked list. Deleted images, and
images moved to the recycle bin, leave the selection on their own.

In Select mode, **Ctrl/Cmd+A** from a card or the Select button adds every
loaded card up to the selection cap. Picks hidden by another filter stay
picked; unloaded results are not fetched. In the bin it adds only bin cards.

**Picked ▾** (between **Clear** and **Select**; just ▾, with Clear a ✕,
where the row is narrow) is a menu for the picked cards: **Star picked**
and **Unstar picked** star or unstar all of them at once, and **Bin
picked** moves them to the recycle bin at once, as a card's ⌦ does - the
note it leaves has **Restore** for all of them. It is held, with the
reason, while nothing is picked and in the recycle bin. F still stars only
the card it is pressed on, in select mode too. There is no sheet of the
picked cards: a sheet lays out what the search box finds, and a run's
`run:` or `job:` search is the way to one.

**Swap tags…** (beside **Select**) runs the selected images again - or,
with nothing selected, what the filter matches - with some tags swapped:
each row finds comma-separated tags and puts its replacement where the
first of them sat (an empty replacement removes them), in the prompt or the
negative prompt. Case, spacing, underscores and a weight wrapping the
whole tag do not matter, and the replacement keeps the tag's underscores
and weight. Inside a prose prompt a single tag also matches as a phrase.
An image has to have every tag of every row; the dialog lists, per image,
what it would run (the swap marked, and what it takes out struck through
under it) or why it is left out - a missing tag,
already rendered, already queued, a deleted source or LoRA, the same as another image.
Each image keeps its own seed, size and settings; one job is queued per
size.

**Refine…** (beside **Swap tags…**) runs a refine pass of each selected
image - or, with nothing selected, of what the filter matches - from its
own pixels, at one strength and one scale; each keeps its own prompt,
negative prompt, seed, CFG, mu and LoRA, and Steps left blank keeps each
image's own. It opens at the scale last run, lowered where the largest
image would go past what fits on the card, and lists per image the pass it
makes or why it is left out: a refine pass itself (untick **Skip refine
passes** to refine those again), another checkpoint's (tick **Include
other checkpoints**: they take this checkpoint's steps, CFG and mu), too
large at this scale, no step left at this strength, a missing file or
LoRA, already refined, or this pass already queued. One job is queued per size.

**More seeds…** (beside **Refine…**) makes more images of each batch the
selection holds - or, with nothing selected, of each batch the filter
matches. A batch is the images of one run that differ only in their
seed - one cell of a grid - so picking one image of a Batch 4 takes all
four, and picking two of them still asks for one set more. Type how many
more each batch gets (4 to start, then the number last queued); where that
number times the batches is more than one press may queue (1024), the
dialog opens saying so, for a smaller number. Each new
image keeps the batch's prompt, size and settings, takes the next seed
after the batch's highest - stepping over any seed already made or still
queued at those settings - and runs at the Batch the batch was made with;
images from before the Batch was recorded run at the panel's **Batch**,
and a Batch too large for the size runs fewer at a time. The dialog lists
each batch's seeds now and the seeds it will get, or why it gets none
(another checkpoint's, a deleted refine source or LoRA). One job is
queued per run, and the new images join it: the **Runs** list, `run:` and
the run's sheet show them as one set. The same dialog opens where a run
ends, on that run's images rather than the selection; see
[Continue a run](#continue-a-run).

### Continue a run

"Continue a run" is one of two things, and each button names which:

- **Finish run** renders missing stored cells, including work a Repeat,
  comparison or grid did not reach. It is on finish lines, the last-run
  banner and run sheets. Reload/restart and an unrelated composer do not
  change its definition. Its preview names missing work, deleted successful
  cells to redraw, original budget and holds. Binned or queued cells are
  not duplicated. Older/imported runs without retained cells use the
  panel's explicit Resume tick.
- **More seeds…** makes new images with the same settings and the next
  seeds, joining the run. **More seeds…** on the finished-job line (after
  **Use seed** on a plain Generate's) takes the batches that job made: a
  plain Generate, a More seeds job, or a single refine, inpaint or outpaint
  pass. **More seeds of
  this batch…** - in an image's right-click menu, beside **Reuse** in the
  lightbox, and as **More seeds…** beside a fresh result at 1800 px and
  wider - takes the image's batch. **More seeds of this run…** - on a
  finished grid's line after **Show sheet**, as **More seeds…** on the
  last-run line beside **Show run**, and in the run's sheet beside **Finish run** - takes every image of the run, starred or not, whatever the
  gallery's filter; its dialog leads with what it will queue ("2 more of
  each of 6 cells = 12 images · seeds 8–9 · Starred ignored"). "N more" is
  N more images of every cell, not N more seed values crossed with the
  grid.

These open at the batch's own Batch where its batches agree on one (the
footer's More seeds… keeps the number last queued from it), and Enter
queues. They are held, with the reason as their tooltip, for images in the
recycle bin, another checkpoint's images, and while the server is not
answering - all but the last-run line's, which is held only while the
server is not answering: its dialog marks the cells another checkpoint
made and queues none of them. Pause and **Resume** on a running job are a different thing:
they hold and carry on the job itself. A run's sheet ends with a line
naming the run, its image count and the checkpoint it was made on.

### Keyboard and lightbox

| Where | Keys |
| --- | --- |
| Anywhere | Ctrl/Cmd+Enter generates; inside an open Grid field runs the grid; in Refine runs the pass; in Refine… (Mass refine) runs the batch; in Swap tags queues; in More seeds… queues the batches |
| Gallery | Tab enters the cards; arrows/Home/End navigate, Enter opens, F stars; in select mode Space picks, Shift+Space picks a range, Esc clears |
| Lightbox | Left/Right browse, F stars, R opens Refine, S compares a pass with its source or another image with its CFG neighbour, Delete moves the image to the recycle bin (in the bin: asks to delete it permanently), Esc closes |
| Lightbox zoom | Wheel or +/− zoom, drag pans, 0 fits, 1 or double-click shows actual pixels |
| Right-click an image | A menu of its actions: Star, Reuse settings, Use seed, More seeds of this batch…, Refine…, Inpaint…, Outpaint…, reference, Show run as sheet, Copy prompt / seed / image, Open PNG in new tab, Move to recycle bin. On a picked card: the picks' actions. The Menu key or Shift+F10 opens it from the keyboard; Shift+right-click gives the browser's own menu |
| Contact sheet | Arrows move between populated cells; Home/End use row ends, Ctrl+Home/End the corners. Tab reaches the current cell's star; Enter/Space opens its lightbox, F stars |
| Prompt/tag chips | Tab/Enter accepts a suggestion after three matching characters; Alt+Left/Right reorders chips, Delete removes, Enter edits |
| Prompt replacements | Ctrl+Z restores text after Apply, Reuse (or a loaded PNG), Expand, Examples or a chip edit |
| Inpaint | With the picture focused: B brush, E eraser, `[` `]` or Alt+wheel size, I invert, Ctrl+Z / Ctrl+Shift+Z undo and redo, arrows move the brush, Space paints; Ctrl+Enter runs (Outpaint too) |
| Refine | Enter in a one-line field runs; Strength arrows step by 0.05, or 0.01 with Shift |

Lightbox zoom and position carry between images of the same shape, useful
for comparing a detail across seeds or settings. Its facts include source,
references and run links, plus **Use seed** and **Use these references**.
**More seeds…** beside **Reuse** makes more of the image's batch, the dialog
opening over the lightbox (Esc closes the dialog first); at 900 px wide or
narrower it is in the right-click menu instead.
**⌕** beside a setting (model, size, sampling, LoRA, source, seed, day, run)
adds it to the gallery search; a value of the same kind replaces the old
one. **⌕ Search for this prompt** replaces the search's text with the
prompt's words and keeps its filters; it finds prompts containing all of
those words in any order.

**Compare CFG** in the lightbox sets the image beside another that differs
from it only in CFG: the same prompt, negative, seed, steps, size, mu, LoRA,
references and checkpoint, such as the cells of a CFG grid or two Generates
at one seed. The lower CFG is on the left (or on top), with **Swipe** or
**Side by side** as for a pass. **← Lower CFG** and **Higher CFG →** step
along the values found, and **Against** picks the other image. S turns it on
and off on any image that is not a refine pass. Where no other CFG value is
found the button is held and its tooltip says why; the gallery is searched
in its newest images, so the tooltip says when more may be further back.

## Generate from the terminal

The CLI uses the same backend and gallery without a browser:

```powershell
uv run python -m backend.generate "a lighthouse in fog"
uv run python -m backend.generate "a lighthouse in fog" --aspect 16:9 --mp 2 --runs 12 --seed 42
uv run python -m backend.generate "{size} dog" --grid-prompt "size = big | small | none" --grid-steps 6-12x2 --grid-seeds 4 --seed 42 --resume
uv run python -m backend.generate --from-image outputs/example.png --refine 412 --scale 1.5
uv run python -m backend.generate --help
```

`--dry-run` prints resolved settings without loading weights; `--quiet`
prints one path and seed per saved image. `--true-cfg`, `--reference 12,15`,
`--lora style.safetensors --lora-scale 0.8` and `--grid-lora-scale 0-1x0.25`
match the browser controls. Refine supports `--strength` and `--grid-strength`.
`--refine 412 --mask mask.png` inpaints: white repaints, black keeps, grey
blends, at the source's own size. `--refine 412 --outpaint 0,0,256,0` (left,
top, right, bottom) or `--outpaint 16:9` / `--outpaint +25%` outpaints, with
`--band` setting the band (default 32).
`--from-image` restores PNG settings; explicit flags override them. Add
`--scale` explicitly to enlarge its recorded size. Quantization is not carried
over, allowing a WSL PNG to be used on Windows, and differences are reported.

Without `--via-server`, the CLI loads its own weights. It first probes the
configured server and stops if it is ready, loading or times out; use
`--via-server` or unload the server model first.
`--ignore-server` overrides this advisory check, which cannot see a server
across the Windows/WSL boundary. Ctrl+C cancels cooperatively and preserves
saved images. Exit codes: 0 complete, 130 cancelled, 2 bad arguments, 1 failed
load or skipped cells, 4 server holding the GPU. Failed cells are reported
and skipped; repeated failures stop the run. If another Local Diffusion
process holds the gallery (a bundle export, say) when an image is ready, the
CLI says once that it is still waiting to save and keeps the image until the
gallery is free; a second Ctrl+C ends it.

Redirected CLI output is UTF-8. In Windows PowerShell, set
`[Console]::OutputEncoding = [Text.Encoding]::UTF8` before using `>` or `|`
to preserve non-ASCII prompts.

### Scripting the running server

Keep the browser server open and queue scripts through it:

```powershell
uv run python -m backend.generate "a fox" --via-server --seed 42 -q
uv run python -m backend.generate "a fox" --via-server --seed 42 --grid-steps 20,30 --grid-seeds 4 --resume
```

The address comes from `LOCAL_DIFFUSION_HOST` and `LOCAL_DIFFUSION_PORT`
(default `http://127.0.0.1:7860`). Defaults and request normalization belong
to that server. Explicit model, quantization and offload flags or environment
must match its current choices; select them in Models first. Each submission
pins the model revision. Nothing loads weights locally or switches the server.
`--quiet` prints the full server PNG URL and seed for each saved image.
Ctrl+C cancels only this invocation's job and waits for its step boundary;
a second Ctrl+C stops waiting. An unconfirmed submission is never retried
automatically: inspect the reported jobs URL before submitting again.

Grid, Resume, budget, gallery references, Refine, mask upload and Outpaint
use the public routes and the server's source provenance checks. PNG-carried
grid/comparison labels cannot pass through these generation routes and are
refused; define a new grid or use the direct CLI for those records.
`--dry-run` and `--ignore-server` are alternatives to `--via-server`.
The API explorer is at `/docs`, with its schema at `/openapi.json` on the
same server. Successful completion exits 0; cancellation (including an incomplete budget
stop) exits 130, bad arguments/refusals 2, and runtime failures 1. Saved
images remain available.

## Review starred rates

```powershell
uv run python -m backend.stats
uv run python -m backend.stats --run ab12cd --by steps,cfg,prompt --min 3
uv run python -m backend.stats --job ab12cd --by variant --min 4
```

The default scope is the newest run. Rates compare made versus starred,
excluding groups below `--min` (default 5). An untouched run is 0% starred,
not evidence of rejection. `--all` covers the gallery but mixes prompts and
conditions, so its results are only indicative. Use `--job` for comparisons;
if labels have different meanings across jobs, narrow the scope or group by
`variant,prompt`.

## Gallery bundles

Press **Bundles…** at the end of the row under the search box to open the
Gallery bundles dialog. Its counts cover the
entire gallery: every checkpoint, every favourite and the recycle bin,
regardless of the current search or filters; the preview says how many of
the records are in the recycle bin.

**Back up (keep gallery)** saves and verifies a bundle and leaves the
gallery exactly as it was: no count to type, nothing removed. Choose what it
holds under **Bundle**:

- **Entire gallery**: everything Bundle & clear would take (below).
- **Picked (N)**: the cards picked with **Select**, whatever the search or
  filter. Picks made before opening the recycle bin still count while it is
  shown. A picked image binned or deleted since is left out, and the result
  says how many were.
- **Starred (N)**: the starred images of every checkpoint, outside the
  recycle bin (up to the grid limit, 1024 by default).

A picked or starred bundle holds those images, their thumbnails and their
gallery records only: no other prompt travels in it, which makes it the way
to give someone ten images with their settings. A refine source or a
reference image travels only when it is chosen too; the result counts the
links that point outside the bundle, and an import of it says the same and
leaves those links empty.

To move the whole gallery out, keep **Entire gallery**, type the displayed
record count and press **Bundle & clear gallery** (a picked or starred
bundle never clears anything). To protect the bundle, enter a password twice first: the bundle is
then saved as an encrypted `.ldbundle` file that only Local Diffusion can
open (file names included). The password is not saved anywhere and cannot be
recovered, and the gallery is cleared after export, so keep it safe. Finish or cancel generation first; loading and other gallery work
must also finish before this operation can start.

The server saves the bundle (a ZIP, or with a password an encrypted
`.ldbundle`) in `bundles/` beside its gallery database, reads it back and
checks every file against its checksum (decrypting and authenticating every
part of a protected bundle), and only then removes the original PNGs and
thumbnails. The old database is replaced with a compact empty gallery, ready
for new images. The bundle includes the database, full-size PNGs, thumbnails
and unindexed PNGs; generation settings, favourites, timings and image
relationships are retained, and images in the recycle bin come back in it.
Model weights, LoRAs and browser preferences are not part of a gallery
bundle.

The dialog lists saved bundles with **Download** and **Import** controls.
You can also choose a bundle file from your computer and import it. Saved
bundles marked **Protected**, and selected `.ldbundle` files, need the
password under **Password for protected bundles**; a wrong password is
refused before a selected file is uploaded. Import
merges into the current gallery, keeps existing images, remaps source-image
links, and skips images already present (an image already present keeps
this gallery's star and recycle-bin state). An image the gallery still lists
but whose file is gone is restored from the bundle when the bundle holds that
same picture. Different files using the same name
are refused before changing the gallery. Source images the bundle does not
hold cannot be linked; the result counts those links, which are left empty.

After import, **Show new gallery images** opens the newly added live rows;
**Show new images in bin** opens additions that were binned when bundled.
These actions clear old search, Starred and model filters. No new rows means
no navigation action, even when missing files of known rows were restored.

Keep the server running until the operation finishes; closing the dialog or
browser does not cancel it. **Cancel upload** stops a selected file only
while it is still being sent (so does closing the tab then); nothing is
imported and the partial upload is removed. Once the upload has arrived the
import is the server's to finish. If another Local Diffusion process (such as
a running `reindex`) is using the gallery, the operation fails at once and
can be retried when it ends. Creating and importing a bundle need free space
for the archive and staged files: the bundle and the merged database on the
database's drive, uploads and unpacked images in `.local-diffusion-staging`
on the images' drive. An export without room for every file it would copy
(the images, the database and its working copies) is refused before it
writes anything. If an image is locked or files change during
export, the verified bundle remains available and the dialog lists what was
not cleared. The status line shows the current step and how far it has got. Review that result before removing anything by hand.
A gallery image whose PNG is missing from disk stops the counts: the line
names it as `#N` with its file name, and **Show #N** closes the dialog and
searches the gallery (or the recycle bin) for it. Put the file back, or
move the image to the recycle bin and delete it permanently there. A
`.part` file in the images folder is an image still being written: try
again when the render finishes. One left by a crash or a power cut is
settled an hour after it was written, at the next server start or dialog
refresh: put in place when it is an image the gallery already lists,
removed otherwise. In Windows
and WSL, the saved-bundle directory belongs to the server environment whose
gallery is open.

## Clear or recover a gallery

Deleting an image in the browser - a card's **⌦**, a result's **Delete**,
the lightbox's **Delete** or its Delete key - moves it to the **recycle
bin** without asking. The note it leaves ("Moved #12 to the recycle bin.")
has a **Restore** button; deletes in a row gather into one note. Nothing
leaves the disk until the bin is emptied. While an image is in the bin it
is left out of the gallery, the runs list, search results, `stats` and
every action on several images; it cannot be starred, used as a reference
or refined from, and a queued job that uses it fails that image. A starred
image keeps its star in the bin and gets it back when restored, but while
binned it does not protect the images it was made from.

**Bin** (in the footer row, between **More seeds…** and **Bundles…**) shows
how many images the bin holds and opens it: every checkpoint's binned
images, newest first, narrowed by the search box. Each card offers
**Restore** and **Delete permanently**, which asks first; **Restore picked**
restores the cards picked with **Select** in the bin; cards picked in the
gallery are kept aside and picked again when you leave it.
**Delete picked permanently** confirms and deletes only those bin picks.
If a pick was restored, deleted elsewhere or changed while confirming,
nothing in the requested set is deleted; review the picks and try again.
**Empty recycle bin** deletes
every image in the bin from disk, whatever the search shows, after a
confirmation that gives the count and size and says which of them a live
image was made from (those can no longer be made again). Actions that work
on live images are held while the bin is open, the **Runs** picker among
them; typing in the search box narrows the bin rather than leaving it.
Press **Bin** again or **Show all** to leave it, or follow a search or run
link from an image. There is no automatic purge.

**Delete unstarred** moves unstarred images matching the current gallery
scope to the recycle bin after showing a preview and asking for the count.
It preserves starred images and their sources/references, and leaves images
already in the bin alone. Use an empty search or `key:value` terms; free
text is refused for deletion. With All models off, deletion stays within
the current checkpoint. From the terminal:

```powershell
uv run python -m backend.clear --dry-run
uv run python -m backend.clear --where "run:ab12cd" --keep-favorites
uv run python -m backend.clear --keep-favorites
uv run python -m backend.clear --bin
```

`clear` previews and asks before deleting; `--yes` skips confirmation. It
deletes for good, and reaches the recycle bin as well: the whole clear and
`--where` delete binned images like any other (the preview says how many),
and `--keep-favorites` does not keep a starred image that is in the bin.
`--bin` empties the recycle bin alone, as **Empty recycle bin** does; it
takes neither `--where` nor `--keep-favorites`. `--where` and `--bin`
delete only what their preview showed: if an image matching it arrives (or,
for the bin, a different image is binned) while the prompt waits, nothing
is deleted and the command exits 1 so it can be run again; **Empty recycle
bin** refuses the same way.
Without `--where`, it also removes orphan PNGs, which have no favorite flag
to preserve. Scoped deletion removes matching rows and files without that
orphan sweep. The Windows and WSL galleries are separate; run the command
in the environment whose gallery you intend to clear.

```powershell
uv run python -m backend.reindex --dry-run
uv run python -m backend.reindex
uv run python -m backend.reindex --from D:\backup\outputs --database D:\backup\local-diffusion.db
```

Reindex rebuilds the index from Local Diffusion PNG metadata, or copies in
another gallery with `--from`. Existing filenames are skipped and conflicts
reported without overwriting. PNG-only recovery loses favorites and durations,
and brings images that were in the recycle bin back into the gallery.
It restores reference/refine links only when recorded source identities can
be verified; older or missing sources lose those links and refine strength.
Use `--from` with the matching `--database` to preserve favorites, the recycle
bin and durations and remap source ids. Remapped ids are also written into PNG metadata.
Keep both the database and PNGs for a complete backup. See
[design notes](design-notes.md) for implementation details and limitations.
