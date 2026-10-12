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
the selected checkpoint loads. A failed load shows its reason and **Retry load**;
you can also select another checkpoint.

Prompts and saved images stay available. Sampling starts from that checkpoint's
defaults or its remembered settings, and the guide, supported controls, adapters
and gallery scope update. **All models** shows images from the other checkpoints.
Reopen an existing Refine dialog after a switch to use the new schedule. Retry
and Resume for an earlier submission require switching back to its checkpoint.
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
(INT8 a little softer), BF16 is full precision and the largest, and Nunchaku
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
you apply it; the others follow changes saved in another tab.

**Speed** uses recorded RTX 5080 samples around 1 MP (WAI's portrait is 1.38 MP):
**Fast** is up to 15 seconds, **Medium** over 15 through 45 seconds, and **Slow**
over 45 seconds. The sample time and area appear below the category. A
category appears only when the sample used your chosen quantization, offload
and platform. Otherwise the word is **Reference**, or **Unmeasured** for BF16
or sequential offload, and "timed with FP8" or "timed on 4-bit Linux build"
says how the sample differs; these read dimmer than a category. **First run, busy PC** marks the busy-CPU first
images. The tooltip gives steps, settings and date. **Unmeasured** models
without a sample show default steps instead of an invented timing. These are
previous runs, not estimates for your current prompt, size or steps; loading
time is separate.

Some checkpoints are gated on Hugging Face. FLUX.1 Krea [dev] and FLUX.1 [dev]
need their licence accepted on the model page; Krea 2 Turbo and Raw need the
authors' approval, which can take days. Then sign in once with
`uv run hf auth login`. The 4-bit lite-infer builds are not gated. Each
row says **Open**, **Accept licence** or **Needs approval** under the name.
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
picker. If a switch cannot release the running model, that model stays
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
  The model remains loaded, so Unload is unavailable while paused.

The browser remembers prompts and controls; sampling settings are remembered
per checkpoint. **Reset** restores defaults. Resume is never remembered as
ticked. **Reuse** on an image restores its settings; check the warning when
reusing one from another checkpoint. Prompt replacements support Ctrl+Z.

## Write and check prompts

**Examples** supplies the authors' prompts; templates select the first slot
to fill. **Tags**, where available, supplies the checkpoint's own vocabulary
and recommended negative prompt. Its comma chips can be reordered or removed.
The negative prompt is inactive when the current guidance settings do not
use it; the caption explains how to enable it.

**Check** measures the prompt with the checkpoint's own tokenizers and checks
its documented guidance. It runs on the CPU without loading image weights,
including while images generate. The first use may download tokenizer files
(megabytes); after a 30-second wait, any unfinished download continues for
the next check. Encoder rows show token counts and where reading stops.
Order checks apply to supported comma-tag prompts; prose ordering advice
does not imply every sentence has been checked. Recheck after editing.

Tag suggestions and relevant checks query Danbooru's index with tag text.
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
generation submissions while it runs.

## Use references and adapters

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
unchecked. LoRAs are unavailable with Nunchaku, Anima Turbo and NewBie.
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
Resume skips settings the gallery already holds, including matching images
from other jobs. An entirely completed grid is refused. The browser records
a drawn grid seed in the Seed box; if recovering it from images, use the
oldest image's seed, not the newest. **Resume this run** helps restore the
seed and size, but the panel must still contain the matching grid.

Budget includes loading and queued waiting; a cell started before expiry
finishes. Saved images remain after cancellation, a failure or a restart.
The finished-job line offers retrying skipped cells where possible.

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
| `hash:3f9a` | Identical pixels by hash prefix |
| `size:1024x768` | Width and height exactly |
| `date:today`, `date:yesterday`, `date:2026-10-07` | A day in the server's local timezone |

**Re-render** renders the filtered images at the panel's size with their
recorded settings and skips completed cells. It refuses images from another
checkpoint. Star favorites at a smaller size, filter by Starred and choose
a larger size; expect new compositions for direct renders.

**Sheet** lays out varying settings as rows, columns and selectors. Prompt
comparison versions take the rows; refine sweeps include their source.
**Fill width** enlarges cells, **Export PNG** saves the current sheet with
labels and settings, and **Export all** includes every pinned slice, splitting
large exports. For a prompt comparison, use its `job:` scope so version labels
retain their meaning. Star images directly on the sheet or in the lightbox.

**Select** (under the search box, beside **Bundles…**) picks cards for an
action on several images: in select mode a click or Space picks a card,
Shift picks a range from the last card picked, and Enter still opens the
lightbox. The selection survives searches, filters and paging; its count
says how many picked images the current view does not show. **Clear** or
Esc empties it, and a second Esc leaves select mode. Deleted images, and
images moved to the recycle bin, leave the selection on their own.

**Swap tags…** (beside **Select**) runs the selected images again - or,
with nothing selected, what the filter matches - with some tags swapped:
each row finds comma-separated tags and puts its replacement where the
first of them sat (an empty replacement removes them), in the prompt or the
negative prompt. Case, spacing, underscores and a weight wrapping the
whole tag do not matter, and the replacement keeps the tag's underscores
and weight. Inside a prose prompt a single tag also matches as a phrase.
An image has to have every tag of every row; the dialog lists, per image,
what it would run (the swap marked) or why it is left out - a missing tag,
already rendered, a deleted source or LoRA, the same as another image.
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
LoRA, already refined. One job is queued per size.

**More seeds…** (beside **Refine…**) makes more images of each batch the
selection holds - or, with nothing selected, of each batch the filter
matches. A batch is what hovering a card shows: the images of one run that
differ only in their seed, so picking one image of a Batch 4 takes all
four, and picking two of them still asks for one set more. Type how many
more each batch gets (4 to start, then the number last queued). Each new
image keeps the batch's prompt, size and settings, takes the next seed
after the batch's highest - stepping over any seed already made or still
queued at those settings - and runs at the Batch the batch was made with;
images from before the Batch was recorded run at the panel's **Batch**,
and a Batch too large for the size runs fewer at a time. The dialog lists
each batch's seeds now and the seeds it will get, or why it gets none
(another checkpoint's, a deleted refine source or LoRA). One job is
queued per run, and the new images join it: the batch's hover, the
**Runs** list and `run:` show them as one set.

### Keyboard and lightbox

| Where | Keys |
| --- | --- |
| Anywhere | Ctrl/Cmd+Enter generates; inside an open Grid field runs the grid; in Refine runs the pass; in Refine… (Mass refine) runs the batch; in Swap tags queues |
| Gallery | Tab enters the cards; arrows/Home/End navigate, Enter opens, F stars; in select mode Space picks, Shift+Space picks a range, Esc clears |
| Lightbox | Left/Right browse, F stars, S compares a pass with its source, Delete moves the image to the recycle bin (in the bin: asks to delete it permanently), Esc closes |
| Lightbox zoom | Wheel or +/− zoom, drag pans, 0 fits, 1 or double-click shows actual pixels |
| Contact sheet | F stars the focused cell |
| Prompt/tag chips | Tab/Enter accepts a suggestion after three matching characters; Alt+Left/Right reorders chips, Delete removes, Enter edits |
| Prompt replacements | Ctrl+Z restores text after Apply, Reuse, Expand, Examples or a chip edit |
| Refine | Enter in a one-line field runs; Strength arrows step by 0.05, or 0.01 with Shift |

Lightbox zoom and position carry between images of the same shape, useful
for comparing a detail across seeds or settings. Its facts include source,
references and run links, plus **Use seed** and **Use these references**.
**⌕** beside a setting (model, size, sampling, LoRA, source, seed, day, run)
adds it to the gallery search; a value of the same kind replaces the old
one. **⌕ Search for this prompt** replaces the search's text with the
prompt's words and keeps its filters; it finds prompts containing all of
those words in any order.

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
`--from-image` restores PNG settings; explicit flags override them. Add
`--scale` explicitly to enlarge its recorded size. Quantization is not carried
over, allowing a WSL PNG to be used on Windows, and differences are reported.

The CLI loads its own weights. It first probes the configured server and
stops if it is ready, loading or times out; unload the server model first.
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

Open **Gallery bundles** beside the gallery controls. Its counts cover the
entire gallery: every checkpoint, every favourite and the recycle bin,
regardless of the current search or filters; the preview says how many of
the records are in the recycle bin. Type the displayed record count and press **Bundle & clear
gallery**. To protect the bundle, enter a password twice first: the bundle is
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
are refused before changing the gallery. Missing source images cannot be
restored; the result reports links that could not be recovered.

Keep the server running until the operation finishes; closing the dialog or
browser does not cancel it. **Cancel upload** stops a selected file only
while it is still being sent (so does closing the tab then); nothing is
imported and the partial upload is removed. Once the upload has arrived the
import is the server's to finish. If another Local Diffusion process (such as
a running `reindex`) is using the gallery, the operation fails at once and
can be retried when it ends. Creating and importing a bundle need free space
for the archive and staged files: the bundle and the merged database on the
database's drive, uploads and unpacked images in `.local-diffusion-staging`
on the images' drive. If an image is locked or files change during
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
restores the cards picked with **Select**. **Empty recycle bin** deletes
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
