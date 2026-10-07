# Using Local Diffusion

For installation, launch options, sizes and environment variables, start with
the [README](../README.md). The [anime model guide](anime-model-guide.md) helps
choose a checkpoint; [verified.md](verified.md) records measured performance.

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
without them.

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
in the gallery, but they can no longer be reproduced from that source.

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
| `refine:412`, `refine:none`, `strength:0.3-0.5` | Passes from a source, direct renders, strength |
| `hash:3f9a` | Identical pixels by hash prefix |
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

### Keyboard and lightbox

| Where | Keys |
| --- | --- |
| Anywhere | Ctrl/Cmd+Enter generates; inside an open Grid field runs the grid; in Refine runs the pass |
| Gallery | Tab enters the cards; arrows/Home/End navigate, Enter opens, F stars |
| Lightbox | Left/Right browse, F stars, S compares a pass with its source, Delete asks to delete, Esc closes |
| Lightbox zoom | Wheel or +/− zoom, drag pans, 0 fits, 1 or double-click shows actual pixels |
| Contact sheet | F stars the focused cell |
| Prompt/tag chips | Tab/Enter accepts a suggestion after three matching characters; Alt+Left/Right reorders chips, Delete removes, Enter edits |
| Prompt replacements | Ctrl+Z restores text after Apply, Reuse, Expand, Examples or a chip edit |
| Refine | Enter in a one-line field runs; Strength arrows step by 0.05, or 0.01 with Shift |

Lightbox zoom and position carry between images of the same shape, useful
for comparing a detail across seeds or settings. Its facts include source,
references and run links, plus **Use seed** and **Use these references**.

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
and skipped; repeated failures stop the run.

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

## Clear or recover a gallery

**Delete unstarred** removes unstarred images matching the current gallery
scope after showing a preview and asking for the count. It preserves starred
images and their sources/references. Use an empty search or `key:value` terms;
free text is refused for deletion. With All models off, deletion stays within
the current checkpoint. From the terminal:

```powershell
uv run python -m backend.clear --dry-run
uv run python -m backend.clear --where "run:ab12cd" --keep-favorites
uv run python -m backend.clear --keep-favorites
```

`clear` previews and asks before deleting; `--yes` skips confirmation.
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
reported without overwriting. PNG-only recovery loses favorites and durations.
It restores reference/refine links only when recorded source identities can
be verified; older or missing sources lose those links and refine strength.
Use `--from` with the matching `--database` to preserve favorites and durations
and remap source ids. Remapped ids are also written into PNG metadata.
Keep both the database and PNGs for a complete backup. See
[design notes](design-notes.md) for implementation details and limitations.
