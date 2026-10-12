# Plan: inpainting and regional edit

A design plan for F6 in `docs/review-2026-10-12.md`, written 2026-10-12 against
main at 586b941. Nothing here is built. Every statement about the installed
libraries was read from diffusers 0.40 in `.venv` (the files are named where
it matters); every statement about the app was read from the code and from
`docs/design-notes.md`, `docs/rules.md` and `docs/verified.md`. What only the
card can say is listed as an open measurement, not guessed.

## What it is

Inpainting re-denoises the part of a gallery image under a painted mask and
keeps the rest. In this app it is a refine pass with a mask: the same source
(a gallery id), the same strength, the same prompt, seed and settings, run
through the inpaint class diffusers ships beside the loaded pipeline instead
of its image-to-image class. Everything the refine pass already solved - the
source as an id, one call per image, the sibling built over the loaded
components, the schedule index a strength names, the source deleted or binned
mid-pass, the size guard - is reused. What is new is the mask: a picture the
user paints, which has to become part of the reproduction set as completely
as the source already is.

Out of scope for this plan: outpainting (it changes the source's size, see
"Later"), FLUX.1 Fill [dev] and other inpaint-only checkpoints, ControlNet
inpainting, and importing outside pictures (U61 part B, which stays deferred;
see "Relation to Refine and to U61 B").

## The installed pipelines

`diffusers/__init__.py` exports an inpaint class for five of the catalogue's
families, each named by the library's own pattern `{Base}InpaintPipeline` -
the same naming `refine.sibling` already relies on for `Img2Img`. A script
over `inspect.signature` of each class (run 2026-10-12 against the `.venv`)
found:

- every one takes `image`, `mask_image`, `strength`, `generator` and
  `callback_on_step_end`, so each meets `load()`'s rule and `refine.REQUIRED`;
- every argument in `refine.CARRIED` that the text-to-image class takes, the
  inpaint class takes too (no `negative_prompt`, `true_cfg_scale` or `sigmas`
  would be dropped by the backend's drop loop);
- every inpaint `__init__` takes the same components as its text-to-image
  class. The one difference is SDXL's `requires_aesthetics_score`, a config
  flag with a default, which `refine.build` already fills from the config or
  leaves at its default. So `refine.build` constructs each over the loaded
  components with no new weights, and `from_pipe` stays avoided for the
  reason the refine notes give (it ends in `.to(dtype=...)` over modules the
  loaded pipeline still uses).

| Family (catalogue entries) | Text-to-image class | Inpaint class | New weights | `strength` default in the class | `padding_mask_crop` | VAE encode of the source |
| --- | --- | --- | --- | --- | --- | --- |
| FLUX.1 (Krea dev, dev, schnell) | `FluxPipeline` | `FluxInpaintPipeline` | none | 0.6 | yes | sampled, per generator in a list |
| SDXL (Animagine, Kohaku, Illustrious, WAI, RouWei, NoobAI) | `StableDiffusionXLPipeline` | `StableDiffusionXLInpaintPipeline` | none | 0.9999 | yes | sampled, per generator in a list |
| Z-Image (Turbo, undistilled, Z-Anime) | `ZImagePipeline` | `ZImageInpaintPipeline` | none | 1.0 | no | sampled, per generator in a list |
| Qwen-Image | `QwenImagePipeline` | `QwenImageInpaintPipeline` | none | 0.6 | yes | sampled, per generator in a list |
| FLUX.2 [klein] (4B, 9B, base 4B, base 9B) | `Flux2KleinPipeline` | `Flux2KleinInpaintPipeline` | none | 0.8 | yes | `sample_mode="argmax"`, deterministic |
| Krea 2 Turbo / Raw, ERNIE-Image, NetaYume (Lumina 2), Anima, NewBie | - | none exported | - | - | - | - |

FLUX.1 Fill [dev] is `FluxFillPipeline`: a separate checkpoint (the review
records it as about 23 GB), whose call has no `negative_prompt` or
`true_cfg_scale` and which cannot render text-to-image at all, so it cannot be
a catalogue entry under the current model, where every checkpoint renders
from a prompt. It stays out (see "Later"). `QwenImageEditInpaintPipeline`
needs the Qwen-Image-Edit weights and is likewise out.

### Mask semantics, as the classes implement them

- **White is repainted, black is kept.** Each class's docstring says so
  ("White pixels in the mask are repainted while black pixels are
  preserved"), and a PIL mask is converted to one luminance channel.
- **The mask is binarised before it reaches the latents.** Every class
  builds its `mask_processor` with `do_binarize=True`
  (`VaeImageProcessor.binarize`: below 0.5 to 0, the rest to 1), then
  resizes it to the latent grid. A soft (feathered) edge therefore does
  nothing inside the denoise; it matters only where the decoded image is
  composited back over the source.
- **The kept region is re-imposed every step**, not once: after each
  scheduler step the loop writes `latents = (1 - init_mask) *
  init_latents_proper + init_mask * latents`, where `init_latents_proper` is
  the source's latents noised to the next timestep (`scheduler.scale_noise`
  on the flow classes, `add_noise` on SDXL). The FLUX.1 loop marks this
  "for 64 channel transformer only": the base checkpoints here are that
  case, not a dedicated inpainting transformer.
- **SDXL's class branches on the UNet's input channels** (`num_channels_unet`
  4 or 9). Every catalogued SDXL checkpoint is a 4-channel base UNet, so it
  takes the per-step blend path, as the FLUX.1, Z-Image and Qwen-Image
  classes do.
- **klein conditions on the source as well as blending it.** Its call
  appends the encoded source to the reference latents (`ref_images =
  [image_latents_encoded]`, then `image_reference` if given) and
  concatenates them onto the sequence every step, as its text-to-image call
  does with references. It takes **one** `image_reference`
  (`image_reference[0]` where a list is given), where the text-to-image call
  takes up to `MAX_REFERENCES`.
- **The unkept region still goes through the VAE.** Only with
  `padding_mask_crop` does a class paste the result back over the original
  (`image_processor.apply_overlay`, which uses the mask's grey levels as the
  alpha, so a feathered edge does blend there). Without it the whole picture
  is decoded from latents, and the "kept" area is the VAE's round trip of
  the source, not its pixels.
- **`padding_mask_crop`** crops the source and the mask to the masked area's
  bounding box (grown to the request's aspect and by the padding), denoises
  that crop at the request's `width` x `height`, and pastes it back at the
  source's size (`get_crop_region`, `apply_overlay`). Z-Image's class has no
  such parameter.
- **Strength and steps round as the image-to-image classes do.** SDXL's
  inpaint `get_timesteps` (with no `denoising_start`) runs
  `int(steps * strength)` steps; FLUX.1's, Qwen-Image's, Z-Image's and
  klein's floor the steps skipped. `refine.start_index` and `steps_run`
  already encode both, keyed off the profile.

### What the authors say about it

The rule for guides holds: cite the checkpoint's authors or say nothing. On
2026-10-12 the model cards were read for inpainting or masks:

- FLUX.2 [klein] 4B: "supports text-to-image, image editing, and
  multi-reference"; nothing on masks or inpainting.
- Qwen-Image: "precise image editing"; nothing on masks or inpainting.
- Z-Image-Turbo: names Z-Image-Edit, "a variant fine-tuned on Z-Image
  specifically for image editing tasks", as not yet released; nothing on
  masks or inpainting.
- The SDXL finetunes' and FLUX.1 base checkpoints' guides in `guides.py`
  say nothing about inpainting (no match for inpaint, outpaint or mask in
  `backend/` or the guides).
- FLUX.1 Fill [dev], the one card that is about filling ("capable of
  filling areas in existing images based on a text description"), lists
  "There may be slight-color shifts in areas that are not filled in" and
  "Filling in complex textures may produce lines at the edges of the
  filled-area." That is a different checkpoint; it is quoted here only as
  the closest author statement about the VAE round trip and the seam, and
  it must not be put in another checkpoint's guide.

So no catalogue checkpoint's author documents mask inpainting with its base
weights. Anything the app says about how well it works, or how to word an
inpaint prompt, is the card's measurement or `heuristic`, never a guide
sentence.

## Which checkpoints get it

The capability is the installed library's answer, as the refine pass's is:
`refine.sibling(pipeline, kind)` with `kind` either `"Img2Img"` or
`"Inpaint"`, under the same exact-export rule (only a class diffusers itself
exports under the pipeline's name), and an inpaint `REQUIRED` that adds
`mask_image`. With diffusers 0.40 that is FLUX.1 (Krea dev, dev, schnell),
the six SDXL checkpoints, Z-Image Turbo, Z-Image and Z-Anime, Qwen-Image and
the four klein checkpoints; and not Krea 2, ERNIE-Image, NetaYume, Anima or
NewBie. A profile carries `inpaint_supported` as its guess before a load,
`describe()` answers from the class after it, `/api/status` merges the two
and says whether the answer is settled (`inpaint_settled`), exactly as
`refine_supported` / `refine_settled` do, and a test holds every guess to the
installed diffusers.

klein matters most: it has no `Img2Img` sibling, so it has no refine pass,
and inpainting would be its first way to change part of an image.

The capability check has to split on the mask everywhere the code asks
"can this pipeline run a pass from a source": `submission.unusable_input`
(`refine_unavailable`), `applied()`, `refine.start_index` (which returns None
unless the profile's `refine_supported` - false on klein) and the page's
greying of ⤢. With a mask the question is `inpaint_supported`, without one
`refine_supported`.

## Memory and offload on 16 GB

What `verified.md` already says that bears on this:

- Sharing the loaded weights does not mean the same peak. With model
  offload a refine pass keeps the VAE resident beside the denoiser after the
  source's encode; WAI and FLUX.1 peaked about 0.16 GiB above a direct
  render of the same size.
- FLUX.1 Krea dev at fp8 sits near the top of the card at 1 MP (14,615-14,950
  MiB whole-card mid-denoise, the desktop included) and a 1.5x refine to
  2.4 MP reached 15,278 MiB whole-card, "within a gigabyte of the card's
  top"; 4.2 MP spilled.
- SDXL refines ran at 3.7-5.5 GiB allocated up to 8.4 MP; Z-Anime at 8.10 /
  10.58 GiB at 2.4 MP; Qwen-Image fits only as the NVFP4 re-upload under WSL,
  11.75 GiB at 1 MP.
- klein 4B at fp8 has a time but no peak recorded; references cost no
  measurable time under offload (10.2 s, 9.2 / 8.2 s with one, 11.3 s with
  two).

What inpainting adds, from the source read above: a second VAE encode per
image (the masked image, in `prepare_mask_latents`, on FLUX.1, Z-Image,
Qwen-Image and SDXL), the mask tensors at latent resolution, and on klein the
source's tokens on the sequence every step (the sequence at least doubles,
and with a reference triples). An inpaint runs at the source's own size
(below), so its pixels are those of the source, which may be a refine output
up to the refine guard's 8.4 MP.

Nothing about inpaint memory is known until it is measured. Open
measurements, each at fp8 with model offload on Windows unless named, one
size per run and stopping at the first that fills the card (the standing
rule for this card):

1. SDXL (WAI, then one of Animagine / NoobAI): 1 MP and the largest refine
   row (2560x3360) - allocated / reserved / whole-card, time per step.
2. FLUX.1 Krea dev: 1 MP. Whether the second VAE encode pushes it from
   15,278 MiB into a spill; if it does, inpaint on FLUX.1 needs
   `padding_mask_crop` at a smaller working size from the start.
3. Z-Image Turbo and Z-Anime: 1 MP and 2.4 MP.
4. klein 4B: 1 MP with no reference and with one (the sequence doubles and
   triples). klein 9B at 1 MP.
5. Qwen-Image under NVFP4 / WSL: 1 MP - first whether the inpaint class over
   a Nunchaku transformer runs at all.
6. For each: the device after a pass reads what it reads after a
   text-to-image call (the loaded pipeline's `maybe_free_model_hooks`
   resets the chain, as `_refine` does now).

The profile then gets an `inpaint_measured` row where it differs from
`refine_measured`, and the dialog's tight line uses it as the Refine dialog
uses `refine_tight_pixels`.

## The reproduction set

### The mask is a stored picture with an identity, never a path

A mask is a PNG the server stores and names by the SHA-256 of its decoded
pixels, all 64 hex digits - not the 16-digit `content_hash`, which the
bundle rules already refuse as an identity. Its canonical form is mode `L`,
exactly the source's width and height, 0 to 255, white repainted; the brush,
eraser, feather and invert are baked into those levels in the browser, so
the hash is the whole of what the user painted and there is no separate
blur or invert setting to record.

- **Storage.** `outputs_dir/masks/<hash>.png`, written as `.png.part` and
  renamed after its row commits, as `save_image` does. A `masks` table
  (`hash` primary key, `width`, `height`, `created_at`) created by
  `CREATE TABLE IF NOT EXISTS` on every connection beside `_migrate`, since
  a database file can be replaced under a running server.
- **Never a path.** The request carries the hash, validated as
  `^[0-9a-f]{64}$` and resolved through the table by one function
  (`storage.mask_path`, the counterpart of `reference_paths`), which names an
  unknown hash and a missing file. Nothing joins request text onto a
  directory.
- **Content-addressed, so no remapping.** Unlike a source id, a mask hash
  means the same thing in every gallery: a bundle merge or a reindex never
  renumbers it, and two identical masks are one file.
- **Masks are not gallery images.** They have no row in `images`, no
  thumbnail, no card, no star, no bin state, and they can never be a
  reference or a source. This keeps U61 B's problem (an imported picture
  becoming a row that is no render) out of this feature.

### The row and the chunk

- `images.mask TEXT` (the hash), through a `_migrate` entry, `_COLUMNS`,
  both explicit dicts (`save_image` and `storage.index_file`) and
  `_row_to_record`, which reads unguarded. `ImageRecord.mask`.
- The source stays `refine` and the strength `strength`: an inpaint is a
  refine pass with a mask. That keeps every piece of source machinery
  working unchanged - `refine_missing` / `refine_binned`, the merge's id
  remap, `refine_source` provenance, the lightbox's Against-source compare,
  the passes row, `refine:` in search. Labels read "inpaint of #N @ 0.6"
  where `mask` is set.
- The `local-diffusion` chunk gains `mask` (the hash) and `mask_source`
  (`filename`, the full hash, `width`, `height`), null together on every
  other row, as `refine` / `refine_source` are. The A1111 block gains
  `, Inpaint mask: <first 16 hex>` after `Denoising strength` and `Refine`.
- **Recommended: the mask's own PNG bytes travel inside the result's PNG**,
  base64 in a separate text chunk (`local-diffusion-mask`), so the PNG
  alone reproduces the image and a PNG-only reindex can recreate a mask file
  that was lost. A mostly-black `L` PNG is small next to the image; its size
  at 1 MP and 8 MP is an open measurement. The bundle code's identity
  comparison (pixel-deciding chunks, decoded identity) must be checked to
  ignore a text chunk, which it should, since the `local-diffusion` chunk is
  already one. This is an open question for the user.

### CellKey and run_id

`mask` becomes the fifth optional extension of `CellKey`, after refine:
(prompt, negative, steps, CFG, mu, true CFG, references, LoRA, refine, mask,
seed), `grid.mask_key(hash)` spelled once and built from a request in `grid`
and from the column in `storage.completed_cells`; `_KEY_EXTENSIONS` becomes 5.
`run_id` drops trailing None extensions, so every existing run id, refine
runs included, is unchanged - pinned in `tests/test_conditioning.py` beside
the existing pins.

One concrete hazard: `grid.duplicate_count` rebuilds a refine key as
`(*key[:8], f"{refine}@step:{start}", key[9])`, which assumes the seed is
element 9. With the mask at 9 that line would drop the mask and compare two
different masks as duplicates. It becomes `(*key[:8], ..., *key[9:])`.

### applied()

`GenerateRequest.applied()` stays the one normalisation:

- it takes `inpaint_supported` beside `refine_supported`;
- a mask without a source is dropped (a mask is not a setting without the
  image it covers, as a strength is not);
- with a mask, the source survives only where `inpaint_supported`; without
  one, only where `refine_supported` (today's rule). Where the source goes,
  the mask goes with it and the strength returns to its default;
- references: today a refine pass drops them (the sibling's `image` is the
  source). On an inpaint they survive only where the inpaint call takes
  `image_reference` (klein), and only one; more than one is refused at
  submit rather than truncated here, since a dropped reference renders
  something that looks like a success.

### Bundles, reindex, the bin, clear and lineage

- **Bundles.** The inventory gains `(outputs/"masks", "masks/", {".png"})`
  as its own top-level prefix. The export carries the `masks` table with
  the database. The import verifies each mask member as it verifies an
  image (hash of decoded pixels must equal its name, mode `L`, size equals
  its table row), merges `masks` rows by hash (an identical row is the same
  mask; one hash with another size is refused), and installs mask files
  that are not present. Recommended: the manifest stays `version: 1`. An
  older build skips members outside `images/` (the install loop's
  `if not member.startswith("images/"): continue`) and keeps the `mask`
  column through the unlisted-column fallback, so it imports the images and
  their recorded hashes and only cannot re-run them; raising the version
  would make every newer bundle unreadable to it. A row naming a mask the
  bundle does not carry keeps its hash (it still says what ran) and is
  counted as `masks_missing` in the result; Re-render of it is refused
  "the mask of #N is missing".
- **Reindex.** `--database` carries the `masks` table and the column.
  PNG-only recovery rebuilds `masks` rows from `masks/*.png` (a file whose
  pixel hash is not its name is ignored and counted), and from the embedded
  chunk where the file is gone. A result whose mask cannot be found keeps
  its hash, counted as unresolved, as `refine_source` mismatches are.
- **The recycle bin and deletes.** Binning a result changes nothing about
  its mask (a binned row is still "what was made", and a Restore must find
  the mask). A mask is removed only when no row, binned or live, names it,
  and no job queued or running names it, and it is older than the
  settle window (`PARTIAL_SETTLE_SECONDS`): uploaded masks wait unused while
  a dialog is open, and a queued cell's mask is in no row yet. The sweep
  runs where PNGs are already removed - Empty recycle bin, `DELETE
  /api/images/{id}`, `clear` - and asks `jobs` for the hashes in flight
  under the same reservation.
- **clear.** A whole `clear` removes `masks/` and the table with the rest.
  `clear --where` removes rows, then sweeps masks as above. The orphan sweep
  treats a `masks/*.png` with no table row as an orphan after the settle
  window, as it treats untracked PNGs.
- **Lineage and gallery actions.** Records carry `mask`, so
  `grid.cell_from_record` carries it with the source and strength:
  Re-render and More seeds run the same region again (More seeds' next
  seeds over the same mask), and Swap tags rewrites the prompt over the same
  mask, which is a natural regional edit. Mass refine makes the record
  itself the new source and drops the mask, which belongs to the old
  source's pixels. Reuse opens the inpaint dialog on the source with the
  mask loaded (`GET /api/masks/{hash}`).

## Request schema

- `GenerateRequest.mask: str | None`, pattern `^[0-9a-f]{64}$`, after
  `strength`. A validator refuses a mask without `refine` in words
  ("a mask needs the image it covers").
- **An inpaint runs at its source's size.** The mask is painted on the
  source's pixels; a reframing would have to crop the mask too, and the
  scale buttons belong to the Refine dialog. Checked at submit
  (`submission.unusable_input`, kind `mask`), with the database: the source
  exists and is live, the mask exists, mask size equals source size equals
  request size. Upscaling a region is a later phase (`padding_mask_crop`).
- **Strength.** The same field and the same `_a_fraction` refusal. Its
  starting value has no author source, so the profile gets an
  `inpaint_strength` key only once measured (open measurement below); until
  then the starting value is the inpaint class's own signature default,
  read with `inspect` at load (0.6 FLUX.1 and Qwen-Image, 0.9999 SDXL, 1.0
  Z-Image, 0.8 klein) and labelled in the tip as the library's default, not
  the checkpoint's. `check_refine_steps` applies unchanged (SDXL raises
  below one step).
- **Budget.** `check_budget` holds an inpaint to the refine pass's own
  pixels, as a pass is held now.
- **A mask that paints nothing is refused** at upload ("nothing is masked");
  an all-white mask is allowed and the dialog says it repaints the whole
  picture, which is a refine at that strength.
- **The upload route.** `POST /api/masks` takes a PNG body and the source's
  gallery id. It decodes with Pillow under a pixel cap (the refine guard's
  `refine_max_pixels`), converts to `L`, refuses a size other than the
  source's, refuses an empty mask, writes the file and the row, and answers
  the hash; the same picture answers the same hash. `GET /api/masks/{hash}`
  serves it for Reuse and the lightbox's mask outline. Both are model-free
  and take the gallery reservation as other storage writes do.
- **CLI.** `generate.py --mask FILE` beside `--refine ID`: the CLI is the
  user's own process, so a path is fine there; it stores the mask through
  the same function and records the hash. `--from-image` carries `mask` from
  the chunk. The deliberate copies in `jobs.submit` and `generate.py` stay
  equal under `tests/test_cli.py`.

## Grid and comparison

A mask is fixed across a job; it is not an axis. Everything else that is an
axis over a refine pass is an axis over an inpaint: the Strength box holding
a list or a range is `GridRequest.strength_axis` over one source and one
mask, the seed axis is Versions, and a prompt variable runs several prompts
over one region. Each cell's `CellKey` carries the mask, so resume, the
duplicate count and `run_id` treat two masks as two runs. The contact sheet
needs nothing new; a cell label can stay the strength or seed it varies.

The panel's prompt comparison never sends a source (`collect()` sends
`refine: null`), so it never runs an inpaint. A mask axis (several masks on
one sheet) is not planned; it would be a list-valued extension with its own
key rules.

## Backend and jobs

- `DiffusersBackend` gains `_inpaint_class`, `_inpaint_pipe`,
  `_inpaint_params` beside the refine ones, built on the first inpaint by
  `refine.build`, kept, and dropped on `unload` / `try_unload`.
  `describe()` reports `inpaint_supported`.
- `generate()` picks the runner by the request: a mask means the inpaint
  sibling, a source without one the image-to-image sibling. It hands the
  inpaint call `image` (the source, decoded and closed as now),
  `mask_image` (the stored `L` mask), `strength`, and on klein
  `image_reference` (one gallery reference). The source is not resized:
  the size is already the source's.
- **One call per image**, through `_refine`'s loop: FLUX.1, SDXL, Z-Image
  and Qwen-Image sample the source's VAE encoding per generator, and one
  call with one generator draws the sample and the noise from one stream,
  which is what makes image *i* of a batch a solo run of its seed. The
  `finally` that calls the loaded pipeline's `maybe_free_model_hooks`
  applies unchanged.
- **The composite.** After each decode, the backend pastes the result over
  the source with the stored mask as alpha, through the library's own
  `image_processor.apply_overlay(mask, source, image)` - the function the
  classes call with `padding_mask_crop` - so the unmasked pixels are the
  source's exactly and a feathered edge blends. This makes the "kept" area
  kept in pixels rather than through a VAE round trip, which is what the
  user is promised. It is part of what an inpaint *is* in this app, so it
  is not a setting; the chunk records `"mask_blend": "overlay"` so that a
  later change of rule can be told apart. The `content_hash` is of the
  composite. Whether the seam it can leave is worse than the round trip's
  colour shift is an open measurement (below).
- **Schedule and preview.** `schedule.forced_mu` reaches the inpaint call
  the way it reaches a refine (the scheduler object is shared), and the
  sigmas go in as for a refine. `START_RECIPES` gains each flow inpaint
  class's sequence-length and default-schedule statements, held to the
  source by `tests/test_refine_schedules.py`, so the dialog's "starts at N%
  noise" reads off the schedule. klein's class computes its shift with
  `compute_empirical_mu`, not `calculate_shift`, so it needs its own recipe
  entry or no reading. The live preview reads the callback's latents,
  which on the flow classes are the blended latents in the text-to-image
  layout, so FLUX.1 and Z-Image keep their preview; SDXL and klein have no
  fitted map today and show none, as now.
- **Jobs.** No change to the worker: an inpaint is a cell. Cancel and pause
  stay cooperative through `callback_on_step_end`. The re-check after a load
  ("fail if the live capability would drop the requested source") extends to
  the mask, and a cell checks that its mask file is still there as it begins
  (`storage.mask_path`), failing that cell in words.
- **Estimates.** `gallery_reports.refine_cost_ms` groups by denoised steps,
  size and checkpoint; it must also split on `mask IS NULL`, since klein's
  inpaint carries the source on the sequence and FLUX.1's encodes twice.
  An inpaint estimate needs inpaint history, as a refine estimate needs
  refine history.
- **Fakes.** `tests/fakes.py` and `tools/fake_server.py` gain an inpaint
  that changes only masked pixels (a fill by seed), so tests and smokes can
  read the result against the source.

## The UI

### A dialog, opened on the picture

The recommendation is an **Inpaint dialog of its own** (`web/inpaint.js`
and `web/inpaint-mask.js`, `web/styles/inpaint.css`), sharing the Refine
dialog's prompt check (`refine-prompt.js`), strength box rules, Versions,
Seed, preview and refusal handling, rather than a mode of the Refine dialog.
The Refine dialog's scale buttons, W and H, Keep shape and the kept-region
frame are all about resizing, which an inpaint does not do, and a reframing
would misalign the mask; a mode would grey half the dialog. The review's
plan put the canvas in the Refine dialog; this is an open question for the
user. Either way it is a modal dialog, never a panel mode ("A dialog, not a
mode of the panel"), and `collect()` still never sends a source or a mask.

Opened from the lightbox's Inpaint… (beside Refine, closing the lightbox
and given back on a close that runs nothing, as Refine does), a card's and
a viewer result's tool, and Reuse on an inpainted record. Where the
checkpoint has no inpaint class, each is `aria-disabled` with the reason as
its title and a press says why ("Krea 2 Turbo has no inpaint pipeline in the
installed diffusers"), with the guess / settled wording the Refine dialog
uses.

### The canvas

- The source as large as the dialog allows, with the mask drawn over it as
  a translucent tint and an outline, toggled with Show mask. At 1400x900,
  where the smokes run, the picture is the dialog's first row, as large as
  fits beside a fixed-height tools row; from 1800 px a two-pane layout in
  its own `@media (min-width: 1800px)` block. Nothing changes under 1800 px
  beyond the base layout.
- The canvas's backing bitmap is the source's pixel size; the brush is in
  source pixels, drawn at the display scale. At the refine guard's 8.4 MP a
  bitmap is about 34 MB of RGBA, so undo keeps strokes (points, radius,
  tool) and replays them rather than keeping snapshots, with a cap.
- **Tools:** Brush (B), Eraser (E), brush size ([ and ], and a slider),
  Feather (a slider, baked at export as a blur of the mask's levels; shown
  live), Invert (I), Clear, Undo (Ctrl+Z) and Redo (Ctrl+Shift+Z / Ctrl+Y).
  The keys act only while the canvas has focus, so typing in the prompt
  never paints and Ctrl+Z in a box stays the box's undo (the
  `writeBox` / `replaceBoxText` rule is untouched). Every tool is a button
  with its key in `aria-keyshortcuts`; the canvas is focusable, named
  "Mask over #N", and arrow keys move a keyboard brush with Space to paint,
  so a mask can be drawn without a pointer.
- **Pointer rules.** Painting uses pointer events with
  `setPointerCapture` and `touch-action: none` on the canvas only. A plain
  wheel scrolls the dialog as anywhere else (no wheel hijack); Alt+wheel
  changes the brush size. The dialog never scrolls by itself and never
  moves a control under the pointer: the tools row and the status line have
  fixed heights, a refusal replaces the preview line at its size (as the
  Refine dialog's does), and the canvas never resizes while the pointer is
  down. The dialog is anchored at a fixed top and grows downward, as the
  Refine dialog is; never `overflow-y: auto` on the panel.
- **Held, not disabled.** Inpaint is `aria-disabled` with the reason as its
  title until something is masked, while the mask uploads, and while the
  preview refuses; a press says why.
- **Upload.** On each stroke's end, debounced, the page exports the mask
  (`L` levels from the canvas, feather applied) and `POST`s it; the answer's
  hash goes into the preview request and the submit. Content addressing
  makes repeated uploads of the same mask free. A close that ran nothing
  keeps the strokes in page memory for the next open of the same image, as
  the Refine dialog keeps its edits, and never in `localStorage`.

### Elsewhere on the page

- Cards and the facts list say "inpaint of #N @ 0.6"; the facts list shows
  the mask as a small outline over the source's thumbnail.
- The lightbox's compare against the source works unchanged (the source is
  `refine`); an optional mask outline toggle on an inpainted record is a
  later nicety.
- Search: `mask:` (a hash prefix) and `inpaint:yes`; `stats --by mask`.
  Both are optional for the MVP.

## Check and the critic

Check is unchanged: the dialog checks its own prompt against the loaded
checkpoint's encoders, exactly as the Refine dialog does through
`refine-prompt.js`, cold and off the gate. The inpaint classes build their
prompt strings with the same `encode_prompt` as the text-to-image classes
on the families read here; `tests/test_prompt_encoding.py` should hold the
inpaint classes' copies too, so a drift fails a test.

No new rule. No catalogue author says whether an inpaint prompt should
describe the whole picture or only the region, so the dialog starts from
the source's prompt (what the rest of the picture already is) and says
nothing about how to word it; any advice would be `heuristic` and is not
proposed. The critic's sources are the guides, which have nothing on
inpainting, so it is unchanged and never asked about masks.

## Relation to Refine and to U61 B

An inpaint is a refine pass with a mask: same source field, same strength
semantics, same one-call-per-image invariant, same guard, same source
deletion handling. The difference is which sibling runs, the mask in the
reproduction set, and the composite. klein, which has no refine pass,
gains its first.

U61 part B (importing one's own photo or sketch) stays deferred. Inpainting
works only over gallery images: the source is a gallery id resolved by
`storage.reference_paths`, and a mask is never a gallery row, never a
source, never a reference. `POST /api/masks` accepts any PNG, but what it
stores can only ever be used as a mask over the gallery image it was sized
to, so it does not become an import path. If U61 B is built later, an
imported row becomes a source like any other and inpainting applies to it
with no change here.

## Tests and smokes

Python (`uv run --group dev pytest`, no GPU):

- `tests/test_inpaint.py`: `refine.sibling(..., "Inpaint")` for every
  catalogue class against the installed diffusers (exact export, `REQUIRED`
  with `mask_image`, `CARRIED`); every profile's `inpaint_supported` guess
  held to it; `refine.build` constructs each inpaint class over stand-in
  components; the class strength defaults are read, not copied; mask
  validation and canonical form; the composite keeps every pixel where the
  mask is 0 and takes the result's where it is 255.
- `tests/test_refine_schedules.py`: inpaint recipes held to the source;
  `start_index` against each inpaint class's `get_timesteps`.
- `tests/test_conditioning.py`: the CellKey widening and `run_id` pins
  (existing ids unchanged); `duplicate_count` with two masks.
- Storage: `_migrate` adds `mask` to an old database; the `masks` table on
  a replaced file; `save_image` and `index_file` write and read the chunk
  keys; `completed_cells` keys on the mask.
- `tests/test_api_inpaint.py`: the upload route (non-PNG, too large, wrong
  size, empty, idempotent hash), `GET`, submit refusals (unknown hash,
  missing file, size mismatch, binned source, more than one reference on
  klein, a checkpoint without the class), the grid preview, Re-render /
  More seeds / Swap tags carrying the mask and Mass refine dropping it.
- Bundles (`test_bundles*.py`): export carries masks and the table; import
  merges by hash, refuses a corrupted or renamed mask, counts
  `masks_missing`; password bundles; source cleanup.
- Reindex (`test_reindex*.py`): `--database` carries masks; PNG-only
  recovery from files and from the embedded chunk.
- `tests/test_clear.py` and bin tests: the mask sweep spares masks named by
  any row, by a queued job, or younger than the settle window.
- `tests/test_cli.py`: `--mask` and the equal loops.
- Jobs: cancel mid-inpaint, a mask file gone before a cell.

Smokes (a fake server, then `node tools/browser.mjs`), at 1400x900:

- `smoke-inpaint` (animagine fake, port 7904, fresh scratch `--data`): open
  from the lightbox; brush, eraser, size keys, feather, invert, undo and
  redo, the keyboard brush; Inpaint held until something is masked, with
  its reason; no self-scroll (`scrollTop` unchanged through painting and a
  refusal) and the tools row's boxes unchanged under the pointer; submit;
  the result says "inpaint of #N"; Reuse reopens with the mask; a strength
  list opens a sheet; the canvas's unmasked pixels equal the source's.
- `smoke-inpaint-unavailable` (default Krea 2 Turbo fake, port 7905): every
  entry point `aria-disabled` with the reason, and a press says it.
- Later: `smoke-inpaint-klein` (klein fake) for the one-reference limit,
  and a wide-tier case in `smoke-wide`.

`docs/rules.md` gains the rules (mask by hash, never a path; the fifth
extension; the composite; the sweep), its Tests section the smokes and
ports, `design-notes.md` an "Inpainting" section, and `source-audit.md` the
new files.

## Phases

Each phase ends on main with its tests and docs. Effort uses the review's
scale: S a day or less, M a few days, L a project.

**Phase 0 - a spike on the card (S).** A throwaway script, not app code:
load WAI and FLUX.1 Krea dev as the app does (fp8, model offload), build
each inpaint class with `refine.build`, run one 1 MP inpaint with a hand
mask, and record peak, time, determinism (same hash twice), batch image 2
against a solo run of its seed, the device after the pass, and how the
round trip and the composite look at the mask's edge. This decides whether
FLUX.1 needs the crop from the start, and confirms `build` over quantized,
hooked modules for the inpaint classes, before anything is designed around
it.

**Phase 1 - backend and reproduction set on SDXL (L).** `sibling(kind)`,
the inpaint runner and composite, `inpaint_supported` in profiles, status
and `describe()`; the `masks` table and files, `storage.mask_path`, the
upload and read routes, the column, the chunk, the A1111 key; `CellKey` /
`run_id` / `duplicate_count`; `applied()` and the submit checks; bundles,
reindex, the sweep, `clear`; the CLI; the fakes; every Python test above.
SDXL first because it has the most headroom on the card (3.7-5.5 GiB
allocated through 8.4 MP refines), runs a step in a fraction of a second so
card checks are quick, covers six catalogue checkpoints, and uses the
plainest inpaint path (4-channel UNet, per-step blend, no sequence growth).
Its cost: no live preview and no start-sigma reading on SDXL, which the
dialog already copes with for Refine.

**Phase 2 - the dialog (M).** `inpaint.js`, `inpaint-mask.js`, the CSS,
the entry points, Reuse, the card and facts wording, the two smokes, the
rules and design notes.

**Phase 3 - FLUX.1 and Z-Image (M).** Mostly measurement and recipes:
`START_RECIPES` for their inpaint classes, the preview through the blend,
`inpaint_measured` rows, the starting strength per profile once measured.
FLUX.1 Krea dev is the default checkpoint, so this is where most users
meet it; its margin on the card is the reason it is not first.

**Phase 4 - klein (M).** One reference beside the source, the refusal of
more, the `compute_empirical_mu` recipe or no reading, memory with the
sequence doubled and tripled, `smoke-inpaint-klein`. The highest value per
user for klein, which gets its first edit of part of an image.

**Phase 5 - Qwen-Image under Nunchaku (S-M).** Only if the inpaint class
over the Nunchaku transformer runs at all; WSL only, as the checkpoint is.

**Phase 6 - only the masked area (M).** `padding_mask_crop` with a working
size: a small region of a large image denoised at a preset size and pasted
back, which is both the memory answer for large sources on FLUX.1 and the
detail answer for small regions. A new recorded value (`mask_crop`, the
padding) in the row, the chunk and the mask key element (`hash+pad`),
`run_id` unchanged where it is None. Not available on Z-Image, whose class
has no such parameter.

Total: an L, about two weeks of work for phases 0-2 and a few days for each
of 3-6, plus card time that only the user can run.

## Open measurements (the card's to say)

1. Peak memory and time per family at 1 MP, and the largest size each holds
   (see "Memory and offload"), including FLUX.1 at 1 MP against the spill.
2. Determinism, batch-against-solo seeds, cancel-then-rerun and the
   text-to-image hash after inpaints, per family, as `verified.md` records
   them for Refine.
3. A starting strength per profile (`inpaint_strength`) that repaints the
   region while it still joins the surroundings, measured as the refine
   strengths were, and recorded as measured or arithmetic.
4. The seam: the composite's edge against the VAE round trip's colour
   shift, with and without feather, per family.
5. A LoRA through an inpaint (FLUX.1), weight 0 byte-identical to none.
6. klein with one reference: conditioning visible, memory, time.
7. Qwen-Image's inpaint class over Nunchaku's transformer under WSL.
8. The embedded mask's size in a result PNG at 1 MP and 8 MP.

## Risks

- **Reproduction completeness.** A mask is the first input that is a file
  outside the `images` table. Every place that carries a source - bundles,
  merges, reindex, the sweep, `clear`, lineage, Re-render, the CLI - has to
  carry the mask too, or an image becomes unreproducible without saying so.
  The tests above are the guard; the embedded chunk is the backstop.
- **Sweep races.** A mask uploaded for an open dialog or a queued job is in
  no row; sweeping it fails a cell. The settle window and the in-flight
  check under the reservation are the guard.
- **Quality.** No catalogue author documents base-model inpainting. Results
  may show seams or ignore the prompt in the region; the app can only say
  what was measured.
- **FLUX.1 memory.** Already at the card's top at 1 MP; an inpaint may
  spill where a text-to-image render does not.
- **klein's single reference.** A klein user with four references in the
  panel cannot carry them into an inpaint; the refusal must say so plainly.
- **Canvas size.** 8.4 MP masks in the browser: stroke replay keeps memory
  bounded, but export and blur time at that size are untested.
- **Bundle compatibility.** Older builds import newer bundles without their
  masks; this is deliberate but must be stated in the import result.

## Open questions for the user

1. A dialog of its own (recommended) or a mask mode in the Refine dialog,
   as the review's plan said?
2. Embed the mask's bytes in each result PNG (recommended, small, makes a
   PNG self-sufficient) or rely on the masks folder and bundles alone?
3. SDXL first (recommended, for headroom and speed on the card) or FLUX.1
   Krea dev first, the default checkpoint, accepting a measurement-first
   start?
4. Always composite over the source (recommended: unmasked pixels exactly
   kept), or offer the pipeline's own output as an option, which would add
   a recorded setting?
5. Keep the bundle manifest at version 1 (recommended) so older builds can
   still import newer bundles without masks, or bump it so they refuse?
6. The mask sweep's policy: swept with Empty bin, a delete and `clear` once
   unreferenced and settled (recommended), or kept until `clear`?
7. Later: outpainting (a larger canvas with the new area masked, which
   changes the source's size and so the size rule above), and FLUX.1 Fill
   [dev] as an inpaint-only checkpoint (separate weights, the
   non-commercial dev licence, and a catalogue model where not every
   checkpoint renders from a prompt alone). Wanted, and in which order?
