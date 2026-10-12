# Plan: outpainting

**Status, 2026-10-13: a plan; nothing is built.** Written on
`feat-outpainting`, branched from main at b3bc975 right after inpainting was
merged. It is grounded in the inpainting code as built (the functions are
named where it matters) rather than in `docs/plan-inpainting.md`'s design,
which the build departed from in places. The user has already decided:

1. **Controls:** a pixel box per side (left, top, right, bottom), each on the
   request's 16-pixel grid so the final width x height is valid, plus
   presets - "to 16:9", "to 2:3" (and the other aspects the size presets
   use) and "+25% all round" - and a frame preview showing where the
   original lands on the new canvas.
2. **Fill:** before denoising, the new area is the source's edge pixels
   stretched outward and blurred. The server computes it, deterministically.
3. **Mask:** the server builds it: the new border plus a feathered overlap
   band reaching into the original. The user sets the band's width in
   pixels and never paints an outpaint's mask.
4. **Checkpoints:** the inpainting ones - the six SDXL checkpoints, Z-Image
   Turbo, Z-Image and Z-Anime. FLUX.1, FLUX.2 [klein] and Qwen-Image are
   deferred with inpainting. The gate is `inpaint_supported`.
5. **Composite:** always over the padded source with the mask as alpha, as
   an inpaint is, so every unmasked original pixel is kept exactly.

Every statement about the app below was read from the code on this branch;
every statement about the installed libraries from diffusers 0.40 in
`.venv`. What only the card can say is listed as an open measurement, not
guessed. Nothing here claims anything about how outpainted pictures look.

## What it is

An outpaint makes the canvas larger than a gallery image and paints the new
area. In this app it is **an inpaint over a padded copy of the source**: the
same source (a gallery id), the same strength, prompt, seed and settings,
the same inpaint class beside the loaded pipeline (`refine.sibling(...,
"Inpaint")`, built over the loaded components by `refine.build`), the same
one call per image through `DiffusersBackend._refine`, and the same
composite (`refine.composite`). Two things differ from an inpaint, and
everything else in this plan follows from them:

- **The picture handed to the class is not the source.** It is the source
  placed on a larger canvas whose new area holds the fill (decision 2). The
  fill is derived from the source and the padding at run time, never
  stored.
- **The mask is not painted.** The server builds it from the canvas size,
  the padding and the band (decision 3), and stores it with
  `storage.store_mask`, so it becomes an ordinary stored mask with a hash,
  and all of the mask machinery applies unchanged: the `masks` table,
  `mask_holds`, the cell's check as it begins, the chunk keys, the embedded
  copy, bundles, reindex and the sweep.

So an outpaint's row is an inpaint's row (`refine`, `strength`, `mask`)
plus one column saying how the canvas was padded. Its size is the source's
plus the padding, which is the one place it breaks the inpaint's size rule.

Out of scope: a painted mask together with an outpaint (exclusive, below),
outpainting on the deferred families (it arrives with their inpaint phases,
since the gate is the same), `padding_mask_crop` (Z-Image's class has no
such parameter, as `plan-inpainting.md` recorded), an outpaint that crops
(a negative padding), and any import of outside pictures (U61 B stays
deferred).

## The installed classes, for an outpaint

Read from `pipeline_stable_diffusion_xl_inpaint.py` and
`pipeline_z_image_inpaint.py`:

- **Both take the canvas size from the request and the picture together.**
  `generate()` already hands `width` and `height` (the request's) and
  `image`; for an outpaint both are the padded canvas's, so nothing is
  resized. Z-Image's class rounds a size to `2 * (size // (vae_scale_factor
  * 2))`, i.e. to 16 px, and SDXL's to the VAE's 8: a canvas on the
  request's 16-pixel grid (`GenerateRequest._multiple_of_16`) is unchanged
  by either.
- **The starting strength is the class's signature default** - 0.9999 on
  SDXL, 1.0 on Z-Image - read with `inspect` (`refine.library_strength`,
  `schemas_generate.default_inpaint_strength`) as for an inpaint. SDXL's
  call sets `is_strength_max = strength == 1.0` and otherwise starts from
  `scheduler.add_noise(image_latents, noise, timestep)` at the first
  timestep the strength leaves; Z-Image's starts from
  `scheduler.scale_noise(image_latents, timestep, noise)`. So at the
  library default the new area starts from the fill noised to the top of the
  schedule. The new area is fill, so a high strength is what is expected
  there; how much the fill shows through at that strength or any lower one
  is the card's to say (open measurements).
- **The mask is binarised and sampled per latent cell**, which is why the
  backend hands the class `refine.pipeline_mask` (thresholded at > 0,
  widened to whole cells) and composites with the levels. Nothing about that
  changes for an outpaint; the geometry below makes the widening a no-op.

No catalogue author says anything about outpainting (the 2026-10-12 reading
of the model cards found no match for inpaint, outpaint or mask, and
`guides.py` has none). Anything the app says about how to word an outpaint
prompt or how wide a band should be would be `heuristic` or measured, and
none is proposed.

## The request

### The field

```python
class Outpaint(BaseModel):
    left: int = Field(default=0, ge=0, le=settings.refine_max_side)
    top: int = Field(default=0, ge=0, le=settings.refine_max_side)
    right: int = Field(default=0, ge=0, le=settings.refine_max_side)
    bottom: int = Field(default=0, ge=0, le=settings.refine_max_side)
    band: int = Field(default=OUTPAINT_BAND, ge=0, le=OUTPAINT_MAX_BAND)
```

`GenerateRequest.outpaint: Outpaint | None = None`, after `mask`, in
`schemas_generate.py` (re-exported by `schemas.py`). One nested object
rather than five top-level fields: the five numbers mean nothing apart, and
`applied()` drops them together, as it drops `lora_scale` with `lora`.

- **Each side is a multiple of 16**, refused with the value and its two
  neighbours by the same custom error the size uses (`multiple_of_16`,
  "50 is not a multiple of 16: 48 or 64"). A source's sides are already on
  that grid (every gallery image was made through the same validator), so
  the canvas is too; and the source then lands on whole latent cells for
  both families (8 px on SDXL, 16 px on Z-Image's 2x2 patches), which keeps
  `refine.pipeline_mask` from widening the repainted area past the band.
- **At least one side is more than 0** ("an outpaint extends at least one
  side"). A zero padding is an inpaint, and an all-zero object would record
  an outpaint that is none.
- **The band is a multiple of 8**, 0 to `OUTPAINT_MAX_BAND` (256). A
  multiple of 8 makes the band end on a latent-cell edge, so the class's
  repainted cells and the composite's blend cover the same pixels exactly.
  The band's default, `OUTPAINT_BAND = 32` (four cells), is this app's
  starting value, not a sourced or measured number, and the dialog says
  so; 0 is allowed and is a hard edge at the original's border.
- **Validators on `GenerateRequest`:** an outpaint without `refine` is
  refused, "an outpaint needs the image it extends", as a mask is ("a mask
  needs the image it covers"); an outpaint with a `mask` is refused, "an
  outpaint builds its own mask; send none with it" - the user mask and the
  outpaint are exclusive in every request a client sends. The server's own
  stamping (below) adds the derived hash with `model_copy`, which does not
  validate, after the last place a cell is built.
- **A canonical spelling** `Outpaint.key()` = `"L,T,R,B@band"` (for example
  `"400,0,400,0@32"`) and its inverse `Outpaint.parse_key(text)`: the column,
  the key element and the facts all read from it.

### The size: source plus padding, exactly

The request's `width` x `height` must equal the source's size plus the
padding: `width == source.width + left + right` and `height ==
source.height + top + bottom`. The page computes it, the CLI computes it,
and the server re-checks it. There is no rounding on the server: the page
rounds a typed side to the grid as it is committed (below), the validator
refuses a side off the grid, and the submit check refuses a size that does
not add up. Making the size implicit (derived from the source on the server)
was rejected: every guard that reads `request.width` / `height`
(`check_budget`, `_direct_renders_stay_in_the_presets`, the batch budget,
`completed_cells`' size scope, `run_id`'s `size`, the estimates) would have
to learn to look the source up first; an explicit size checked once keeps
them all as they are.

The guards then apply unchanged, because the request is a refine pass
(`refine` set): `refine_max_side` on each side through the field bounds, and
`check_budget`'s refine budget (`refine_max_pixels`) on the whole canvas.
The stored mask is the canvas's size, which is within the same budget the
mask upload route's pixel cap enforces.

### applied()

`GenerateRequest.applied()` stays the one normalisation, and the outpaint
follows the mask through it:

- **One predicate for "runs the inpaint class":** a property
  `GenerateRequest.inpainting` = `refine is not None and (mask is not None
  or outpaint is not None)`. Every site that today asks `request.mask is
  not None` to choose the inpaint class or its numbers asks it instead:
  `applied()` itself, `_profile_defaults` (the starting strength),
  `grid.duplicate_count`, `grid.denoised_steps`, `grid._cell_ms`,
  `submission.unusable_input`, `DiffusersBackend.start_sigma` and
  `generate()`, `GenerationBackend.check_refine_supported`'s wording, the
  More seeds and Re-render estimates (`api_more_seeds.py`,
  `api_generate.py`'s preview), and `tests/fakes.py`' copy of `applied()`.
  Sites that ask about *the stored mask file* (`jobs._cell_masks`, the cell's
  check as it begins, `storage.save_image`) keep asking `mask`, which an
  outpaint has by the time they run (stamping).
- With an outpaint, the source survives only where `inpaint_supported`, as
  with a mask; where the source goes, the outpaint goes with it (and the
  mask, and the strength back to its default). An outpaint without a source
  is not a setting and is dropped.
- References are dropped, as on an inpaint (SDXL's and Z-Image's classes
  take none).

### Submit checks

`submission.unusable_input` gains kind `outpaint`, after the capability
(`inpaint_unavailable`, whose words gain "which an outpaint runs through")
and the live source (`storage.reference_paths`), through a blocking
`submission.outpaint_unusable(request)` beside `mask_unusable`:

- the source's size plus the padding is the request's size, else "an
  outpaint of #N (WxH) by +L/+T/+R/+B is W'xH', not the WxH asked for";
- the band leaves a kept rectangle of at least 16 px each way on the
  source (the source's side minus the band on each padded end of it), else
  "a band of B px on both sides repaints the whole of #N; narrow it";
- a request that carries a mask beside an outpaint and reaches this far (a
  client that skipped the validator cannot; this is for a stamped cell
  re-checked) must carry the outpaint's own hash, else kind `outpaint`.

Swap tags and More seeds report the kind per image, as they report
`mask_missing` (`UNUSABLE_STATUS` gains `outpaint`). An outpaint never
reports `mask_missing`: its mask is a function of the request and is built
again at the submit, which also heals a row whose mask file was lost
(`submission.missing_masks` skips outpaint rows).

## The fill and the mask: exact algorithms

Both live in a new leaf module, `backend/outpaint.py` (numpy and Pillow,
nothing from the package at the top, as `refine` and `seeds` are leaves),
and both are pure functions of their arguments with integer arithmetic only,
so the same source and padding give the same bytes on every machine and
every numpy version. `tests/test_outpaint.py` pins the SHA-256 of each for a
fixed fixture, so any change to either fails a test rather than silently
making old outpaints unreproducible.

### The padded source (`outpaint.padded(source, spec)`)

`FILL = "edge-box3-r16"`, recorded in every outpaint's chunk.

1. `src` is the source decoded as RGB, `uint8`, shape `(sh, sw, 3)`.
2. **Edge stretch:** `stretched = np.pad(src, ((top, bottom), (left,
   right), (0, 0)), mode="edge")` - every new pixel takes the nearest
   source pixel (clamped coordinates); a corner takes the corner pixel.
3. **Blur:** three passes of an integer box blur of radius `r = 16`
   (window 33) along the width, then three along the height, over the
   whole stretched canvas. One pass along an axis: pad by `r` with
   `mode="edge"` on that axis, take a cumulative sum in `int64` with a
   leading zero, window sum `s[i] = c[i + 33] - c[i]`, and
   `out[i] = (s[i] + 16) // 33` (round half up), back to `uint8`. Three box
   passes approximate a Gaussian; the radius is fixed, not a function of
   the padding, so the fill name says the whole algorithm.
4. **The original back exactly:** `canvas[top:top+sh, left:left+sw] = src`.
   Only the new area holds blurred values; the blur reaches it from the
   source's own edge, so the new area starts in the colours the edge has.

The fill is never stored: it is derived from the source's pixels (already
in the reproduction set, by id and by `refine_source`'s pixel hash) and the
padding, both recorded. Storing it would be a second large file per
outpaint for no information. `refine.outpaint_source(image_id, spec, width,
height)` beside `refine.inpaint_source` loads the source through
`storage.reference_paths`, closes the handle, refuses a size that is not
source plus padding (the backstop behind the submit check, as
`inpaint_source` refuses another size) and returns `outpaint.padded(...)`.

### The mask (`outpaint.mask(width, height, spec)`)

Mode `L`, the canvas's size, white repainted:

- **Outside the source's rectangle** (the new border and its corners): 255.
- **Inside it:** for each *padded* side, the distance `d` of the pixel from
  that edge of the source, counted in pixels from 0 at the source's first
  row or column (`x - left` for the left side, `(left + sw - 1) - x` for the
  right, and the same for top and bottom). A side with no padding has no
  band. The level is `0` where `d >= band` for every padded side, else
  `max` over the padded sides of `(255 * (band - d) + band // 2) // band`:
  a linear ramp from 255 at the original's edge to 0 at the band's inner
  edge. With `band = 0` every source pixel is 0 (a hard edge).

So the mask is a function of the canvas size and the spec alone (the
source's size is the canvas's minus the padding). Its kept rectangle is the
source inset by the band on its padded sides; the submit check above holds
it to at least 16 px each way. Because each side and the band are on the
latent grid, `refine.pipeline_mask` of this mask is its own threshold - the
class repaints exactly the band and the border - and `refine.composite`
blends the band by the ramp's levels and keeps every pixel inside the kept
rectangle as the source's. Tested.

### Stamping: the mask becomes a stored mask

At the submit, after the producers have built their cells and before the
job is queued, `outpaint.stamp(cells)` builds each distinct outpaint's mask
once (one per canvas size and spec, so normally one per job), stores it with
`storage.store_mask` (idempotent, content addressed; it takes the gallery
reservation itself and restarts the settle window) and answers the cells
with `mask` set to its hash by `model_copy`. It is called in exactly two
places: `JobManager.submit_cells`, just before `_cell_masks` and
`storage.hold_masks`, so every producer (Generate, a grid, Re-render, Swap
tags, More seeds) gets it once; and `generate.py`'s
deliberate copy of the loop, held equal by `tests/test_cli.py`. An
`OSError` while storing is a worded 503 ("the outpaint's mask could not be
written ..."), as `POST /api/masks` says one.

From there an outpaint's cell is an inpaint's cell with an `outpaint`
beside it: held in `mask_holds` while queued, its mask file checked as the
cell begins (`jobs`, `storage.mask_path`), read by `refine.mask_image`,
recorded by `save_image`, embedded in `local-diffusion-mask`, carried by
bundles and reindex, spared by `gallery_delete.sweep_masks` while any row
names it. Nothing in that machinery learns the word outpaint.

Rejected alternative: deriving the mask in the backend at run time and
storing it at save. That puts a second mask builder site in the backend,
leaves the mask outside `mask_holds` while the job is queued, and makes the
save path store a file a moment before the row that names it. Stamping
stores it once per job, through the one store, before anything reads it.

## The backend

In `DiffusersBackend.generate()`, the inpainting branch (`inpainting =
refining and request.mask is not None` becomes `request.inpainting`) picks
its picture by the request:

```python
if request.outpaint is not None:
    kwargs["image"] = refine.outpaint_source(request.refine, request.outpaint, request.width, request.height)
else:
    kwargs["image"] = refine.inpaint_source(request.refine, request.width, request.height)
painted = refine.mask_image(request.mask, request.width, request.height)
```

Everything after is unchanged: `pipeline_mask` for the class, `strength`,
one call per image through `_refine`, and the composite over
`kwargs["image"]` - the padded canvas - with the painted levels. So every
pixel inside the kept rectangle is the source's exactly, the band blends by
the ramp, and the border is the class's output. The fill itself survives
nowhere in the result: every pixel of the new area is under 255. The
`content_hash` is the composite's, as for an inpaint.

- `describe()` and `/api/status` gain nothing: `inpaint_supported` /
  `inpaint_settled` answer for an outpaint too, and the page reads them.
- `start_sigma` is None for an outpaint, as for an inpaint (no
  `START_RECIPES` entry for an inpaint class yet).
- The live preview reads the callback's latents as it does for an inpaint:
  Z-Image keeps its preview, SDXL shows none.
- The fakes: `tests/fakes.py` and `tools/fake_server.py` repaint the masked
  pixels with a flat colour by seed through the real composite; they gain
  the same picture choice (`refine.outpaint_source` for an outpaint), so a
  test and a smoke can read the kept rectangle against the source and the
  border against the fill colour.

## The reproduction set

### The row

- `images.outpaint TEXT`: `Outpaint.key()`, NULL on every other row, through
  a `_migrate` entry (with an index `idx_images_outpaint`, as `mask` has
  one), `_COLUMNS`, both explicit dicts (`save_image`, `storage.index_file`)
  and `_row_to_record`, which reads it unguarded through
  `Outpaint.parse_key` (only validated values are ever written).
  `ImageRecord.outpaint: Outpaint | None`.
- `refine`, `strength` and `mask` are set as on an inpaint: `refine` is the
  unpadded source, `mask` the stamped hash. The row's `width` x `height` is
  the canvas.

### The chunk and the A1111 block

- The `local-diffusion` chunk gains `outpaint` (`{"left", "top", "right",
  "bottom", "band"}`) and `outpaint_fill` (`outpaint.FILL`), null together on
  every other row. `mask`, `mask_source` (`storage.mask_provenance`, at the
  canvas's size) and `mask_blend: "overlay"` are written as for any inpaint,
  and the mask's PNG rides in `local-diffusion-mask` as for any inpaint, so
  the PNG alone carries its mask even though it could be rebuilt.
  `refine_source` is the unpadded source's provenance (its pixel hash and
  its own size), which is what `--from-image`'s `_check_carried_source`
  compares.
- The A1111 block gains `, Outpaint: L/T/R/B, Outpaint band: B` after
  `Inpaint mask` (slashes, since an A1111 value holds no comma).

### CellKey and run_id: folded into the mask element

The outpaint goes **into the existing fifth extension**, not a sixth:
`grid.mask_key(mask, outpaint=None)` answers `f"outpaint:{outpaint}"` (the
canonical spelling) where there is an outpaint, the hash where there is only
a mask, and None otherwise. Built from a request in `grid.cell_key` and from
the `mask` and `outpaint` columns in `storage.completed_cells`.
`_KEY_EXTENSIONS` stays 5.

- **Why the spec and not the hash:** the hash is a function of the canvas
  size and the spec, and the size is already in `completed_cells`' scope and
  `run_id`'s definition, so the spec names the same picture; and the
  page's preview (no stamped hash yet) and the stamped cell then key the
  same, so a grid preview's "N already rendered" and the job agree without
  storing anything on a keystroke.
- **Why fold it in:** an outpaint and a painted mask never meet in one
  request, and both answer the one question "which region is repainted",
  which is what the element means. A sixth element would add a None to the
  end of every key and shift `duplicate_count`'s slicing again for no
  information.
- **run_id unchanged for every existing run**, inpaints included: no
  existing row has an outpaint, so every element is what it was. Pinned in
  `tests/test_conditioning.py` beside the existing pins, with a new pin for
  an outpaint run.
- **`duplicate_count`** keeps every element after the refine one
  (`(*key[:8], f"{refine}@step:{start}", *key[9:])`), so two paddings or two
  bands over one source are two pictures; its `start_index(...,
  inpaint=...)` reads `request.inpainting`.
- **A rule:** `outpaint.padded` and `outpaint.mask` are frozen. A change to
  either is a new `FILL` name, a recorded column for it and a key element
  that carries it; the pinned hashes in `tests/test_outpaint.py` are what
  make that impossible to forget.

### Bundles, reindex, the bin, clear

- **Bundles:** nothing new for the mask - it is a stored mask, carried as
  `masks/<hash>.png` with the `masks` table, verified by
  `bundle_images._inspect_mask`, merged by `bundles._merge_masks`. The
  `outpaint` column travels in the database; the merge's id remap touches
  only `refine`, since the spec holds no id. Where neither the bundle nor the
  result's embedded copy (`bundles._embedded_member`) has an outpaint's mask,
  the import rebuilds it from the spec and the row's size and installs it
  only if its hash is the row's; it counts `outpaint_masks_rebuilt`, and a
  mismatch keeps the row's hash and counts in `masks_missing` as today. The
  manifest stays `version: 1`. An older build imports an outpaint as an
  inpaint row with an unknown column (kept through the unlisted-column
  fallback); its Re-render then refuses it, since its mask and source are
  not one size (`submission.mask_unusable`), which is the safe failure.
- **Reindex:** `storage.index_file` reads `outpaint` from the chunk where
  `outpaint_fill` is this build's `FILL` (validated through `Outpaint`), and
  leaves it NULL otherwise, counted `unknown_fill`; mask recovery is as for
  any inpaint, plus the same rebuild from the spec where no file and no
  embedded copy is found.
- **The bin, deletes, `clear`, the sweep:** unchanged. An outpaint's mask is
  named by its row, so `sweep_masks` spares it while the row lives, binned
  or not, and takes it with the row's last permanent removal.

## Lineage and gallery actions

`grid.cell_from_record` carries `outpaint` with the source and strength and
sets `mask` to None where the record has an outpaint (the stamp rebuilds and
re-stores it, and the validator refuses the two together): `"outpaint":
record.outpaint if record.refine is not None else None`, `"mask":
record.mask if record.refine is not None and record.outpaint is None else
None`.

- **Re-render** runs the same padding over the same source at the record's
  own size. At another size it is refused for the whole selection as an
  inpaint is (`submission.resized_inpaints` already catches it, since the
  row has a mask); `resized_inpaint_detail` says "#M is an outpaint of #N at
  WxH" for it, and "an outpaint runs at its source's size plus its
  padding". The way out stays "Add inpaint:no to the search", which leaves
  outpaints out too.
- **More seeds** runs the next seeds over the same padding and band.
- **Swap tags** rewrites the prompt over the same padding.
- **Mass refine** makes the record itself the new source and drops both the
  mask and the outpaint (`refine_batch` passes `mask=None, outpaint=None`);
  its "Skip refine passes" line (`refine_batch.py`) says "an outpaint of #N".
- **Reuse** on an outpainted record opens the Outpaint dialog on its source
  with the four sides, the band, the strength, the seed and the prompt
  (`reuse.js`, beside `reuseInpaint`; an outpaint is checked first, since it
  also has a mask). A source that is gone or binned is said as the Inpaint
  dialog says it.
- **Load settings from PNG** carries no source, so no mask and no outpaint:
  `applied()` drops it with the source, and the panel takes the PNG's size,
  prompt and settings, as it does for an inpaint's PNG. `png_settings` reads
  the two new chunk keys as known and ignores them.
- **`--from-image`** carries the outpaint with its source where
  `_check_carried_source` passes, re-derives the mask and compares its hash
  with the chunk's `mask`; a mismatch is a refusal ("this build fills an
  outpaint differently from the one that made this image").
- **Search:** `outpaint:yes|no` (`outpaint IS NOT NULL` / `IS NULL`, through
  an `outpaint_clause` beside `search.inpaint_clause`). `inpaint:yes` still
  means "a mask on the row" and so includes outpaints - mechanically each is
  an inpaint, and both share the size rule Re-render refuses by - and
  `inpaint:yes outpaint:no` is the painted ones. `stats --by outpaint` is
  optional.
- **Words:** cards and the job line say "outpaint of #N"; the facts list
  says "outpaint of #N +L/+T/+R/+B @ 0.9999, band 32 px" (the strength as
  recorded). The job snapshot (`Job.one_source`) gains the outpaint beside
  the mask so the page can name a queued job.
- **The lightbox's Against source** compares at the record's own geometry:
  for an outpaint the source is drawn at `(left, top)` at the result's scale
  rather than scaled to cover it (`lightbox-compare.js`), so the divider
  shows the original against the kept rectangle pixel for pixel.

## Estimates

`gallery_reports.refine_cost_ms` already splits on `mask IS NULL` and groups
by denoised steps and size, so an outpaint is priced from inpaint history at
its canvas size, outpaints included. No new split: the call is the same
class over a picture of that size. An outpaint at a size no inpaint has run
at has no estimate, as an inpaint at a new size has none.

## The UI

### An Outpaint dialog of its own

The recommendation is **a modal dialog of its own** - `web/outpaint.js`,
`web/styles/outpaint.css`, a `<dialog id="outpaint-dialog">` in
`index.html` - not an Extend tab or mode of the Inpaint dialog:

- The Inpaint dialog's stage is the source at its pixel size under a
  focusable mask canvas, and its tools row (brush, eraser, size, feather,
  invert, clear, undo, redo, show mask) is all about painting. An outpaint
  has no painted mask; in a mode every one of those would be held, and the
  stage would swap for a frame of another shape, moving every control under
  it when the tab changes - against "never moves a control under the
  pointer" - and greying half the dialog, the reason the user gave inpaint
  a dialog apart from Refine's.
- `inpaint.js`'s state is the mask's (upload epochs, the keep timer, kept
  strokes per image, `inpaintMaskState`); an outpaint mode would need an
  exemption at each step of it.
- What is the same is shared, not copied: the prompt check's lines
  (`refine-prompt.js`), the strength box's rules (one value, or a list or
  range that is a sheet), Versions, Seed, the server's preview and how a
  refusal is said. `outpaint.js` loads after `inpaint.js` and calls its
  top-level helpers where they are the same - the start strength and its
  note (`inpaintStartStrength`, `inpaintStrengthNote`,
  `takeLibraryStrength`), the unavailable wording (`inpaintUnavailableText`)
  - generalised in place where a name says "inpaint" but means "the inpaint
  class". Top-level bindings stay top-level; `docs/rules.md`'s Layout and
  `docs/source-audit.md` gain the files.

It is a dialog, never a panel mode, and the panel's `collect()` still never
sends a source, a mask or an outpaint.

### Entry points

- The lightbox's **Outpaint…** in the row with Refine and Inpaint…, closing
  the lightbox and given back on a close that runs nothing, as both do; the
  row must not wrap at 1400 px (checked in the smoke).
- A viewer result's tool, beside its inpaint tool.
- Reuse on an outpainted record.
- Cards get no new glyph in this plan (a card's tool row already carries
  the pass and the inpaint); that is an open question.

Where the checkpoint has no inpaint class (or its family is deferred), each
entry point is `aria-disabled` with the reason as its title and a press
says why, in the Inpaint dialog's guess / settled wording
(`syncInpaintCapability`): "Krea 2 Turbo has no inpaint pipeline in the
installed diffusers, which an outpaint runs through", or "... not enabled
yet" for a deferred family.

### The controls

At 1400x900, top to bottom, every row a fixed height; the dialog is anchored
at a fixed top and grows downward as the Inpaint dialog does, never
`overflow-y: auto` on the panel, never scrolling by itself:

- **The frame**, in a box of fixed size whatever the canvas's shape: the new
  canvas letterboxed into it, its new area hatched (the page draws no fill
  of its own - the fill is the server's), the source's picture at its place
  on the canvas, the band's inner edge dashed, and a line under it: "1824 x
  1024 - #12 1024 x 1024 at +400/+0/+400/+0". The four side boxes sit
  around the frame in a cross (Top above, Left and Right beside, Bottom
  below), at fixed positions; the frame's content scales inside its box, so
  nothing moves when a number changes.
- **Side boxes:** `type="number"`, `step="16"`, `min="0"`. A typed value is
  rounded to the nearest multiple of 16 (half up, as `refine.scaled`
  writes it) when it is committed (change, Enter, blur), never while
  typing, and the fixed-height status line says so ("left 50 -> 48, a
  multiple of 16"); arrows step by 16. The rounded numbers are what the
  preview and the submit send; the server re-checks them.
- **Presets**, one fixed row: **+25% all round**, **to 16:9**, **to 2:3**,
  and a **To shape…** select holding every aspect of `sizes.SIZES` (the
  thirteen the panel's preset strip uses). A preset **replaces** the four
  boxes, never adds to them, and leaves the band alone. The server decides
  the numbers (`GET /api/images/{id}/outpaint-presets`, model-free, through
  `outpaint.presets(width, height)`, which the CLI's `--outpaint 16:9` uses
  too), so the page holds no copy of the arithmetic:
  - **to a:b** extends only the axis that is short of the ratio and never
    crops: for a source narrower than a:b, the canvas's width is the
    smallest multiple of 16 at or above `height * a / b` (integer
    arithmetic, `-(-height * a // (b * 16)) * 16`) and the height stays;
    otherwise the same for the height. The padding on that axis is split
    evenly, the first side taking `pad // 32 * 16` and the second the rest
    (the odd 16 px to the right or bottom). A source already at the ratio
    within the grid gets no padding, and the preset is held: "#12 is
    already 16:9".
  - **+25% all round** grows each axis by a quarter of its side, rounded to
    16 half up with at least 32 so each end gets at least 16
    (`max(32, int(side * 0.25 / 16 + 0.5) * 16)`), split evenly the same
    way: a 1024 x 1024 source becomes 1280 x 1280 at +128 on every side.
  - A preset whose canvas would pass `refine_max_side` or
    `refine_max_pixels` is held with the size it would be and the budget it
    passes.
- **Band:** a number box, `step="8"`, 0-256, rounded to 8 as the sides are,
  with its note "this app's starting value; nothing measured says what
  width joins best".
- Then the shared rows: the prompt (starting from the source's, said as
  the Inpaint dialog says it), its Check lines, Strength (starting at the
  library's default, said as the library's), Versions, Seed, the preview
  line and **Outpaint**.

### Held states

**Outpaint** is `aria-disabled` with the reason as its title, and a press
says it, while: no side is extended ("extend at least one side"); the canvas
passes the side or pixel budget (the size and the budget); the band leaves
no kept rectangle; the preview refuses (its words); the source is gone or
binned; the checkpoint has no inpaint class; a submit is in flight. A
refusal replaces the preview line at its height, as the Inpaint dialog's
does. A close that ran nothing keeps the four sides and the band in page
memory for the next open of the same image, as the Inpaint dialog keeps its
strokes, never in `localStorage`.

## The CLI

`generate.py --refine ID --outpaint L,T,R,B [--band PX]`:

- `--outpaint` takes four multiples of 16 or a preset name (`16:9` or any
  `sizes.SIZES` aspect, or `+25%`), resolved by `outpaint.presets`; a side
  off the grid is refused with its neighbours, never rounded silently.
- `--band` defaults to `OUTPAINT_BAND`; it needs `--outpaint`.
- The size is the source's plus the padding; `--width`, `--height`,
  `--aspect`, `--mp` and `--scale` are refused beside `--outpaint`, as is
  `--mask` ("an outpaint builds its own mask").
- The mask is stamped through `outpaint.stamp` in the CLI's copy of the
  submit loop; `--dry-run` prints the canvas and the padding and stores
  nothing.
- `--from-image` carries `outpaint` from the chunk as above.

## Grid and comparison

The padding and the band are fixed across a job, as a mask is; neither is an
axis. Everything that is an axis over an inpaint is one over an outpaint:
the Strength box's list or range (`GridRequest.strength_axis`), the seed
axis (Versions), a prompt variable. Each cell's key carries the outpaint in
the mask element, so resume, the duplicate count and `run_id` treat two
paddings as two runs. The contact sheet needs nothing new. The panel's
prompt comparison never sends a source, so it never outpaints.

## Tests and smokes

Python (`uv run --group dev pytest`, no GPU):

- `tests/test_outpaint.py`: the field's validation (off-grid sides with
  their neighbours, no side extended, band off its grid or past 256, an
  outpaint without a source, with a mask); `presets` for each aspect against
  hand-worked numbers (1024 x 1024 to 16:9 is +400/+0/+400/+0 at 1824 x 1024;
  to 2:3 is +0/+256/+0/+256 at 1024 x 1536; +25% is +128 all round) and the
  held cases; `mask` levels at the border, at `d = 0`, `band - 1` and
  `band`, in corners where two bands meet, with `band = 0`, and with
  unpadded sides having no band; `padded` keeping every source pixel and
  stretching the edge; both pinned by SHA-256 on a fixture; `pipeline_mask`
  of an outpaint mask equal to its own threshold; `refine.composite` over
  the padded canvas keeping the kept rectangle exactly.
- `tests/test_api_outpaint.py`: the submit's refusals (size not source plus
  padding, past the budget, band too wide, binned or deleted source, a
  checkpoint without the class, a mask beside an outpaint); stamping (each
  queued cell carries the derived hash, the mask is stored and held in
  `mask_holds`, a lost mask file healed by the next submit); the presets
  route; the grid preview with a strength axis; Re-render, More seeds and
  Swap tags carrying the outpaint; Mass refine dropping it; Re-render at
  another size refused naming "an outpaint of #N".
- `tests/test_conditioning.py`: `mask_key` with an outpaint; every existing
  `run_id` pin unchanged; a pin for an outpaint run; `duplicate_count` with
  two paddings and with two bands.
- Storage: `_migrate` adds `outpaint`; `save_image` and `index_file` write
  and read the chunk keys and the A1111 line; `completed_cells` keys an
  outpaint row on its spec; `unknown_fill`.
- Bundles and reindex: the column travels; an outpaint's mask rebuilt from
  the spec where nothing carries it, refused where the hash differs.
- `tests/test_cli.py`: `--outpaint` (sides and presets), `--band`, the
  refusals, the equal loops with the stamp in both.
- `tests/test_inpaint_ui.py` or a sibling: the page's held reasons against
  the status keys.

Smokes, at 1400x900:

- **`smoke-outpaint`** (Animagine fake, port 7907, fresh scratch `--data`):
  open from the lightbox; a typed 50 committed as 48 with its line; the
  presets' exact numbers on a 1024 x 1024 planted image; a preset held
  where the image already has the shape; the frame's source placement
  against the boxes; Outpaint held with each reason; no self-scroll
  (`scrollTop` unchanged through edits, presets and a refusal) and every
  control's box unchanged under the pointer as numbers change; the
  lightbox row not wrapping; submit; the result says "outpaint of #N
  +400/+0/+400/+0" and its kept rectangle equals the source's pixels (the
  fake); Reuse reopens with the sides and band; a strength list opens a
  sheet.
- **`smoke-inpaint-unavailable`** (default Krea 2 Turbo fake, port 7906)
  gains the outpaint entry points: `aria-disabled` with the reason, and a
  press says it.
- Later: a wide-tier case in `smoke-wide`.

When phase 1 lands, `docs/rules.md` gains the rules (the field, the size
rule's exception, stamping, the frozen fill and mask, the folded key
element), its Tests section the smoke and port, `CLAUDE.md` / `AGENTS.md`
the port row, `docs/design-notes.md` an "Outpainting" section, and
`docs/verified.md` an "Outpaint" section listing the measurements below as
not yet measured.

## Phases

Effort on the review's scale: S a day or less, M a few days, L a project.

**Phase 1 - backend and reproduction set (M).** `backend/outpaint.py` (fill,
mask, presets, stamp); the field, the validators, `inpainting`, `applied()`,
the submit check; `refine.outpaint_source` and the backend branch; the
column, the chunk keys, the A1111 line; `mask_key`; `cell_from_record`,
Re-render's wording, Mass refine's drop; bundles' and reindex's rebuild; the
presets route; search; the CLI; the fakes; every Python test above. Smaller
than inpainting's phase 1, because the mask machinery exists and is reused
whole.

**Phase 2 - the dialog (M).** `outpaint.js`, the CSS, the entry points,
Reuse, the facts and card wording, Against source at the offset, the
smokes, the docs listed above.

**Phase 3 - with the measurements (S).** Once the card has said anything:
a band default that is measured rather than this app's, and anything the
inpaint phases change for every inpaint (a `START_RECIPES` line, the
preview through the blend). The deferred families get outpainting the day
their inpaint is enabled, since the gate is `inpaint_supported`; their
memory at outpainted sizes is part of their phase.

**Later, if wanted:** the server's fill shown in the frame (a model-free
`GET` answering the padded canvas as a JPEG, so the page still draws
nothing of its own); a card tool; `padding_mask_crop` for large canvases on
SDXL; an outpaint over a painted mask together.

## Open measurements (the card's to say)

1. Peak memory and time per step at outpainted sizes for WAI and one other
   SDXL checkpoint, and Z-Image Turbo and Z-Anime: an outpaint's canvas is
   larger than its source (a 1 MP source to 16:9 is about 1.8 MP, +25% about
   1.6 MP), so it meets the card's limits sooner than an inpaint of the same
   source. One size per run, stopping at the first that fills the card.
2. Determinism: the same outpaint twice, batch image *i* against a solo run
   of its seed, cancel-then-rerun, the text-to-image hash afterwards.
3. What the fill contributes at the library default (0.9999 SDXL, 1.0
   Z-Image) and at lower strengths - whether the stretched edge shows.
4. The seam at band widths 0, 32, 64 and 128 px per family, against the
   VAE round trip at the original's edge.
5. A starting band width, if one is measured to join better than 32.
6. The embedded mask's size in an outpaint's PNG (mostly white, unlike an
   inpaint's mostly black one).
7. The fill's and the mask's build time on the CPU at the refine guard's
   largest canvas (both are numpy passes over the whole canvas).

## Risks

- **A frozen algorithm.** The key element names an outpaint by its spec, not
  its hash, so a changed fill or mask would quietly make a resume count old
  pictures as done. The pinned hashes, the recorded `outpaint_fill`, and the
  rule that a change is a new name are the guard.
- **Revalidation of a stamped cell.** A stamped cell carries a mask and an
  outpaint, which the validator refuses together. Nothing after
  `submit_cells` rebuilds a request from fields today; a test pins that a
  queued outpaint cell keeps both, and the rule names the hazard for any
  later code that would.
- **The size rule's exception.** Every place that checks "an inpaint runs at
  its source's size" - `submission.mask_unusable`, `refine.inpaint_source`,
  `resized_inpaints`, `generate.py`'s `_mask_unusable` - has to branch on
  the outpaint, or an outpaint is refused as a misaligned inpaint, or worse,
  an inpaint runs at a padded size. The tests above check each.
- **Quality.** No catalogue author documents outpainting, and the fill is
  this app's own choice; results may show seams or repeat the edge. The app
  can say only what was measured.
- **Memory.** A canvas larger than any direct render of the checkpoint's
  presets is reachable within the refine budget; only the card can say
  where each family stops.
- **Older builds.** An older build imports an outpaint as an inpaint whose
  mask and source differ in size, and refuses to re-run it; it cannot
  reproduce it. This is the intended failure, and the import result should
  say outpaints were imported as records only there.

## Open questions for the user

1. A card tool for Outpaint beside the pass and the inpaint, or only the
   lightbox, the viewer and Reuse (recommended for now)?
2. The band's starting value: 32 px (this app's, unmeasured) until the card
   says otherwise?
3. "+25% all round" as written here (each axis a quarter larger, split
   evenly), or a quarter of each side added on every side (each axis half
   larger)?
4. The server's fill shown in the frame preview in phase 2, or later?
