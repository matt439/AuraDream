# Plan: outpainting

**Status, 2026-10-13: a plan; nothing is built.** Written on
`feat-outpainting`, branched from main at b3bc975 right after inpainting was
merged, and revised the same day against an eighteen-point review of the
first draft (each point and what became of it is listed at the end, under
"Review of 2026-10-13"). It is grounded in the inpainting code as built (the
functions are named where it matters) rather than in
`docs/plan-inpainting.md`'s design, which the build departed from in places.
The user has already decided:

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
  fill is derived from the source, the padding and the fill's name at run
  time, never stored.
- **The mask is not painted.** The server builds it from the canvas size,
  the padding and the band (decision 3), and stores it with
  `storage.store_mask`, so it becomes an ordinary stored mask with a hash,
  and the mask machinery applies: the `masks` table, `mask_holds`, the
  cell's check as it begins, the chunk keys, the embedded copy, bundles,
  reindex and the sweep.

So an outpaint's row is an inpaint's row (`refine`, `strength`, `mask`)
plus one column naming the fill and how the canvas was padded. Its size is
the source's plus the padding, which is the one place it breaks the
inpaint's size rule. **Invariant, held at every carry site below:** an
outpaint is never on a row without both `refine` and `mask`; wherever
either is dropped, the outpaint is dropped with it.

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
  `schemas_generate.default_inpaint_strength`) as for an inpaint. What each
  class does with it, as read:
  - **SDXL:** `get_timesteps` takes `init_timestep = int(steps * 0.9999)`,
    which is `steps - 1` for any step count the app allows, so `t_start = 1`
    and the call runs from the *second* timestep of the schedule. Its
    `is_strength_max = strength == 1.0` is False at 0.9999, so the latents
    start from `scheduler.add_noise(image_latents, noise, latent_timestep)`
    at that second timestep - the fill, noised to that level - not from pure
    noise.
  - **Z-Image:** at 1.0 its call starts from `scheduler.scale_noise(
    image_latents, timestep, noise)` at the first timestep of the schedule.

  What that leaves of the fill in either family, at the default or any
  lower strength, is the card's to say (open measurement 3); this plan says
  nothing more about it. The new area is fill, so a high strength is what
  is expected there.
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

### Two types: the stored spec and the request's

A stored row must parse whatever the server's settings are now: a row
written under one `LOCAL_DIFFUSION_REFINE_MAX_SIDE` is read under another, or
by the WSL build with its own settings, and `_row_to_record` reads unguarded
- a stored type bounded by a setting would make every gallery read fail
once the setting is lowered. So there are two types in
`schemas_generate.py` (re-exported by `schemas.py`):

```python
class OutpaintSpec(BaseModel):            # stored: ImageRecord.outpaint
    model_config = ConfigDict(frozen=True)
    fill: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,31}$")
    left: int = Field(ge=0)
    top: int = Field(ge=0)
    right: int = Field(ge=0)
    bottom: int = Field(ge=0)
    band: int = Field(ge=0)
    # validators: each side a multiple of 16, the band a multiple of 8

class Outpaint(OutpaintSpec):             # the request's
    fill: str = Field(default=outpaint.FILL, pattern=...)
    band: int = Field(default=outpaint.BAND, ge=0, le=outpaint.MAX_BAND)
    # validator: at least one side above 0
```

- **The stored type has plain ints and grid checks only**: no bound read
  from settings, nothing that a setting change could make a stored row
  fail. Bundles refuse a malformed one before it is ever written
  (`_validate_row`, below).
- **The limits live in the request**, and only there. No side carries a
  `le=settings.refine_max_side`: the canvas is `width` x `height`, and
  `check_budget` already holds a refine pass's `width`, `height` and pixels
  to `refine_max_side` / `refine_max_pixels`, which bounds every side
  through the canvas. The band's `le=MAX_BAND` (256) is a constant, not a
  setting.
- `GenerateRequest.outpaint: Outpaint | None = None`, after `mask`. One
  nested object rather than six top-level fields: the numbers mean nothing
  apart, and `applied()` drops them together, as it drops `lora_scale` with
  `lora`. Frozen, so it is hashable and can sit in a check cache's tuple.
- **Each side is a multiple of 16**, refused with the value and its two
  neighbours by the same custom error the size uses (`multiple_of_16`,
  "50 is not a multiple of 16: 48 or 64"). The source then lands on whole
  latent cells for both families (8 px on SDXL, 16 px on Z-Image's 2x2
  patches), which keeps `refine.pipeline_mask` from widening the repainted
  area past the band - provided the source is on the grid too, which is not
  guaranteed (below, "An off-grid source").
- **At least one side is more than 0** ("an outpaint extends at least one
  side"), on the request. A zero padding is an inpaint, and an all-zero
  object would record an outpaint that is none.
- **The band is a multiple of 8**, 0 to `MAX_BAND`. A multiple of 8 makes
  the band end on a latent-cell edge, so the class's repainted cells and
  the composite's blend cover the same pixels exactly. The default,
  `BAND = 32` (four cells), is this app's starting value, not a sourced or
  measured number, and the dialog says so; 0 is allowed and is a hard edge
  at the original's border.
- **The fill is part of the spec.** `fill` names the algorithm that made
  the new area; the request type accepts any well-formed name, and the
  submit check refuses one this build does not have, by name ("this build
  has no outpaint fill 'edge-box5-r8'; it fills with edge-box3-r16"), rather
  than any later step nulling it. A client never needs to send it.
- **Validators on `GenerateRequest`:** an outpaint without `refine` is
  refused, "an outpaint needs the image it extends", as a mask is ("a mask
  needs the image it covers"); an outpaint with a `mask` is refused, "an
  outpaint builds its own mask; send none with it" - the user mask and the
  outpaint are exclusive in every request a client sends. The server's own
  stamping (below) adds the derived hash with `model_copy`, which does not
  validate.
- **A canonical spelling** `OutpaintSpec.key()` =
  `"fill:L,T,R,B@band"` (for example `"edge-box3-r16:400,0,400,0@32"`) and
  its inverse `OutpaintSpec.parse_key(text)`: the column, the key element,
  the facts and the CLI's `--outpaint` echo all read from it. Because the
  fill is in the key, the column, the `CellKey` element and `run_id` all
  carry it, and two fills over one padding are two runs.

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
(`refine` set): `check_budget`'s refine limits (`refine_max_side` on each
side, `refine_max_pixels` on the whole canvas). The stored mask is the
canvas's size, which is within the same budget the mask upload route's
pixel cap enforces.

### An off-grid source

Not every gallery image is on the 16-px grid: rows from a reindex or a
bundle take the PNG's decoded size, whatever it is. An off-grid source makes
every canvas over it off-grid (the sides are multiples of 16), and the width
validator would then refuse "1816 is not a multiple of 16" about a size the
user never typed. So the source is named before that can happen:

- `outpaint.source_off_grid(width, height)` answers the words "#N is
  1016 x 1016, off the 16-px grid, so no outpaint of it is either".
- `GET /api/images/{id}/outpaint-presets` (below) answers it as
  `off_grid` with those words, and the dialog holds **Outpaint** with them
  from the moment it opens; the CLI asks it before building its request.
- `submission.outpaint_unusable` asks it too, first, for a request that
  reaches the submit check. Over HTTP that is only a client that skipped
  the presets route with a hand-built size; such a request meets the width
  validator first, and that message stays the backstop. A gallery action
  never meets either: an outpaint row is only ever written on-grid.

### applied()

`GenerateRequest.applied()` stays the one normalisation, and the outpaint
follows the mask through it:

- **One predicate for "runs the inpaint class":** a property
  `GenerateRequest.inpainting` = `refine is not None and (mask is not None
  or outpaint is not None)`. Every site that today asks `request.mask is
  not None` *to choose the inpaint class or its numbers or its words* asks
  it instead: `applied()` itself, `_profile_defaults` (the starting
  strength), `grid.duplicate_count`, `grid.denoised_steps`,
  `grid._cell_ms`, `DiffusersBackend.start_sigma` and `generate()`,
  `GenerationBackend.check_refine_supported` (base.py:119, whose branch on
  `mask` would otherwise give an unstamped outpaint the refine pass's
  words), the More seeds and Re-render estimates (`api_more_seeds.py`,
  `api_generate.py`'s preview), and `tests/fakes.py`' copy of `applied()`.
- Sites that ask about *the stored mask file* keep asking `mask is not
  None`: `jobs._cell_masks`, the cell's check as it begins,
  `storage.save_image`, and the file checks in `submission.unusable_input`
  (below). An outpaint has its mask by the time the first three run
  (stamping) and none yet at the fourth, which is right: there is no file
  to check.
- With an outpaint, the source survives only where `inpaint_supported`, as
  with a mask; where the source goes, the outpaint goes with it (and the
  mask, and the strength back to its default). An outpaint without a source
  is not a setting and is dropped.
- References are dropped, as on an inpaint (SDXL's and Z-Image's classes
  take none).

### Submit checks

`submission.unusable_input` (lines 46-62) today uses one `inpainting =
request.mask is not None` both to pick the class's question and to run
`storage.mask_path(request.mask)` and `mask_unusable`. Switching that one
flag to the new predicate would call `mask_path(None)`. It is split:

- `request.inpainting` picks the capability question (`inpaint_unavailable`
  vs `refine_unavailable`) and its kind and words;
  `inpaint_unavailable`'s words gain ", which an outpaint runs through"
  where the request has an outpaint;
- `request.mask is not None` gates `mask_path` (kind `mask_missing`) and
  `mask_unusable` (kind `mask`), unchanged;
- `request.outpaint is not None` gates a new blocking
  `submission.outpaint_unusable(request)` beside `mask_unusable`, kind
  `outpaint`, after the live source (`storage.reference_paths`):
  - the source is on the grid (above);
  - the fill is one this build has (`outpaint.FILLS`), by name;
  - the source's size plus the padding is the request's size, else "an
    outpaint of #N (WxH) by +L/+T/+R/+B is W'xH', not the WxH asked for";
  - the band leaves a kept rectangle of at least 16 px each way on the
    source (the source's side minus the band on each padded end of it),
    else "a band of B px on both sides repaints the whole of #N; narrow it".

The first draft's third check (a stamped cell, mask and outpaint together,
re-checked here) is dropped: nothing calls `unusable_input` after the stamp,
so it could never run. The hazard it named is the "Revalidation of a
stamped cell" risk below.

**The status values and their carry sites.** `UNUSABLE_STATUS` gains
`"outpaint": "outpaint"`, so Swap tags and More seeds report the kind per
image as they report `mask_missing`. The value has to be accepted
everywhere a status is: `SwapStatus` (schemas_rerun.py:112) and
`MoreSeedsStatus` (:283) gain `"outpaint"`, or the response model rejects
it; the page's labels gain it in `web/tag-swap.js` (`SWAP_STATUS_WORDS`,
`SWAP_CHIP`, lines 25 and 31: "outpaint refused", "outpaint") and
`web/more-seeds.js` (`MS_STATUS_WORDS`, `MS_CHIP`, lines 26 and 31).

An outpaint never reports `mask_missing`: its mask is a function of the
request and is built again at the submit, which also heals a row whose mask
file was lost (`submission.missing_masks` skips outpaint rows).

**Words that say "paint".** An outpaint's user never paints, so a refusal
that says "paint it again" is wrong for one:

- `jobs.MaskGone` ("mask ... has just gone from this gallery; paint it
  again", jobs.py:609) says "submit it again" when any cell naming that hash
  has an outpaint.
- `storage.mask_path`'s errors as the cell begins are reworded by their
  caller in `jobs` for an outpaint cell: "the outpaint's mask has gone from
  this gallery; submit it again". `mask_path` itself keeps one wording;
  `unusable_input` never calls it for an outpaint (no mask yet).
- `check_refine_supported` branches on `request.inpainting`, as above.

## The fill and the mask: exact algorithms

`backend/outpaint.py` holds the pure parts: `FILL`, `FILLS`, `BAND`,
`MAX_BAND`, `padded`, `mask`, `presets` and `source_off_grid`. It imports
numpy, Pillow and `sizes` (itself a leaf) at the top and nothing else from
the package: `presets` takes the limits as arguments (`max_side`,
`max_pixels`, which its callers read from `current_settings`), and the one
part that needs `storage`, the stamp, lives in `storage` (below). The first
draft called it a leaf while giving it `stamp` and `presets`, which need
`storage` and `settings`; that is what moved.

Both builders are pure functions of their arguments with integer arithmetic
only, so the same source and spec give the same bytes on every machine and
every numpy version. `tests/test_outpaint.py` pins the SHA-256 of each for a
fixed fixture, so any change to either fails a test rather than silently
making old outpaints unreproducible.

### The padded source (`outpaint.padded(source, spec)`)

`FILL = "edge-box3-r16"`, `FILLS = {FILL: padded_edge_box3_r16}`. The name
is in every outpaint's key, column and chunk.

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

`padded` dispatches on `spec.fill` through `FILLS` and raises on a name it
lacks (the backstop behind the submit check). The fill is never stored: it
is derived from the source's pixels (already in the reproduction set, by id
and by `refine_source`'s pixel hash), the padding and the fill's name, all
recorded. Storing it would be a second large file per outpaint for no
information. `refine.outpaint_source(image_id, spec, width, height)` beside
`refine.inpaint_source` loads the source through `storage.reference_paths`,
closes the handle, refuses a size that is not source plus padding (the
backstop behind the submit check, as `inpaint_source` refuses another size)
and returns `outpaint.padded(...)`.

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

So the mask is a function of the canvas size, the four sides and the band
(the source's size is the canvas's minus the padding); the fill does not
enter it. Its kept rectangle is the source inset by the band on its padded
sides; the submit check holds it to at least 16 px each way. Because each
side and the band are on the latent grid, `refine.pipeline_mask` of this
mask is its own threshold - the class repaints exactly the band and the
border - and `refine.composite` blends the band by the ramp's levels and
keeps every pixel inside the kept rectangle as the source's. Tested.

### Stamping: the mask becomes a stored mask

`storage.stamp_outpaints(cells)` builds each distinct outpaint's mask once
(one per canvas size and spec, so normally one per job), stores it with
`store_mask` (idempotent, content addressed; it takes the gallery
reservation itself, waiting up to `gallery_lock.SESSION_WAIT`, and restarts
the settle window) and answers new cells with `mask` set to its hash by
`model_copy` (`dataclasses.replace` on the frozen `Cell`). It is called in
exactly two places:

- **`JobManager.submit_cells`, as its first step**, before `Job(...)` is
  built. `Job(cells=cells, request=cells[0].request, run_id=...)` keeps the
  list it is handed, so a stamp after it (the first draft's "just before
  `_cell_masks`") would leave `job.cells` unstamped: the worker would run
  the outpaint with `mask=None`, `refine.mask_image(None, ...)` would fail,
  and the hold would cover a hash no cell names. Stamped first, `job.cells`,
  `job.request`, `_cell_masks(cells)` and `storage.hold_masks` all see the
  stamped cells. It runs through `asyncio.to_thread`, since `store_mask`
  encodes a PNG and may wait on the reservation, neither of which belongs on
  the event loop. Both `OSError` and `gallery_lock.GalleryBusy` are a worded
  503 ("the outpaint's mask could not be written: ..." / "the gallery is
  busy with another operation; submit the outpaint again"), as
  `POST /api/masks` says one. Every producer (Generate, a grid, Re-render,
  Swap tags, More seeds) goes through `submit_cells`, so each gets it once.
- **`generate.py`, where `_mask_unusable(..., store=not args.dry_run)`
  runs** (line 843): after `expand` and the resume prune, before
  `backend.load()`. Under `--dry-run` nothing is stamped or stored (the
  cells keep `mask=None`, which nothing in a dry run reads; their keys are
  the same either way, below). An `OSError` or `GalleryBusy` is a worded
  exit 2. The CLI writes no `mask_holds` rows, as `--mask` writes none
  today: it runs its cells in its own process straight after, and the store
  restarts the settle window that keeps a sweep off a fresh mask. The first
  draft's "the CLI's copy of the submit loop" named nothing: the deliberate
  copy `tests/test_cli.py` holds equal is `jobs.submit()`'s seed loop, and
  the CLI holds no copy of `submit_cells`.

`run_id` is the same over stamped and unstamped cells, because the key
element names an outpaint by its spec, never by its hash (below); a test
pins it, so a producer that passes a `run_id` computed before the stamp and
`submit_cells`' own `_run_id(cells, ...)` after it agree.

From there an outpaint's cell is an inpaint's cell with an `outpaint`
beside it: held in `mask_holds` while queued, its mask file checked as the
cell begins (`jobs`, `storage.mask_path`, reworded as above), read by
`refine.mask_image`, recorded by `save_image`, embedded in
`local-diffusion-mask`, carried by bundles and reindex, spared by
`gallery_delete.sweep_masks` while any row names it.

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

- `images.outpaint TEXT`: `OutpaintSpec.key()`, NULL on every other row,
  through a `_migrate` entry (with an index `idx_images_outpaint`, as `mask`
  has one), `_COLUMNS`, both explicit dicts (`save_image`,
  `storage.index_file`) and `_row_to_record`, which reads it unguarded
  through `OutpaintSpec.parse_key` - safe because the stored type has no
  setting-bound limit and only validated values are ever written.
  `ImageRecord.outpaint: OutpaintSpec | None`.
- `refine`, `strength` and `mask` are set as on an inpaint: `refine` is the
  unpadded source, `mask` the stamped hash. The row's `width` x `height` is
  the canvas.

### The chunk and the A1111 block

- The `local-diffusion` chunk gains `outpaint` (`{"left", "top", "right",
  "bottom", "band"}`) and `outpaint_fill` (the spec's fill name), null
  together on every other row. `mask`, `mask_source`
  (`storage.mask_provenance`, at the canvas's size) and `mask_blend:
  "overlay"` are written as for any inpaint, and the mask's PNG rides in
  `local-diffusion-mask` as for any inpaint, so the PNG alone carries its
  mask even though it could be rebuilt. `refine_source` is the unpadded
  source's provenance (its pixel hash and its own size), which is what
  `--from-image`'s `_check_carried_source` compares.
- The A1111 block gains `, Outpaint: L/T/R/B, Outpaint band: B, Outpaint
  fill: NAME` after `Inpaint mask` (slashes, since an A1111 value holds no
  comma). Besides being readable, it is the one copy of the spec an older
  build leaves in place (below, "Older builds").

### CellKey and run_id: folded into the mask element

The outpaint goes **into the existing fifth extension**, not a sixth:
`grid.mask_key(mask, outpaint=None)`, where `outpaint` is the key text,
answers `f"outpaint:{outpaint}"` (for example
`"outpaint:edge-box3-r16:400,0,400,0@32"`) where there is an outpaint,
whatever the mask; the hash where there is only a mask; and None otherwise.
Every caller passes both:

- `grid.cell_key`, from a request (`request.outpaint.key()` or None);
- `storage.completed_cells`, from the `mask` and `outpaint` columns;
- `more_seeds.settings_key` (more_seeds.py:66), from the record
  (`grid.mask_key(record.mask, record.outpaint.key() if record.outpaint
  else None)`), so its "the resume key's elements" stay equal to
  `completed_cells`' - otherwise More seeds would group an outpaint's batch
  under its mask hash while resume keys it by spec.

`_KEY_EXTENSIONS` stays 5.

- **Why the spec and not the hash:** the hash is a function of the canvas
  size and the spec, and the size is already in `completed_cells`' scope and
  `run_id`'s definition, so the spec names the same picture; the page's
  preview (no stamped hash yet), the unstamped cell, the stamped cell and a
  `--dry-run` then key the same, so a grid preview's "N already rendered"
  and the job agree without storing anything on a keystroke.
- **Why the fill is in it:** the spec names the picture only if it names
  how the new area was filled. With the fill in the key, a changed fill is
  a new key, so a resume can never count pictures made with another fill as
  done; this replaces the first draft's `unknown_fill` nulling, which left a
  row with a mask but no outpaint - shown as an inpaint whose mask is the
  canvas's size, Reused into the Inpaint dialog with a mismatched mask, and
  refused by Re-render with "paint the mask on that image again".
- **Why fold it in:** an outpaint and a painted mask never meet in one
  request, and both answer the one question "which region is repainted",
  which is what the element means. A sixth element would add a None to the
  end of every key and shift `duplicate_count`'s slicing again for no
  information.
- **run_id unchanged for every existing run**, inpaints included: no
  existing row has an outpaint, so every element is what it was. Pinned in
  `tests/test_conditioning.py` beside the existing pins, with a new pin for
  an outpaint run and a test that stamped and unstamped cells give one
  `run_id`.
- **`duplicate_count`** keeps every element after the refine one
  (`(*key[:8], f"{refine}@step:{start}", *key[9:])`), so two paddings, two
  bands or two fills over one source are two pictures; its `start_index(...,
  inpaint=...)` reads `request.inpainting`.
- **A rule:** each fill in `FILLS` and `outpaint.mask` are frozen. A change
  to a fill is a new `FILL` name (the old one may stay in `FILLS` to keep
  its outpaints reproducible); a change to the mask is a change to every
  outpaint and needs its own key element. The pinned hashes in
  `tests/test_outpaint.py` are what make that impossible to forget.

### Bundles

- **The mask:** nothing new - it is a stored mask, carried as
  `masks/<hash>.png` with the `masks` table, verified by
  `bundle_images._inspect_mask`, merged by `bundles._merge_masks`. Where
  neither the bundle nor the result's embedded copy
  (`bundles._embedded_member`) has an outpaint's mask, the import rebuilds
  it from the spec and the row's size and installs it only if its hash is
  the row's; it counts `outpaint_masks_rebuilt`, and a mismatch keeps the
  row's hash and counts in `masks_missing` as today. The manifest stays
  `version: 1`.
- **The column** travels in the database (`_COLUMNS`); the merge's id remap
  touches only `refine`, since the spec holds no id.
- **The chunk.** `bundle_images._metadata` (lines 186-210) rebuilds the
  `local-diffusion` chunk from the row on every merge, and
  `_rewrite_metadata` replaces it; without the new keys, an imported
  outpaint's PNG would lose its spec and a later reindex would read it back
  as a plain inpaint. `_metadata` gains `outpaint` and `outpaint_fill` from
  the row's column, null together where it is NULL.
- **The source's fate.** In `bundles.py`'s merge (lines 775-780), where the
  source does not resolve, `values["outpaint"]` is cleared beside
  `strength` and `mask`, by the invariant.
- **`_validate_row`** (`bundle_images.py:133-139`) refuses an outpaint
  without `refine`, without `mask`, or whose key `OutpaintSpec.parse_key`
  refuses or whose sides are all 0: "invalid outpaint on image N".

### Reindex

Three builders, each carrying the column and the keys, each holding the
invariant:

- **`storage.index_file`** (the PNG-only path) reads `outpaint` and
  `outpaint_fill` from the chunk into `OutpaintSpec` (a malformed one is
  left NULL and counted `outpaint_malformed`). Any well-formed fill is kept,
  known to this build or not: the submit refuses an unknown one by name, so
  the row still shows as what it is. It drops `outpaint` whenever `refine`
  or `mask` is None. Mask recovery is as for any inpaint, plus the same
  rebuild from the spec where no file and no embedded copy is found.
- **The directory reindex** (reindex.py:583-600) spreads `{**meta}`, which
  would keep the outpaint keys when `refine` fails to resolve and the mask
  is dropped, leaving an outpaint with no source. It sets `outpaint` and
  `outpaint_fill` to None wherever `refine` (or `mask`) is None, beside
  `strength` and `mask_source`, and counts the change in `links_changed`.
- **`reindex.merge_database`** builds its own row (`values`, reindex.py:
  440-470) and its own chunk (`_database_meta`, 300-337); both gain
  `outpaint` (and the chunk `outpaint_fill`), dropped where `refine` or
  `mask` is.

### Older builds

The first draft said an older build imports an outpaint "as an inpaint row
with an unknown column (kept through the unlisted-column fallback)". That is
wrong: an older build's merge copies only its own `_COLUMNS` (bundles.py:758)
and rewrites the chunk through its own `_metadata`, so the column and both
chunk keys are dropped; the unlisted-column fallback is about older tables
gaining columns, not about extra columns in a bundle. Nor can "the import
result should say" anything be added to builds already shipped. So:

- In an older build an outpaint becomes an inpaint whose mask is the
  canvas's size over a smaller source. Its Re-render refuses it
  (`submission.mask_unusable`: the mask and the source are not one size),
  which is the safe failure there.
- A round trip back would make that permanent - except that an older
  `_rewrite_metadata` replaces only the `local-diffusion` chunk (and the
  legacy `krea`), so the A1111 `parameters` chunk, with its `Outpaint: ...,
  Outpaint band: ..., Outpaint fill: ...`, survives. **Adopted:** this
  build's bundle merge and both reindex paths recover the spec from it
  where a row has `refine` and `mask` and no outpaint, its mask's size
  differs from its source's size, the canvas is the source plus the
  parsed padding, and `outpaint.mask` of that canvas and spec hashes to the
  row's mask. Counted `outpaints_recovered`. The hash test makes the
  recovery exact or nothing; where the source is gone the row stays the
  older build's inpaint.

### The bin, deletes, `clear`, the sweep

Unchanged. An outpaint's mask is named by its row, so `sweep_masks` spares
it while the row lives, binned or not, and takes it with the row's last
permanent removal.

## Lineage and gallery actions

`grid.cell_from_record` carries `outpaint` with the source and strength and
sets `mask` to None where the record has an outpaint (the stamp rebuilds and
re-stores it, and the validator refuses the two together): `"outpaint":
record.outpaint if record.refine is not None else None`, `"mask":
record.mask if record.refine is not None and record.outpaint is None else
None`. `GenerateRequest(**fields)` validates the stored spec as an
`Outpaint`; a record whose band is past `MAX_BAND` raises the
`ValidationError` the producers already report as `invalid`.

**The spec-keyed check caches.** Three producers check once per distinct
combination of inputs, keyed on a tuple that today ends with `mask`. An
outpaint cell has `mask=None` (from `cell_from_record`), so without the
outpaint in the tuple it would share a verdict with a plain refine of the
same source and its size check would be skipped or run on another cell:

- **Re-render** (`api_generate.py:218-224`): the tuple gains
  `c.request.outpaint` (frozen, so hashable; its `key()` is what compares),
  and the `model_copy(update=...)` gains `"outpaint": spec[4]`, so
  `outpaint_unusable` runs at the asked size for each distinct padding.
- **Swap tags** (`api_swap.py:104`) and **More seeds**
  (`api_more_seeds.py:104`): the cache tuple gains the outpaint the same
  way.

The actions:

- **Re-render** runs the same padding over the same source at the record's
  own size. At another size it is refused for the whole selection as an
  inpaint is (`submission.resized_inpaints` already catches it, since the
  row has a mask); `resized_inpaint_detail` says "#M is an outpaint of #N at
  WxH" for it, and "an outpaint runs at its source's size plus its
  padding". The way out stays "Add inpaint:no to the search", which leaves
  outpaints out too.
- **More seeds** runs the next seeds over the same padding, band and fill.
- **Swap tags** rewrites the prompt over the same padding.
- **Mass refine** makes the record itself the new source and drops both the
  mask and the outpaint (`refine_batch` passes `mask=None, outpaint=None`);
  its "Skip refine passes" line (`refine_batch.py`) says "an outpaint of #N".
- **Reuse** on an outpainted record opens the Outpaint dialog on its source
  with the four sides, the band, the strength, the seed and the prompt
  (`reuse.js`, beside `reuseInpaint`; an outpaint is checked first, since it
  also has a mask). A source that is gone or binned is said as the Inpaint
  dialog says it. An unknown fill opens the dialog with Outpaint held by
  the submit check's words.
- **Load settings from PNG** carries no source, so no mask and no outpaint.
  The first draft said the panel takes the PNG's size "as it does for an
  inpaint's PNG"; the code does the opposite: `api_png._from_record` sets
  `size = {}` for any record with a `refine` (api_png.py:173-181), so a
  pass's, an inpaint's and an outpaint's prompt and settings fill the panel
  at the panel's own size. That stays, and the line's `what` gains a third
  case: `"an outpaint"` where the record has one, before `"an inpaint"`
  where it has a mask. `png_settings` reads the two new chunk keys as known
  and ignores them.
- **`--from-image`**: see the CLI.
- **Search:** `outpaint:yes|no` (`outpaint IS NOT NULL` / `IS NULL`, through
  an `outpaint_clause` beside `search.inpaint_clause`, in `search.FILTERS`),
  and `web/gallery-search.js`' `SEARCH_KEYS` gains `'outpaint'` (held equal
  to `search.FILTERS`' keys by `tests/test_storage.py`). `inpaint:yes`
  still means "a mask on the row" and so includes outpaints - mechanically
  each is an inpaint, and both share the size rule Re-render refuses by -
  and `inpaint:yes outpaint:no` is the painted ones. `stats --by outpaint`
  is optional.
- **Words:** cards and the job line say "outpaint of #N"; the facts list
  says "Outpainted from #N @ 0.9999" (the strength as recorded) and an
  "Outpaint" row "+L/+T/+R/+B, band 32 px, fill edge-box3-r16". The job
  snapshot (`Job.one_source`) gains the outpaint beside the mask so the
  page can name a queued job.
- **The lightbox's Against source** compares at the record's own geometry:
  for an outpaint the source is drawn at `(left, top)` at the result's scale
  rather than scaled to cover it (`lightbox-compare.js`), so the divider
  shows the original against the kept rectangle pixel for pixel.

## Estimates

`gallery_reports.refine_cost_ms` already splits on `mask IS NULL` and groups
by denoised steps and size, so an outpaint is priced from inpaint history at
its canvas size, outpaints included. No new split: the call is the same
class over a picture of that size. But an outpaint's canvas is almost never
a preset size (it is a source plus a padding), so in practice
`inpaint_costs` at the canvas size will usually have no history, and the
dialog should expect no estimate and say it as the existing wording does
("no estimate: nothing has run at 1824 x 1024 here").

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

### Page carry sites keyed on `rec.mask` or on open dialogs

Each of these would treat an outpaint as an inpaint, or the open Outpaint
dialog as absent:

- **`app.js` keyboard (lines 519-561).** Ctrl/Cmd+Enter submits the open
  dialog's run and otherwise the panel's `generate()`; Escape closes the
  model picker or the prompt check only when no dialog is open. Each guard
  gains `els.outpaintDialog.open` (Ctrl+Enter: `else if
  (els.outpaintDialog.open) void runOutpaint();`), or Ctrl+Enter in the
  Outpaint dialog would run the panel and Escape would close the picker or
  check under it.
- **Reuse buttons (`gallery.js:779-880`).** `syncReuseButton` keeps
  `reuseMask` in the button's data attributes and `syncReuseButtons`
  rebuilds the record from them after a status refresh; without the
  outpaint there, an outpaint's button turns back into "Reuse inpaint…"
  and opens the Inpaint dialog with a canvas-sized mask over a smaller
  source. It gains `dataset.reuseOutpaint` (the key text), read back into
  the rebuilt record, and the button's text, title, aria-label and tip say
  "Reuse outpaint…" / "Open the Outpaint dialog on #N with this outpaint's
  sides, band, strength and seed".
- **Labels:** the card's `inpainted` class (`gallery.js:409`) gains an
  `outpainted` sibling with its own badge; the card line (`gallery.js:841`,
  "inpaint/refine of #N") says "outpaint of #N"; the viewer's
  `passLineage` (`viewer.js:418`) "Outpaint of"; the job line
  (`job-progress.js:282`) "the outpaint of #N" from the snapshot's
  outpaint; the facts (`facts.js:83,86`) "Outpainted from" and the Outpaint
  row beside the Mask row (whose title says the server built it).
- **Search keys** (`gallery-search.js:11`), above.

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
  `outpaint.presets(width, height, max_side=..., max_pixels=...)`, which the
  CLI's `--outpaint 16:9` uses too), so the page holds no copy of the
  arithmetic. The route also answers the source's size and `off_grid`.
  All arithmetic is integer:
  - **to a:b** extends only the axis that is short of the ratio and never
    crops: a source with `width * b < height * a` is narrower than a:b, and
    the canvas's width is the smallest multiple of 16 at or above
    `height * a / b`, `-(-height * a // (b * 16)) * 16`, with the height
    kept; otherwise the same for the height,
    `-(-width * b // (a * 16)) * 16`. The padding on that axis is split
    evenly, the first side taking `pad // 32 * 16` and the second the rest
    (the odd 16 px to the right or bottom).
  - **"Already at the ratio"** is defined, because the `SIZES` entries only
    approximate their aspects and the arithmetic alone would pad them: the
    16:9 preset's 1360 x 768 to 16:9 gives a width of
    `-(-768 * 16 // 144) * 16` = 1376, +0/+0/+16/+0, and the 3:2 preset's
    1264 x 848 to 3:2 gives +16 too. A to-a:b preset is **held** when the
    source is that aspect's `SIZES` entry at some tier ("#12 is already
    16:9 - the 16:9 preset at 1 MP"), or when its total padding would be
    under 32 px ("#12 is within 16 px of 16:9"). Both cases are tested.
  - **+25% all round** grows each axis by about a quarter of its side,
    rounded to 16 half up with at least 32 so each end gets at least 16:
    `max(32, (side + 32) // 64 * 16)`, split evenly the same way. A
    1024 x 1024 source becomes 1280 x 1280 at +128 on every side.
  - A preset whose canvas would pass `max_side` or `max_pixels` is held with
    the size it would be and the budget it passes.
  - Over an off-grid source every preset is held with `off_grid`'s words.
- **Band:** a number box, `step="8"`, 0-256, rounded to 8 as the sides are,
  with its note "this app's starting value; nothing measured says what
  width joins best".
- Then the shared rows: the prompt (starting from the source's, said as
  the Inpaint dialog says it), its Check lines, Strength (starting at the
  library's default, said as the library's), Versions, Seed, the preview
  line (with its estimate, usually none, above) and **Outpaint**.

### Held states

**Outpaint** is `aria-disabled` with the reason as its title, and a press
says it, while: no side is extended ("extend at least one side"); the source
is off the grid (its words); the canvas passes the side or pixel budget (the
size and the budget); the band leaves no kept rectangle; the preview refuses
(its words, including an unknown fill); the source is gone or binned; the
checkpoint has no inpaint class; a submit is in flight. A refusal replaces
the preview line at its height, as the Inpaint dialog's does. A close that
ran nothing keeps the four sides and the band in page memory for the next
open of the same image, as the Inpaint dialog keeps its strokes, never in
`localStorage`.

## The CLI

`generate.py --refine ID --outpaint L,T,R,B [--band PX]`:

- `--outpaint` takes four multiples of 16 or a preset name (`16:9` or any
  `sizes.SIZES` aspect, or `+25%`), resolved by `outpaint.presets`; a side
  off the grid is refused with its neighbours, never rounded silently; an
  off-grid source is refused with `source_off_grid`'s words before any
  request is built. The fill is always this build's `FILL`.
- `--band` defaults to `outpaint.BAND`; it needs `--outpaint`.
- The size is the source's plus the padding. `--width`, `--height`,
  `--aspect` and `--mp` are refused beside `--outpaint` only where the user
  typed them - the refusal reads `size_flags` (generate.py:600), the user's
  own flags before a PNG fills any in - and `--scale` and `--mask` are
  refused beside it ("an outpaint builds its own mask"). With an outpaint,
  a size `--from-image` filled from the PNG (`args.width` / `args.height`,
  generate.py:338-343) is ignored: the canvas is always source plus padding.
- The `refined_from` condition (generate.py:665, `args.scale is not None or
  not sized or inpainting`) includes outpainting, so the source's row is
  read for its size.
- The mask is stamped where `_mask_unusable` runs, as above; `--dry-run`
  prints the canvas and the padding and stores nothing. The CLI writes no
  `mask_holds` rows, as `--mask` writes none.

**`--from-image` of an outpaint.** Three collisions with today's code, each
resolved:

- `_apply_image_settings` sets `args.mask_hash = meta["mask"]`
  (generate.py:335); with an outpaint carried too, the request would hold a
  mask and an outpaint and be refused by the validator. Where the chunk
  carries a well-formed `outpaint` (and `--outpaint` and `--mask` are not
  given, under the same condition that carries a mask today), it sets
  `args.outpaint` from `outpaint` and `outpaint_fill` and never
  `mask_hash`. The chunk's `mask` is kept aside only for one comparison:
  once the source loads, `outpaint.mask` of the canvas and spec must hash
  to it, else a refusal ("this build builds an outpaint's mask differently
  from the one that made this image"). An unknown fill is refused by name,
  as at the server.
- The PNG's `size` fills `args.width` / `height`; for a carried outpaint it
  is ignored as for a given one (above), and the `--width`/`--height`
  refusal reads `size_flags`, never the filled values.
- `_check_carried_source` drops `refine` (generate.py:385-389) when the
  source is not the picture the PNG was made from; it drops `args.outpaint`
  too, or the run would fall through to a plain render at the canvas's size,
  which can be past `max_side`. Its `what` says "an outpaint" where one was
  carried, and its refusal branch counts `--outpaint` among the flags that
  make the drop an error.

## Grid and comparison

The padding, the band and the fill are fixed across a job, as a mask is;
none is an axis. Everything that is an axis over an inpaint is one over an
outpaint: the Strength box's list or range (`GridRequest.strength_axis`),
the seed axis (Versions), a prompt variable. Each cell's key carries the
outpaint in the mask element, so resume, the duplicate count and `run_id`
treat two paddings as two runs. The contact sheet needs nothing new. The
panel's prompt comparison never sends a source, so it never outpaints.

## Tests and smokes

Python (`uv run --group dev pytest`, no GPU):

- `tests/test_outpaint.py`: both types' validation (off-grid sides with
  their neighbours, no side extended on the request but parsed on the
  stored type, band off its grid or past 256, a malformed fill name, an
  outpaint without a source, with a mask); the stored type parsing a row
  whose sides are past a lowered `refine_max_side`; `key` / `parse_key`
  round trips with the fill; `presets` for each aspect against hand-worked
  numbers (1024 x 1024 to 16:9 is +400/+0/+400/+0 at 1824 x 1024; to 2:3 is
  +0/+256/+0/+256 at 1024 x 1536; +25% is +128 all round) and the held
  cases (each `SIZES` entry to its own aspect at every tier; 1376 x 768 to
  16:9, within 16 px; past the budget; off-grid); `mask` levels at the
  border, at `d = 0`, `band - 1` and `band`, in corners where two bands
  meet, with `band = 0`, and with unpadded sides having no band; `padded`
  keeping every source pixel and stretching the edge, and raising on an
  unknown fill; both pinned by SHA-256 on a fixture; `pipeline_mask` of an
  outpaint mask equal to its own threshold; `refine.composite` over the
  padded canvas keeping the kept rectangle exactly.
- `tests/test_api_outpaint.py`: the submit's refusals (size not source plus
  padding, past the budget, band too wide, an unknown fill by name, an
  off-grid source by name, binned or deleted source, a checkpoint without
  the class with the outpaint words, a mask beside an outpaint); stamping
  (each queued cell in `job.cells` carries the derived hash, the mask is
  stored and held in `mask_holds` under that hash, `GalleryBusy` and
  `OSError` while storing are worded 503s, a lost mask file healed by the
  next submit, `MaskGone` and the cell-begin refusal saying "submit it
  again"); the presets route; the grid preview with a strength axis;
  Re-render, More seeds and Swap tags carrying the outpaint, and each
  one's check cache keeping an outpaint apart from a plain refine of the
  same source (an outpaint at the wrong size refused beside a refine that
  passes); `SwapStatus` / `MoreSeedsStatus` answering `outpaint`; Mass
  refine dropping it; Re-render at another size refused naming "an
  outpaint of #N"; Load settings from PNG of an outpaint's PNG saying "an
  outpaint" with no size.
- `tests/test_conditioning.py`: `mask_key` with an outpaint, with the fill
  in it; every existing `run_id` pin unchanged; a pin for an outpaint run;
  `run_id` equal over stamped and unstamped cells; `duplicate_count` with
  two paddings, two bands and two fills; `more_seeds.settings_key` equal to
  `completed_cells`' key for an outpaint row.
- Storage: `_migrate` adds `outpaint`; `save_image` and `index_file` write
  and read the chunk keys and the A1111 line; `completed_cells` keys an
  outpaint row on its spec; an unknown but well-formed fill kept on
  reindex; `index_file` dropping the outpaint where `refine` or `mask` is
  None.
- Bundles and reindex: the column and the chunk keys travel through
  `_metadata` and `_rewrite_metadata`; an unresolved source clears the
  outpaint; `_validate_row` refusing an outpaint without `refine`, without
  `mask` or malformed; the directory reindex and `merge_database` (both its
  `values` and `_database_meta`) carrying and dropping it; an outpaint's
  mask rebuilt from the spec where nothing carries it, refused where the
  hash differs; recovery from the `parameters` line for a row stripped as
  an older build strips it, and no recovery where the rebuilt hash differs.
- `tests/test_cli.py`: `--outpaint` (sides and presets), `--band`, the
  refusals (reading `size_flags`), the stamp skipped under `--dry-run`,
  `--from-image` of an outpaint carrying `args.outpaint` and never
  `mask_hash`, ignoring the PNG's size, the hash comparison's refusal, and
  `_check_carried_source` dropping the outpaint with its source.
- `tests/test_inpaint_ui.py` or a sibling: the page's held reasons against
  the status keys; `SEARCH_KEYS` equal to `search.FILTERS`.

Smokes, at 1400x900:

- **`smoke-outpaint`** (Animagine fake, port 7907, fresh scratch `--data`):
  open from the lightbox; a typed 50 committed as 48 with its line; the
  presets' exact numbers on a 1024 x 1024 planted image; a to-16:9 preset
  held on a planted 1360 x 768 image (the 16:9 entry) and on a planted
  1376 x 768 (within 16 px); the frame's source placement against the
  boxes; Outpaint held with each reason; no self-scroll (`scrollTop`
  unchanged through edits, presets and a refusal) and every control's box
  unchanged under the pointer as numbers change; the lightbox row not
  wrapping; Ctrl+Enter in the dialog runs the outpaint, not the panel, and
  Escape closes only the dialog; submit; the result says "outpaint of #N
  +400/+0/+400/+0" and its kept rectangle equals the source's pixels (the
  fake); Reuse reopens with the sides and band, and still says "Reuse
  outpaint…" after a status refresh; a strength list opens a sheet.
- **`smoke-inpaint-unavailable`** (default Krea 2 Turbo fake, port 7906)
  gains the outpaint entry points: `aria-disabled` with the reason, and a
  press says it.
- Later: a wide-tier case in `smoke-wide`.

When phase 1 lands, `docs/rules.md` gains the rules (the two types and
where the limits live, the size rule's exception, stamping first in
`submit_cells`, the frozen fills and mask, the folded key element with the
fill, the invariant that an outpaint needs `refine` and `mask`), its Tests
section the smoke and port, `CLAUDE.md` / `AGENTS.md` the port row,
`docs/design-notes.md` an "Outpainting" section, and `docs/verified.md` an
"Outpaint" section listing the measurements below as not yet measured.

## Phases

Effort on the review's scale: S a day or less, M a few days, L a project.

**Phase 1 - backend and reproduction set (M).** `backend/outpaint.py` (fills,
mask, presets, off-grid words); the two types, the validators,
`inpainting`, `applied()`, the split submit check and its status carry
sites; `storage.stamp_outpaints` and its two callers; `refine.outpaint_source`
and the backend branch; the column, the chunk keys, the A1111 line;
`mask_key` and its three callers; `cell_from_record`, the three check
caches, Re-render's wording, Mass refine's drop, Load settings from PNG's
words; bundles' `_metadata`, `_validate_row`, the merge's clear, the
rebuild and the recovery; reindex's three builders; the presets route;
search; the CLI and `--from-image`; the fakes; every Python test above.
Smaller than inpainting's phase 1, because the mask machinery exists and is
reused whole.

**Phase 2 - the dialog (M).** `outpaint.js`, the CSS, the entry points, the
keyboard guards, Reuse and its data attributes, the facts, card, viewer and
job wording, the status labels, the search key, Against source at the
offset, the smokes, the docs listed above.

**Phase 3 - with the measurements (S).** Once the card has said anything:
a band default that is measured rather than this app's, and anything the
inpaint phases change for every inpaint (a `START_RECIPES` line, the
preview through the blend). The deferred families get outpainting the day
their inpaint is enabled, since the gate is `inpaint_supported`; their
memory at outpainted sizes is part of their phase.

**Later, if wanted:** the server's fill shown in the frame (a model-free
`GET` answering the padded canvas as a JPEG, so the page still draws
nothing of its own); a card tool; `padding_mask_crop` for large canvases on
SDXL; an outpaint over a painted mask together; a second fill.

## Open measurements (the card's to say)

1. Peak memory and time per step at outpainted sizes for WAI and one other
   SDXL checkpoint, and Z-Image Turbo and Z-Anime: an outpaint's canvas is
   larger than its source (a 1 MP source to 16:9 is about 1.8 MP, +25% about
   1.6 MP), so it meets the card's limits sooner than an inpaint of the same
   source. One size per run, stopping at the first that fills the card.
2. Determinism: the same outpaint twice, batch image *i* against a solo run
   of its seed, cancel-then-rerun, the text-to-image hash afterwards.
3. What the fill contributes at the library default (0.9999 on SDXL, which
   starts at the second timestep from the noised fill; 1.0 on Z-Image) and
   at lower strengths - whether the stretched edge shows.
4. The seam at band widths 0, 32, 64 and 128 px per family, against the
   VAE round trip at the original's edge.
5. A starting band width, if one is measured to join better than 32.
6. The embedded mask's size in an outpaint's PNG (mostly white, unlike an
   inpaint's mostly black one).
7. The fill's and the mask's build time on the CPU at the refine guard's
   largest canvas (both are numpy passes over the whole canvas).

## Risks

- **A frozen algorithm.** The key element names an outpaint by its spec, not
  its hash; with the fill's name in the spec, a changed fill is a new key
  and cannot be counted as done by a resume. What remains is a change to a
  fill *without* a new name, or to the mask (whose builder has no name):
  the pinned hashes in `tests/test_outpaint.py` and the rule are the guard.
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
- **Carry sites.** The outpaint rides beside the mask through many
  hand-built dicts (the row, three reindex builders, the bundle chunk, the
  check caches, the page's data attributes). The review found eleven the
  first draft missed; the invariant (never without `refine` and `mask`) and
  a test per site are the guard, and any new dict that carries `mask` must
  carry `outpaint`.
- **Quality.** No catalogue author documents outpainting, and the fill is
  this app's own choice; how outpainted pictures look is unmeasured, and
  the app says only what is measured.
- **Memory.** A canvas larger than any direct render of the checkpoint's
  presets is reachable within the refine budget; only the card can say
  where each family stops.
- **Older builds.** An older build drops the column and the chunk keys on
  import, so an outpaint there is an inpaint whose mask is the canvas's
  size over a smaller source, which its Re-render refuses. Nothing can be
  added to builds already shipped. Brought back, it is recovered from the
  A1111 `parameters` line where the rebuilt mask's hash matches; where
  that line is gone too (a tool that strips text chunks), it stays an
  inpaint row that cannot be run again.

## Open questions for the user

1. A card tool for Outpaint beside the pass and the inpaint, or only the
   lightbox, the viewer and Reuse (recommended for now)?
2. The band's starting value: 32 px (this app's, unmeasured) until the card
   says otherwise?
3. "+25% all round" as written here (each axis about a quarter larger,
   split evenly), or a quarter of each side added on every side (each axis
   half larger)?
4. The server's fill shown in the frame preview in phase 2, or later?

## Review of 2026-10-13

Each point of the review of the first draft, and what became of it:

1. **Stamp too late in `submit_cells`** - adopted: stamped first, before
   `Job(...)`, through `asyncio.to_thread`; `GalleryBusy` and `OSError` both
   worded; a test that `run_id` is the same over stamped and unstamped cells.
2. **The CLI has no copy of `submit_cells`** - adopted: stamped where
   `_mask_unusable` runs, after `expand` and the resume prune, before
   `backend.load()`; skipped under `--dry-run`; the CLI writes no
   `mask_holds` rows, as `--mask` writes none.
3. **Three spec-keyed check caches** - adopted: the outpaint in each tuple,
   and in Re-render's `update`.
4. **Bundle import drops the outpaint from the PNG** - adopted: `_metadata`
   gains both keys; the merge clears the outpaint with an unresolved
   source; `_validate_row` refuses an outpaint without `refine`, without
   `mask`, or malformed.
5. **Reindex carry sites** - adopted: `merge_database`'s `values` and
   `_database_meta` carry it; `index_file` and the directory path drop it
   whenever `refine` (or `mask`) is None.
6. **The older-build claim** - adopted: the text and the Risk corrected;
   the optional recovery from the `parameters` line adopted, gated on the
   rebuilt mask's hash, with the fill added to that line.
7. **Stored rows depend on a configurable limit** - adopted:
   `OutpaintSpec` (plain ints, grid checks) for stored records; the limits
   only on the request, through `check_budget` on the canvas and a
   constant band bound.
8. **`--from-image` collisions** - adopted: `args.outpaint`, never
   `mask_hash`, with the chunk's mask kept for the comparison only; the
   size refusal on `size_flags` and the PNG's size ignored; the drop in
   `_check_carried_source`; `refined_from` includes outpainting.
9. **Load settings from PNG described wrongly** - adopted: the text says
   `size = {}` for any refine record, and `what` says "an outpaint".
10. **Status carry sites** - adopted: `SwapStatus`, `MoreSeedsStatus` and
    the four page label maps; `unusable_input`'s flag split
    (`request.inpainting` for capability and words, `request.mask is not
    None` for file checks); the unreachable third check dropped.
11. **Page carry sites** - adopted: the keyboard guards, the Reuse
    button's dataset, the labels, `SEARCH_KEYS`.
12. **More seeds builds its own key** - adopted: `settings_key` passes the
    outpaint to `mask_key`.
13. **"Already at the ratio" undefined** - adopted: held when the source is
    that aspect's `SIZES` entry at some tier, or the padding is under
    32 px; "+25%" written as `max(32, (side + 32) // 64 * 16)`.
14. **SDXL's library default** - adopted: the reading corrected (`t_start =
    1`, `add_noise` at the second timestep, `is_strength_max` False); what
    it means for the fill left to measurement 3.
15. **The fill name in the stored spec** - adopted: `fill` in
    `OutpaintSpec` and in `key()` (`edge-box3-r16:L,T,R,B@band`), so the
    column, the key element and `run_id` carry it; no existing `run_id`
    changes; an unknown fill refused by name at submit, never nulled
    (`unknown_fill` removed).
16. **Refusals that say "paint"** - adopted: `MaskGone` and the cell-begin
    `mask_path` refusal say "submit it again" for an outpaint;
    `check_refine_supported` branches on `request.inpainting`.
17. **An off-grid source** - adopted where it can run: the presets route,
    the dialog's held state, the CLI and `outpaint_unusable` name the
    source; over HTTP a hand-built off-grid canvas still meets the width
    validator first, which stays the backstop (a field error is raised
    before any check that could look the source up).
18. **Small points** - adopted: the dialog expects no estimate at most
    canvases; the quality Risk reworded to "unmeasured"; `outpaint.py` is
    numpy, Pillow and `sizes` only, with `presets` taking the limits as
    arguments and the stamp moved to `storage`.
