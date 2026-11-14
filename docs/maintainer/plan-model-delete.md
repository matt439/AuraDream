# Plan: delete a whole checkpoint from Models > On disk

**Status: planned 2026-11-14 (`GEN#72`). Stage 0 built 2026-11-14 (rows
named by entry, the lite and alias repositories on the entry's row, the
page's sub-lines); stage 1 built 2026-11-14 (the read-only planner, its
GET, the codes, the guards, `_deleting_model` wired inert); stage 2 built
2026-11-14 (the worker's converted-tree removal, backend only, no route);
stage 3 built 2026-11-14 (the Hub trash, the POST and its reservation,
Delete… in the page, `smoke-model-delete`); stage 4 built 2026-11-14
(files inside shared repositories, file by file). Built through stage 4;
Matt's acceptance on his real Windows cache is left.** The row's owner was the
[peer review's R9](review-2026-11-10-peers.md#r9-storage-view); this plan
owns it from now on (the TODO row points here). Every stage below is CPU and browser work on scratch
caches; none needs the card. Matt's questions are at the end; he
accepted every recommended answer on 2026-11-14.

Two drafts were merged: one written safety-first, one from the user's
side. Where they disagreed, the choice and one line of why are given in
place.

## Why

The read-only inventory (`GEN#55`, `GEN#71`) shipped 2026-11-11. On
2026-11-14 Matt looked at it over his real Windows cache: a 551.2 GiB Hub
cache, 926.7 GiB free, 48.3 GiB of converted copies, 17.5 GiB not
referenced, and one bare-id row (Krea 2 Turbo's Nunchaku source,
`lite-infer/krea-2-turbo-nunchaku-lite-nvfp4_r32-bnb4-text-encoder`,
10.9 MiB). His answer on what he would delete: **any model**, because he
no longer uses it, it is outdated, he needs the space, or its licence has
ended. So the delete takes a whole checkpoint, not only stale revisions or
unreferenced bytes. R9 asks the design to say what happens to files
another entry shares, to converted copies, to the loaded or selected
checkpoint, to a job or load in flight, and to the Nunchaku row's name.

Today the only route is by hand: `hf cache rm` (`docs/setup.md`, "uv run hf
cache ls") and removing a converted folder with the server stopped. Both
work per repository, which takes a sibling's files where two entries share
one (Anima Turbo and Aesthetic are two commits of `circlestone-labs/Anima`).

## Scope

**The unit is one catalogue entry**, named by its official id and resolved
by `launch.catalogue_entry()`. Its group is the entry's own storage row
(`entry.repo`, or an app selector such as
`circlestone-labs/Anima@aesthetic-v1.1`), each `entry.lite_repo(fmt)` for
nvfp4 and int4, and each cached `entry.lite_alias(fmt)`. One confirmation
deletes one entry; there is no multi-select.

A delete of entry E removes, on this server's cache only:

| Part | Rule |
|---|---|
| Selected Hub files | E's files at E's recorded commits (`backend/model-downloads.json` `sources`), where every owner of the file is in E's group, or the last-holder rule (Shared files) applies. |
| Repositories E owns outright | When no other entry's recipe names the repository, the whole `models--org--name` folder: every revision (GEN#71's other revisions, old downloads beside the recorded one), unselected files at the recorded commit (README, LICENSE, ERNIE's `pe/` and `pe_tokenizer`), `refs/`, `.no_exist/`. Examples: `black-forest-labs/FLUX.1-Krea-dev`, `krea/Krea-2-Turbo`, `baidu/ERNIE-Image-Turbo`. |
| Lite repositories | Whole, with the entry. No other entry claims a `lite-infer/...` repository. |
| Converted copy | `model_info._conversion_target(E)`'s tree and its `<target.name>-XXXXXXXX` work directories. The zero-byte `<target.name>.lock` stays (How files are removed). |
| Older converted recipes | Trees whose marker (`CONVERSION_MARKER`) parses as JSON and whose `source.repo_id` *and* `source.filename` are E's (the filename is what tells Anima Turbo's tree from Aesthetic's in one repository; the shared `shared`/`config` repository, SDXL base or Anima-Base, never matches). Every catalogue entry's current `_conversion_target` is always excluded. A tree whose source repository two entries share and whose filename matches neither entry's current one is held (`model_delete_held`), not guessed. Without a readable marker a tree stays "not referenced". *Why the marker, not the folder stem: the stem alone cannot tell a hand-made folder from ours; the marker is the loader's own record.* |

**Never touched:**

- Gallery rows, PNGs and their `auradream` chunk, masks, bundles: they are
  the reproduction record. Re-render and More seeds download again.
- `auradream.settings.json`, setups and presets, model favourites, saved
  searches. Applying one later selects or downloads again.
- LoRAs, the critic's llama.cpp and GGUF, the translator, the upscalers,
  the compile cache.
- FLUX 3's upsampler `black-forest-labs/flux-3-image-klein-pup` (18.8 GB):
  a helper, not an entry; it stays under Other repositories (question 6).
- Repositories no catalogue entry claims. The Other repositories list gets
  no Delete; `hf cache rm` stays the route for them.
- HF's xet chunk cache (`HF_HOME/xet`) and `.locks/`: not attributable per
  model; the library never removes `.locks` either.
- The legacy `local-diffusion/converted` root while the new root exists:
  the loaders cannot see it either (`legacy_names.hf_home_dir`).
- Another cache: the WSL server's ext4 `~/.cache/huggingface`
  (`wsl/env.sh`) and the Linux server's own.

## Design

### Routes

| Route | What it does |
|---|---|
| `GET /api/models/storage` | Read-only. `tests/test_model_cache_storage.py`'s DELETE-is-405 assertion stays. Each entry row gains `delete_held` (stage 3's review): its plan's `held` detail on the same scan, or null. |
| `GET /api/models/storage/plan?model_id=<id>` **(new)** | A dry run on a daemon thread (`detached(..., "model-cache-plan")`) under `model_cache_storage._scan_lock`. Answers `{entry, label, parts: [{kind: "hub"\|"lite"\|"converted"\|"converted-old", repo?, revision?, bytes, files}], kept_shared: [{bytes, with: [labels]}], also_taken_shared: [{bytes, with: [labels]}], frees_bytes, held: {code, detail} \| null, selected, loaded, load_on_startup, pin}`. |
| `POST /api/models/storage/delete` **(new)** | Body `{model_id, pin}` (pydantic, `extra="forbid"`). Answers 200 `{label, planned_bytes, freed_bytes, remaining: [{kind, bytes, reason}]}`. |

- **Pin.** sha256 over the sorted `(kind, path relative to its root, size,
  mtime_ns)` of every planned removal, plus the kept and also-taken sets
  and the converted target's name. The delete re-plans under its
  reservation and refuses a different pin with nothing removed, as Empty
  bin compares its `expected`/`pin`.
- **Paths.** Answers carry only repository ids, revisions and names
  relative to the cache roots, never an absolute path (the public-repo rule;
  the 503 wording already does this).
- **Partial results.** A refusal before anything changed is a coded 409. A
  delete that removed something and then hit a locked file is a 200 with
  `remaining`. *Why not a 409 (draft A): something did change; a refusal
  code would tell the page nothing happened.*
- **freed_bytes** is measured: the cache volume's free space before and
  after, beside the plan's logical `planned_bytes`. Another writer can move
  it, so the page says "about".

New codes in `backend/refusal_codes.py` and `CODES`, raised through
`refusals.Refused`; the page acts on the code only:

| Code **(new)** | When | Page |
|---|---|---|
| `model_delete_busy` | a job, card holder, load, switch, unload, drain, counted request, scan, Check's tokenizer fetch, or another delete | says the sentence; the row's Delete stays held until the status poll clears it |
| `model_selected` | E is the selected (and so any loaded) checkpoint | says "Choose another checkpoint first" |
| `model_downloading` | an `*.incomplete` under a repository the plan touches | says the sentence, naming the repository and the partial file's name relative to the cache (never `hf cache prune`: in huggingface_hub 1.33 it also removes every ref-less revision in every repository, and the app's commit-pinned downloads, such as Anima Turbo's and Aesthetic's `hf_hub_download` sources, write no ref) |
| `model_cache_changed` | the pin differs | re-reads the plan and asks again |
| `model_files_in_use` | Windows refused a rename: every rename made is put back, nothing changed | says the sentence |
| `model_not_on_disk` | E has nothing cached, or names no catalogue entry (one answer, 404) | refreshes the fold |
| `model_delete_held` | a part the server cannot prove safe: an ambiguous `other_revision_of`, a shared repository before stage 4 (`SHARED_REPOSITORY_DELETE` off), a shared Xet store | the row's Delete is held with the detail as its title |

### Guards: loaded, selected, jobs, loads, downloads

Checked in the route on the event loop, with no await between the last
check and the reservation:

| State | Check | Code |
|---|---|---|
| Selected or loaded | `model_selection.current_settings(settings).model_id` grouped to its entry (the lite id on Linux counts as Krea 2 Turbo); also `backend.is_ready()` with the backend's own id | `model_selected` |
| Saved for the next start | `auradream.settings.json`'s checkpoint is E with "Load automatically on startup" on, though this run's `--model`/`AURADREAM_MODEL_ID` overrides it (`startup_preferences.apply_saved_environment` leaves the saved choice in place) | `model_selected`, its sentence naming the saved choice |
| Jobs queued or running, a More seeds intent pending | `jobs.busy()` | `model_delete_busy` |
| The card held (critic, Expand, an edit) | `jobs.card()[0]` | `model_delete_busy` |
| A load in flight | `_load_lock.locked()` or `_load_state["state"] == "loading"` (both: `POST /api/model/load` reserves "loading" before it takes the lock) | `model_delete_busy` |
| A switch, unload or drain | `_switching`, `_switch_draining`, a live `_switch_task` or `_unload_task` | `model_delete_busy` |
| Counted model requests | `_model_requests > 0`, no drain (the page retries) | `model_delete_busy` |
| Check fetching a tokenizer from any repository the plan touches (E's group and every shared repository, such as SDXL base or Anima-Base; the fetch outlives its request) | **(new)** `prompt_encoding.pending_sources()` over `_loading` | `model_delete_busy` |
| A scan running | `_scan_lock.acquire(blocking=False)` fails | `model_delete_busy` |
| Gallery maintenance | `GalleryAdmission` already counts `/api/models/...` | `gallery_maintenance` (existing) |
| A download by another process | `*.incomplete` under a touched repository | `model_downloading` |

**Selected is refused.** *Why (draft B allowed it with a warning): one rule
then closes every fetch outside a load (Check's tokenizer thread, the LoRA
denoiser read through `loading.hub_file`) and a silent re-download at
start with "Load automatically on startup".* Matt can still delete any
checkpoint: choose another first (question 1).

**Loaded is never deleted, a sharer's files included.** A bf16 load keeps
every shard memory-mapped for the pipeline's life
(`loading.checkpoint_files_released` copies tensors out only while
quantizing). On Linux the unlink succeeds and frees nothing until unload;
on Windows it fails or frees nothing. A file the loaded entry selects is
kept by the shared-files rule and asserted again in the worker.

### Reservation

The pattern is `_unload_weights` / `_release_owned`.

1. **(new)** `main._deleting_model: str | None`, holding the entry's
   label. *Why not reuse `_switching`: `/api/status` reports it, and the
   page would say "switching".* The readers of `_switching` (from
   `grep -n "_switching\b" backend/main.py`) read it too:
   - `hold_model_for_request` refuses counted model routes in its own
     words ("a checkpoint's files are being deleted; wait"), so Generate,
     a load and Check cannot start;
   - `switch_model`, `_unload_weights`, `_idle_unload_busy`;
   - the bundle manager's `lambda: _switching` becomes
     `lambda: _switching or _deleting_model is not None`;
   - `_busy_switching` stays as it is; **(new)** `_busy_deleting` joins
     `KEEP_AWAKE_PROVIDERS` ("deleting a checkpoint");
   - the `/api/status` snapshot, which reads `_switching`, adds
     `model_delete: <label> | null`, so the page needs no new timed poll
     (none joins `_POLLED_GETS`).

   `_refuse_busy_switch` does not read `_switching` (it checks the load
   lock, `_load_state`, `jobs.busy()` and `jobs.card()`); it gains a
   **(new)** `_deleting_model` check of its own.
2. **(new)** `_delete_task`, awaited through `asyncio.shield`, so a caller
   who leaves does not stop it; the lifespan gathers it as it does
   `_unload_task`.
3. The worker runs through `detached(model_cache_delete.delete,
   "model-cache-delete")`, takes `_scan_lock` non-blocking for its whole
   run, and holds `keep_awake.busy("deleting a checkpoint")`.
4. It re-plans, compares the pin, executes, re-scans the touched paths.
5. `finally`: release `_scan_lock`, `model_info.invalidate_cache_facts()`
   (clears `_cached_state` and bumps `cache_revision`, which
   `download_progress._sample`'s lru_cache keys on, so its old entries are
   bypassed, not cleared), clear `_deleting_model`.

### Shared files

- **What "shared" means.** One file, in one repository at one revision,
  that several entries' recipes select (`owners[(repo, commit, file)]` in
  `model_cache_storage._snapshot`). On Matt's cache (Windows,
  no symlinks, `blobs/` empty, no shared-store marker) nothing is shared
  between repositories or revisions. His two cases:
  - `circlestone-labs/Anima-Base-v1.0-Diffusers@073c3a9` (11 files,
    1.4 GiB): Anima Turbo, Aesthetic, Preview;
  - `stabilityai/stable-diffusion-xl-base-1.0@4621659` (17 config and
    tokenizer files, 3.0 MiB): Illustrious, WAI, RouWei.
- **On a symlink cache** a repository's `blobs/<etag>` can serve several
  revisions. A blob goes only when no surviving snapshot entry resolves to
  it (`_identity`).
- **Last-holder rule.** A shared file is kept while another sharer is
  *present*: it has its own exclusive Hub bytes, other-revision bytes
  (GEN#71's `other_revision_bytes`), or a converted tree on disk; or it is
  the selected entry, the loaded entry, the checkpoint saved for the next
  start, or has a tokenizer fetch pending in `prompt_encoding`. *Why: a
  selected Illustrious with nothing downloaded still reads its Check
  tokenizers from SDXL base (`encoder_table._sdxl_single_file`), and its
  next load or conversion needs them.* Otherwise it goes with this delete and the dialog says so ("also
  removes 3.0 MiB shared with Illustrious, which has nothing else on
  disk"). *Why (draft A kept them until a later question): deleting WAI,
  RouWei and Illustrious in turn would otherwise strand SDXL base for
  good.* The kept files matter: Check reads WAI's, RouWei's and Anima's
  tokenizers from them once a converted tree is gone, and a reconversion
  needs them.
- **Huggingface_hub 1.33's shared Xet store** (cross-repository, marker
  `blobs/.huggingface-shared-blobs`) is held with `model_delete_held` until
  designed for. Matt's cache has none.
- **No library deletes.** `HFCacheInfo.delete_revisions` works per
  revision (the whole `models--org--name` folder only when no other
  revision remains), while the plan needs per file: a shared revision such
  as SDXL base or Anima-Base must keep a sharer's files.
  `DeleteCacheStrategy.execute()` swallows `PermissionError` and
  `FileNotFoundError` (`_try_delete_path`), returns nothing to report, and
  coordinates with none of the app's locks (it locks only a shared Xet
  store's sweep, not the repository files). The app runs its own
  file-level plan from the same scan.

### How files are removed

**Converted tree:**

1. Take `WeakFileLock(cache_root() / f"{target.name}.lock")` with a short
   timeout; a timeout is a conversion running: `model_delete_busy`.
2. `os.rename(target, <target.name>-<8 [a-z0-9]>/tree)`: exactly
   `single_file._is_work`'s shape, so the tree stops being a hit
   atomically. A failed rename changed nothing: `model_files_in_use`.
3. `rmtree` the work directory. A failure midway leaves a work directory
   that `sweep_work` and the next delete remove, never a marker-less target
   ("Incomplete converted model cache; remove ... and retry", a refused
   load until removed by hand).
4. Sweep E's other work directories under the lock, then release it.
   The `.lock` file stays: it is zero bytes, the loaders recreate it, and
   unlinking a lock another process may take after our release would let
   two holders (one on an unlinked inode on Linux) run `sweep_work` over
   each other's work directory; on Windows the unlink fails while held.
   The inventory ignores `*.lock`.

**A repository E owns outright:**

1. Confirm every revision under `snapshots/` is E's and no other recipe
   names the repository; refuse on `*.incomplete`.
2. `os.rename(models--org--name, <hub>.auradream-deleting/<8hex>/...)`
   **(new)**: a sibling of the hub directory, outside `scan_cache_dir`'s
   view. *Why not inside the hub (draft A): the scan reports an unknown
   folder there as a corrupted repository.* On Windows an open file inside
   fails the rename before anything changes; a cross-device rename (a hub
   that is a mount point) is refused the same way.
3. `rmtree` the trash. Leftovers are swept at the next delete and named by
   the inventory as a warning.

**Files inside a repository others use (stage 4):**

1. For each revision being emptied, unlink its `refs/*` first, so it
   becomes detached rather than a ref to a missing snapshot (which the
   scan calls corrupted).
2. Unlink each snapshot entry, then each repository blob no surviving
   snapshot resolves to. A `PermissionError` stops at once and lands in
   `remaining`.
3. Remove emptied directories bottom-up, then the revision's
   `.no_exist/<commit>`.

A partial stage-4 state is always "files missing from a pinned revision",
which the next load fills by download, never a refused load.

### Windows, WSL and other processes

- Whether a mapped safetensors file blocks a rename on Windows is
  **unverified** (memmap2's handle may share delete). Refusing the selected
  entry is the guard; the rename-first order is defence in depth. Measuring
  it is the card's to say and is not a stage here.
- Each server deletes only from its own `HF_HOME`. A delete on Windows
  leaves the WSL and Linux caches alone, and the dialog says "this PC's
  cache" when the entry has lite parts.
- A CLI `generate` or a second server on the same `HF_HOME` is outside
  every in-process lock. The `.incomplete` check and the Windows rename
  refusal catch most of it; the docs say "stop other runs using this cache
  first".

### Licences

A delete removes local copies only. It does not withdraw a Hub acceptance,
revoke a token, or touch images made under the licence. A switch back goes
through `hub_access.check`: with the access gone, the Hub's refusal stops
the switch before anything is unloaded. The xet chunk cache can still hold
chunks; the docs say where it is and that `hf` clears it (question 7).

### Naming the Nunchaku row

The bare id appears because the page labels rows from `/api/models`'
`models`, and `model_selection.available()` lists lite ids only for
Nunchaku on Linux. The fix is on the server:

- Each storage `models[]` row gains `entry` (the official id from
  `launch.catalogue_entry`), `label` (`entry.name`) and `variant`
  (`"nvfp4"`, `"int4"` or null, from `Checkpoint.lite_format`).
- **(new)** `_snapshot` maps each cached `lite_alias(fmt)` repository to
  its written lite id (`launch.written_id`) before classifying: an alias
  has no recipe, so adding it to `_catalogue_model_ids` alone would leave
  it out of `recorded`/`repo_models` (other revisions are computed only
  for repositories in `recorded`), its files in `unreferenced_blobs`, and a
  bogus `reference_complete: false` row. The alias repository's revisions
  count as other revisions of the written id's row (its
  `other_revision_bytes`), never as a model row of their own;
  `tools/model_downloads.py --check` still keeps aliases free of recipes.
  A storage test caches an alias repository and asserts its bytes leave
  `unreferenced_bytes`.
- `modelStorageModels` (`web/model-storage.js`) prefers `row.label` and
  shows variant rows as sub-lines of their entry: "Krea 2 Turbo, Nunchaku
  NVFP4 files: 10.9 MiB". One Delete per entry.

This is stage 0 and useful without the delete.

## Page

- **Where.** `web/model-storage.js`: a trailing **Delete…** per entry row,
  built with `modelStorageCell`. The fold is last in the body, so the
  control moves nothing above it. CSS uses the `:root` tokens in the
  fold's section; the phone keeps 44 px under `(pointer: coarse)` in
  `styles/touch.css`. No wide-tier change.
- **Held.** `aria-disabled` with the reason as its title, never
  `disabled`; a press says why. **(new)** `modelStorageDeleteHeld(row)`
  answers the reason: selected, loaded, a job, load, switch, card holder,
  scan or delete running, gallery maintenance, or the plan's `held`.
- **Press.** **(new)** `deleteModelStorage(row)` reads the plan, then
  `askConfirm(message, {okLabel: 'Delete checkpoint', destroy: true})`
  (`web/ask-dialog.js`'s `confirmDialog`). The first line is the title, as
  a bundle's Delete:

  ```
  Delete Anima Aesthetic from this PC?
  Frees about 7.2 GiB: downloaded files 1.9 GiB, converted copy 5.3 GiB.
  Kept: 1.4 GiB shared with Anima Turbo, Anima Preview.
  Gallery images, setups and favourites stay. Using it again downloads it
  again, which needs your Hugging Face access to its licence.
  Other apps using this Hugging Face cache lose these files too.
  This cannot be undone.
  ```

  ERNIE's adds "Also removes its prompt rewriter (Expand)".
- **After the dialog.** Re-check the held state, as `bundles.js` does
  ("anything that started while the question was up holds it now"), then
  POST with the pin. While it runs, every Delete is held with "A checkpoint
  is being deleted" and the fold's status line reads "Deleting Anima
  Aesthetic…" from `/api/status`'s `model_delete`.
- **Refusals by code**, through **(new)** helpers in `web/state.js` beside
  `imageGone`: see the codes table.
- **Afterwards.** One `loadModelStorage({fresh: true})` (a consequence of
  the press, not a timer) and `refreshModelCacheFacts()`; the cache word
  follows `cache_revision`. The deleted entry's row stays in the table as
  its result line ("Deleted Anima Aesthetic: about 7.2 GiB freed"), with
  no control, across that re-read and every later one until the fold is
  closed or Refresh is pressed: `modelStorageModels` drops rows with no
  bytes, so without it every row below would move up and the next row's
  Delete… would land under the pointer. The page never scrolls.
- **Read-only wording to change:** `web/model-storage.js`'s header comment
  and its "Read-only: nothing here deletes files." line, the fold's
  comment in `web/index.html`, and the `note` in
  `backend/model_cache_storage.py`. "Not referenced" still never means
  unused.

## Tests

**CPU, `tests/test_model_cache_delete.py` (new).** The `inventory` fixture's
approach from `tests/test_model_cache_storage.py` (scratch hub and converted
root; `model_info.download_references`, `_catalogue_model_ids`,
`model_info._conversion_target` and `hf_home_dir` patched), in a plain-copy
Windows layout and, where the platform allows, a symlink layout:

- Plans: the Anima three (base kept with sharer names; Aesthetic's delete
  leaves Turbo's revision byte for byte); the SDXL three and the
  last-holder rule both ways; an owned repository whole (old revision,
  `.no_exist`, unselected files, ERNIE's `pe/`); GEN#71's other revision;
  an ambiguous `other_revision_of` held; lite rows and an alias grouped;
  the shared-store marker held; an `.incomplete` refused; an older recipe
  with and without a marker; an older Turbo tree beside an Aesthetic
  delete, kept; the last-holder rule with the selected sharer, a sharer
  with only other-revision bytes, and a pending tokenizer fetch, each
  keeping the shared files.
- Containment: the patched cache root differs from the real `HF_HOME`,
  and nothing outside the patched roots changes (a sentinel tree under a
  second scratch "real" root survives every plan and delete).
- Execution: every file outside the plan byte for byte; the critic and a
  non-catalogue repository untouched; `snapshot()` afterwards shows E with
  no bytes and no new warnings; the pin stable across two scans of an
  unchanged cache, and a file added between plan and delete refused with
  nothing removed.
- Failure injection (`os.rename`, `os.unlink`, `shutil.rmtree` raising
  `PermissionError` with winerror 32 at each step): a converted or Hub
  rename failure removes nothing; an `rmtree` failure leaves only what
  `_is_work` accepts; a stage-4 mid-unlink failure leaves no ref to a
  missing snapshot and reports `remaining`; the trash sweep.
- A converted lock held by another `WeakFileLock` refuses.

**API, `tests/test_api_model_storage_delete.py` (new):** each guard by
code (selected, loaded, `jobs.busy`, card holder, the load lock and the
reserved "loading" state, switching, counted requests, a pending tokenizer,
a scan, gallery maintenance); `_deleting_model` refusing Generate, load,
switch, unload and a bundle, and showing in `/api/status`; the keep-awake
provider; a disconnected caller still finishing and releasing;
`invalidate_cache_facts` bumping `cache_revision`; a cross-site POST 403;
the same 404 for an unknown and an absent entry; no absolute path in any
answer. `tests/test_refusal_codes.py` covers the new codes.

**Passcode walk.** `tests/test_api_passcode_routes.py`'s
`scratch_cache_scan` fixture also stubs `model_cache_delete.plan` and
`delete` to raise, so the route walk can never reach a real cache.

**Fake and smoke.** `tools/fake_server.py`'s `_plant_model_cache` also
plants an Anima-like shared base under two entries, one converted tree and
Krea 2 Turbo's nvfp4 lite tokenizer files. *Today the fake patches only
`model_cache_storage.hf_home_dir`; converted targets resolve through
`single_file.cache_root()` -> `legacy_names.hf_home_dir`, which reads the
real `HF_HOME` (neither the fake nor `tests/conftest.py` sets it), so a
delete resolving its target, lock or work directories that way would reach
the developer's real converted copies.* So **(new)**: the fake, and
`test_model_cache_delete`'s fixture, set `HF_HOME` and `HF_HUB_CACHE` to
scratch directories for the whole process before any backend import, and
`model_cache_delete` resolves every path (Hub repository, converted
target, work directories, lock, trash) once and asserts each is inside
the configured roots before the first rename, refusing otherwise with
nothing changed. The proposed row (the ports are full, so it shares
7959 with its own fresh `--data`, one fake at a time in the band):

| Port | `--model` (default Krea 2 Turbo) | Smokes |
|---|---|---|
| 7959 | FLUX.1-Krea-dev, its own fresh scratch `--data` | `smoke-model-delete` |

It covers: the lite row named under Krea 2 Turbo; Delete held on the
selected row with its reason; the dialog's frees and kept lines and
sharers; confirm, the in-place result and one re-read, the sharer intact, the next
row's Delete… keeping its bounding box across the re-read;
the Models cache word changing; a busy fake (`/api/fake/step` job) holding
the button; a changed cache between plan and confirm re-asking; the a11y
watch and check (listed in `tests/test_a11y_audit.py`); the phone fit with
`phone_audit.js`.

`tools/smoke-model-storage.json` changes in the same commit as the
control: its `__controls().length === 1`, no-`/delete|remove|free up/i` and
"Read-only: nothing here deletes files." steps expect the Delete controls
and the new wording.

## Docs to update

| File | Change |
|---|---|
| `docs/rules.md` | The module map (`api_model_storage.py`/`model_cache_storage.py` lose "no delete mechanism"; add `model_cache_delete.py`); the storage rules (unit, last holder, guards, `_deleting_model` readers, rename-first order, never-touched list); the page's `model-storage.js`; Tests (the new smoke). |
| `docs/design-notes.md` | Beside the GEN#55/GEN#71 paragraph: why file by file and not `delete_revisions`, rename first, last holder, selected refused, the Windows mapping question. |
| `docs/usage.md`, `docs/setup.md` | The button; what stays; re-download and licence access; the xet chunk cache; the manual `hf cache rm` route kept for other repositories and other caches. |
| `AGENTS.md` | The smoke table row; one line under "Reproduction and storage": a checkpoint delete reserves the idle server and the scan lock, never touches the selected, loaded or saved-for-start entry, keeps files a present sharer selects (the selected, loaded and saved ones count as present), and never names `hf cache prune`. |
| `docs/maintainer/review-2026-11-10-peers.md` | R9: a line pointing here. |
| `TODO.md` | `GEN#72`'s pointer to this plan; what is left after each stage. |

## Stages and completion criteria

Riskiest last. Each stage is one or more commits on one branch.

0. **Names and grouping, no delete.** Rows carry `entry`, `label` and
   `variant`; `_snapshot` attributes alias repositories to their written
   id's row; the page groups lite rows.
   *Done when* storage tests cover a lite and an alias row and
   `smoke-model-storage` reads the lite row by its entry.
   *Built 2026-11-14:* `model_cache_storage._lite_aliases`/`_row_names`;
   `tests/test_model_cache_storage.py`'s lite-and-alias test; the fake
   plants Krea 2 Turbo's NVFP4 tokenizer files and its INT4 id under the
   renamed repository, and the smoke reads both lines under the entry's
   row (shown with dashes when the entry itself holds nothing).
1. **Planner, read-only.** `_snapshot` keeps an internal per-file index
   (first-hop paths, repository, revision, owners, converted targets)
   behind an unchanged answer; **(new)** `backend/model_cache_delete.py`
   `plan()`; the plan GET; the refusal codes; `_deleting_model` wired into
   every reader (inert).
   *Done when* every planner case above passes, the existing storage tests
   pass unchanged, a byte-for-byte no-mutation test passes, and ruff and
   `tools/typecheck.py` are green.
   *Built 2026-11-14:* `model_cache_storage._scan` (the answer and its
   `CacheIndex`; `_snapshot` returns the answer alone, unchanged) and
   `hub_cache_dir`; `model_cache_delete.plan` (`PlanContext`, `Plan.answer`,
   `fetching`, the pin; `SHARED_REPOSITORY_DELETE` false until stage 4,
   so a removal inside a shared repository is `held`, its parts
   `shared_repository: true`); the GET in `api_model_storage.create_plan_router`
   with `gallery_images` (question 9: one count, `storage.count_live_images`);
   the seven codes (`refusal_codes.MODEL_DELETE_CODES`; the page reads none
   until stage 3); `main._model_delete_refusal`, `_model_delete_context`,
   `_saved_start_entry`; `_deleting_model` in every reader listed under
   Reservation and `/api/status`'s `model_delete`;
   `prompt_encoding.pending_sources` (every fetch thread still running,
   those `unload()` forgot too, not `_loading` alone). Tests:
   `tests/test_model_cache_delete.py` (both layouts; the symlink one runs
   where links can be made, Linux and WSL, and skips on Windows without the
   privilege), `tests/test_api_model_storage_delete.py`,
   `tests/model_cache_fixtures.py` (scratch `HF_HOME`/`HF_HUB_CACHE` for
   the test through the environment and `huggingface_hub.constants`: the
   suite's conftest imports the backend first, so "before any backend
   import" holds for the fake in stage 3). The fake points the plan's scan
   at its scratch Hub (`hub_cache_dir`); its converted targets still
   resolve under the real `HF_HOME` until stage 3 sets the fake's, so a
   plan there holds a converted entry ("outside ... the converted cache")
   without looking at the real tree, and an entry with nothing cached is
   still the 404. Review fixes: the index's hub is the scan's resolved one
   (a linked cache), `_lexically_inside` says two drives are not inside,
   the Xet marker holds only a plan taking from the Hub, and `Plan.names`
   counts a selected, loaded or saved id outside the catalogue (or a case
   variant) by its repository.
2. **Converted-tree removal**, backend only, not exposed.
   *Done when* rename-to-work-shape, lock contention and every injected
   failure leave only `_is_work` directories, and a later load converts
   cleanly in the scratch cache (`tests/test_converter_cache.py` style).
   *Built 2026-11-14:* `model_cache_delete.delete(model_id, pin, context)`,
   the worker stage 3's route will run on a daemon thread: the scan
   reservation and `keep_awake.busy` for its whole run, a fresh plan, and
   before anything changes `DeleteRefused` (a code) for a different pin, a
   selected/loaded/saved entry, a fetch it touches, a partial download, a
   held part, or any Hub or lite part until stage 3 (`HUB_DELETE`), then
   `assert_inside_roots` (the configured roots read again; every removal,
   unit, converted target and lock inside them). `remove_converted` takes
   each converted part under `WeakFileLock(<root>/<target>.lock,
   timeout=LOCK_TIMEOUT)` (an older tree's own lock too), re-reads the tree
   file for file under it, renames it into a new
   `<target>-<8 [a-z0-9]>/tree` and removes every work directory of the
   target. Choices left open here: an older recipe's tree is renamed into
   its entry's *current* target's work shape, so the entry's next load
   sweeps a leftover no loader would otherwise name; a failure after
   something changed is the `Outcome`'s `remaining`, a refusal only while
   nothing has; `freed_bytes` measures the roots' distinct volumes;
   `model_info.invalidate_cache_facts` runs in the worker's `finally`.
   filelock removes a `.lock` on release on Windows by itself; the delete
   never unlinks one. Tests: `tests/test_model_cache_delete_converted.py`.
3. **Whole-checkpoint delete for entries with no shared part.** The trash
   rename, the POST, the reservation, the page control, the dialog, the
   smoke and the docs. Entries with a shared part show Delete held with
   `model_delete_held` and the sharers' names. On Matt's list the plan, not
   this note, decides which; likely FLUX.1 Krea [dev], Krea 2 Turbo with its
   lite rows, Krea 2 Raw, NetaYume, NewBie, Animagine, Kohaku.
   *Done when* the API and CPU tests pass, `smoke-model-delete` and
   `smoke-model-storage` pass under `run_smoke.py --rerun-failed-alone`
   with no page error and a clean a11y check, and `pytest -n auto` is green.
   *Built 2026-11-14:* `HUB_DELETE` on; `model_cache_delete.remove`
   (every part renamed before any is removed: the converted trees under
   their lock, then every whole repository of the group into
   `model_cache_storage.trash_dir(hub)/<8 hex>/`; a refused rename
   anywhere puts back what moved, so `model_files_in_use` and
   `model_downloading` still mean nothing changed), `sweep_trash` at each
   delete's start (its own `<8 hex>` folders only) and the inventory's
   warning for a leftover; `Plan.removes_expander` for ERNIE's line; the
   POST in `main` (`delete_model_storage`, `_run_model_delete`,
   `_delete_task`, `_model_delete_finished`; `ModelDeleteRequest`), which
   also compares the pin on the loop before reserving. Page:
   `web/model-storage.js` (`modelStorageDeleteHeld`, `deleteModelStorage`,
   the result line keeping the row and its sub-lines,
   `receiveModelDeleteStatus` from the status poll), the five helpers in
   `web/state.js`, the picker kept open while its question is up
   (`web/app.js`), the phone's one-grid-per-line table in
   `styles/touch.css`. Choices where this plan left room: an entry with
   kept shared files left and nothing of its own still shows a row (the
   inventory counts the shared files under it), and its Delete… says there
   is nothing to free instead of asking; a busy refusal holds every
   Delete… until the next status; the inventory's rows carry each plan's
   hold (`delete_held`, `model_cache_delete.annotate_holds` under the same
   scan), so a shared part's row is held as the fold reads, and a hold a
   press meets that the read did not holds its row until Refresh;
   "this PC's cache" is said in the title and a line of its own. The fake
   sets its scratch `HF_HOME`/`HF_HUB_CACHE` in `main()` before any backend
   import for every fake, and plants the delete's fixtures only through a
   knob (`POST /api/fake/model-cache`, `plant`; `touch` changes a
   repository), so the other smokes' Models cache words do not change.
   Tests: `tests/test_model_cache_delete_hub.py`, the POST's in
   `tests/test_api_model_storage_delete.py`.
4. **File-level delete inside shared repositories.** Anima Turbo and
   Aesthetic (one repository, two commits), Anima Preview, Illustrious, WAI
   and RouWei, with the last-holder rule.
   *Done when* the stage-4 failure tests show no ref to a missing snapshot
   and sibling revisions byte for byte, in both layouts where the platform
   allows, and the smoke deletes one of two sharers leaving the other's
   row intact.
   *Built 2026-11-14:* `SHARED_REPOSITORY_DELETE` on (kept as a switch, as
   `HUB_DELETE`); each shared part (one per repository and revision) names
   its `entries`, `blobs` and, when the delete empties its revision
   (`emptied`), the `refs` naming it, which are removals the pin covers; a
   shared repository's folder that is a link is held. `remove()` takes the
   converted and whole-repository renames first, then `_check_files` (no
   partial download, every planned entry, blob and ref as planned, an
   emptied revision holding only the planned entries and refs) and
   `_unlink_files`, one repository at a time: refs read now, entries,
   blobs no surviving link names now, then `_prune_repository` (empty
   folders bottom-up, the snapshot folder only while no ref names it,
   `.no_exist/<commit>`, a nested ref's empty folders, the repository's
   folder once empty). A check that fails or a first unlink refused puts
   every rename back with nothing changed; a later refused unlink stops at
   once, prunes what it emptied, and names the stopped repository's
   touched parts and every later one in `remaining`. Choices where this
   plan left room: refs are found again on disk at the unlink, and a ref
   that appeared since the plan refuses the check (`model_cache_changed`);
   unselected files at the entry's own commit in a shared repository stay
   (question 4 takes them only where the repository is the entry's alone),
   so such a revision is not emptied and keeps its refs and notes; a blob
   left by a stop (at it, or at an entry after its own entries went) is a
   `remaining` row of its own saying the bytes stay unnamed in `blobs/`,
   which no later plan can see, to remove by hand; the page keeps a row the
   last holder's delete emptied (`also_taken_shared`) as "Its files shared
   with … went with it.", so no row below moves, but only when nothing
   stayed: after a stop the re-read shows the sharer as the cache holds it,
   its Delete… free. Tests:
   `tests/test_model_cache_delete_shared.py` (both layouts; the symlink one
   run under WSL), the POST's in `tests/test_api_model_storage_delete.py`;
   `smoke-model-delete` deletes Anima Turbo beside Preview, then Preview as
   the last holder.

**Matt's acceptance (no GPU, after stage 3 and again after 4).** He deletes
one real entry of his choice on Windows with another selected, compares the
measured and planned bytes, sees Models say not downloaded, and presses
Delete on the selected row to see it held; after stage 4, one Anima variant,
with the other two still loading. The result goes in `docs/verified.md`.

## Matt's questions

**Answered 2026-11-14:** Matt accepted every recommended answer below, so
the plan stands as written.

1. **Can the selected checkpoint be deleted?** *Recommended: no;* held with
   "Choose another checkpoint first". Otherwise the next Generate, or a
   start with "Load automatically on startup", downloads it again, and
   Check's tokenizer thread can fetch it outside any lock. Deselecting it
   for him would change saved choices unasked.
2. **Do Krea 2 Turbo's Nunchaku lite files go with the entry?**
   *Recommended: yes,* listed in the dialog. On Windows they are the
   10.9 MiB tokenizer fetch; the Nunchaku weights live in the WSL or Linux
   cache, which this delete never touches.
3. **Shared files when no other sharer has anything on disk** (the last of
   the SDXL three)? *Recommended: remove them with the last holder,* named
   in the dialog, so nothing is stranded.
4. **Old revisions, unselected files and older converted recipes of an
   entry** (old downloads, README, ERNIE's `pe/` rewriter)? *Recommended:
   include them* when the repository is E's alone, or the tree's marker
   names E's source; otherwise they stay "not referenced".
5. **Confirm style.** *Recommended: one confirmation* with the GiB figures
   and "This cannot be undone", not a typed count: the act is rare and
   already two presses on a named row.
6. **FLUX 3's 18.8 GB upsampler when the last FLUX 3 entry goes.**
   *Recommended: never automatic;* it stays under Other repositories and
   can get its own row later.
7. **Clear HF's xet chunk cache for an ended licence?** *Recommended: no
   app control;* it is not attributable per model. The docs say where it
   is and how `hf` clears it.
8. **A converted-copy-only delete** (48.3 GiB on his machine; reconverting
   is cheap with the source present)? *Recommended: not in GEN#72;* a
   separate row if wanted after stage 3.
9. **Name gallery images made with it in the dialog** ("37 images used it;
   Re-render downloads it again")? *Recommended: yes* if it is one count by
   model id on the gallery; otherwise leave it out.
