# Pyright basic baseline - 2026-10-26

`DEV#6`, stage 1. The first complete static check found **334 errors, zero
warnings and zero information diagnostics in 104 files**, with findings in
52 files. Checker time was 10.045 seconds (10.4 seconds wall time), exit 1.
This is a failing baseline to work down, not an enabled CI gate or a claim
about GPU behavior. No application module, model weights or tokenizer was
loaded to produce it.

## Reproduce

The committed [configuration](../pyrightconfig.json) checks all Python
files under `backend/` and `tools/`, in basic mode for Python 3.13. The
`All` platform target checks conditional platform paths consistently;
the measured host was Windows, Node 24.14.0 and Python 3.13.15. Library
imports resolve from the repository's `.venv`; there are no diagnostic
overrides, module exclusions or new inline suppressions. No existing
`type: ignore`, `pyright:` or `@no_type_check` directives were found in
the configured source files at this baseline.

Use **Pyright 1.1.407**, the native npm package, without changing the app's
environment or adding a global tool. From the repository root, after
preparing the normal locked app/development environment:

```powershell
npm.cmd --cache work/dev6-npm-cache install --prefix work/dev6-pyright --no-audit --no-fund --ignore-scripts --save-exact pyright@1.1.407
node work/dev6-pyright/node_modules/pyright/index.js --version
node work/dev6-pyright/node_modules/pyright/index.js --project pyrightconfig.json --outputjson > work/dev6-pyright-baseline.json
```

On Linux the same commands use `npm` in place of `npm.cmd`. That platform
has not been run for this baseline. Installation requires registry access;
the checker itself reads local source, library source and stubs. Preserve
the checker's exit status when collecting JSON: exit 1 is expected until
the remaining errors are fixed. The initial machine-readable output is
retained locally at `work/dev6-pyright-baseline.json`; the tables below
retain its complete counts. The isolated npm package lock records the
registry integrity for `pyright@1.1.407` as
`sha512-zU+peTFEVUdokNQyUBhGQYt+NWI/3aiNlvBbDBSsn5Ti334XElFUs+GDjQzCbchYfkT+DvMAT3OkMcV4CuEfDg==`.

The source snapshot contained `GEN#26`'s fix (`1d2b350`) and the portable
runner work subsequently committed in `a158b69`. Subsequent fixes should
record their new counts separately rather than rewrite this initial
measurement.

The installed dependency versions used for import resolution were:

| Dependency | Version |
|---|---|
| torch | 2.13.0+cu130 |
| diffusers | 0.40.0 |
| transformers | 5.16.1 |
| torchao | 0.18.0 |
| accelerate | 1.14.0 |
| huggingface-hub | 1.29.0 |
| pydantic | 2.13.5 |
| fastapi | 0.141.1 |
| Pillow | 12.3.0 |
| numpy | 2.5.2 |

Pyright's [versioned configuration reference](https://github.com/microsoft/pyright/blob/1.1.407/docs/configuration.md)
defines these options. Basic mode leaves unknown-type and missing-stub
checks less strict than strict mode; a clean future basic check will not
establish complete third-party typing coverage.

## Triage and fix order

These are bounded investigation batches, not permission to mask a
diagnostic. Their counts partition the initial 334 errors. Split a batch
further where a runtime change needs its own focused regression. Keep
existing CPU/browser checks as independent evidence.

| Batch | Initial errors | Scope and next step |
|---|---:|---|
| 1. Benchmark reference contracts | 9 | `model_benchmarks.py`: eight fixed keyword dictionaries are inferred too broadly for `Sample`; describe their actual keys. Prove or validate the optional Nunchaku format before calling `upper()`. Retain all measured wording and timing values. |
| 2. CLI and exception helpers | 15 | `launch`, `generate`, `main`, `clear`, `stats`, `reindex`: mutable environment mappings, fixed tuple shapes, optional exception-chain state, argument-parser return contracts and stdout capabilities. Handle one helper family at a time. |
| 3. Applied requests and route results | 74 | `grid`, `submission`, `api_generate`, `api_refine_batch`, `api_swap`, `api_more_seeds`, `api_png`, `schemas_generate`, `schemas_gallery`, `schemas_snippets`: express invariants after `applied()`, narrow optional selection/preview results and preserve literal statuses. First separate schema type-expression findings from route narrowing; never replace validation with casts. |
| 4. Hub and checkpoint loading | 108 | `loading`, `anima_loading`, `netayume_loading`, `newbie_loading`, `single_file`, `critic_runtime`, `critic_server`: first give heterogeneous Hub/loader keyword dictionaries precise shapes; then inspect lazy exports and config-return overloads in the installed libraries. Keep pinned revisions and download filtering intact. |
| 5. Pipeline and image interfaces | 63 | `diffusers_backend`, `anima_pipeline`, `newbie_pipeline`, `newbie_transformer`, `newbie_text_encoder`, `schedule`, `refine`, `latent_preview`: handle one pipeline/interface at a time, checking installed signatures and real callback/optional-state requirements. Keep scheduling and pixels unchanged unless a defect is demonstrated. |
| 6. Storage and bundle boundaries | 18 | `storage`, `bundles`, `bundle_archive`, `bundle_images`, `png_settings`, `jobs`: narrow optional database results, byte/file interfaces and job state. Preserve atomic writes, provenance and reservation ordering. |
| 7. Prompt and catalogue helpers | 22 | `api_prompt`, `prompt_rules`, `critic_validation`, `critic`, `guides`, `prompt_check`, `tag_lookup`, `model_info`: narrow optional encoders/results and annotate literal metadata without changing prompt interpretation or guide sources. |
| 8. Developer tools | 25 | All six tool files with findings: `fake_server`, `storage_interrupt`, `storage_scale`, `check_run_continue`, `run_smoke`, `fetch_tokenizers`. Address fixture callback contracts and optional CLI/source descriptions first; keep failure behavior and isolated outputs. |

The largest third-party interface groups need examination, not automatic
suppression:

- Diffusers' lazy top-level exports account for most `reportPrivateImportUsage`
  findings. Its `from_config`/`load_config` return types include optional
  metadata tuples, and Anima's `PipelineState` uses dynamic fields and
  mapping operations that its declared interface does not expose. Check
  each installed definition before choosing a direct import, overload
  narrowing or small local protocol.
- Transformers' lazy `T5TokenizerFast` and `GemmaTokenizerFast` aliases
  appear as unknown imported symbols. Several `from_pretrained` calls
  spread mixed-type keyword dictionaries. Establish the actual aliases and
  keyword shape; do not cast an entire pipeline to `Any`.
- Hugging Face Hub's overloaded download functions receive dictionaries
  inferred as arbitrary `str` values, producing many repeated argument
  errors for boolean, numeric and mapping parameters that the dictionaries
  never contain. `WeakFileLock` is also reported as a private re-export.
- Pillow's legacy `Image.LANCZOS`/`Image.NEAREST` constants and an L-mode
  point callback's broadly typed input require explicit supported
  interfaces. Pydantic's custom-error API expects `LiteralString`, and
  dynamically expanded values in `Literal[...]` are invalid type forms.

There were no `reportMissingImports` or `reportMissingModuleSource`
diagnostics. That does not eliminate the interface gaps above. A future
suppression must name its diagnostic and explain a verified local typing
limitation; none was needed to establish this baseline. Add the checker to
the development group and CI only after the full configured scope passes,
then verify a missing checker or empty scope fails the gate.

## Completed batches

2026-10-26, batch 1: `model_benchmarks.py` now describes its four fixed
Nunchaku keyword fields with a `TypedDict` and explicitly checks the
recorded sample's format before formatting it. Timings and selection
behavior are unchanged. Independent review verified that all table entries
satisfy the invariant. The 21 existing benchmark tests and focused ruff
pass. The complete static check now reports **325 errors in 104 files**,
zero warnings: precisely the nine benchmark diagnostics were removed,
with no new file/rule counts. JSON evidence is retained locally at
`work/dev6-pyright-after-benchmarks.json`; exit 1 remains expected.

2026-10-26, batch 2: the six CLI/launcher/lifecycle modules now describe
read-only environment mappings, fixed outpaint tuple shapes, load-state
fields and PNG metadata keys precisely. Stdout reconfiguration still
supports compatible custom streams, `StringIO` and absent output. The
274 existing CLI/storage/API/lifecycle checks passed; after correcting a
ruff B009 finding in the draft, the 13 relevant stdout/outpaint tests and
six-file static check passed. Independent review found no blocking issue.
All 15 initial batch diagnostics are resolved without suppressions.
The final scoped JSON is `work/dev6-cli-final.json`.

2026-10-26, batch 7: prompt/catalogue helpers validate the same dictionaries
at their existing Pydantic response boundaries, express optional
invariants and preserve matched tag text/indices. The critic server's
`_spawn` return annotation now names the existing `_ServerProcess`,
preserving its process-tree contract. Guide prose and citations are
unchanged. All 488 existing focused tests, the scoped static check and
ruff passed; independent review found no actionable issue. All 22 initial
batch diagnostics are resolved without suppressions or casts. Scoped
JSON evidence is `work/dev6-prompt-helpers.json`. These are scoped results;
the full baseline remains failing until the other batches are fixed.

2026-10-26, batch 3 schema/PNG slice: all 17 initial findings in the three
schema modules and PNG-settings route are addressed. The fixed upscaler
literal is shared from the leaf module, optional applied values are
narrowed, error-detail callbacks retain their existing duck-typed contract,
and PNG size keywords name only width/height. All 31 JSON schemas and the
custom error code/template/context match HEAD before this batch; persisted
snippet targets remain required. The 245 existing schema/refine/snippet
tests and 17 PNG API tests passed, with clean scoped static checks and
ruff; both slices received independent review.

Two line-specific exceptions document verified Pydantic typing gaps:
`reportArgumentType` for the internal integer-based custom error code,
and `reportGeneralTypeIssues` for removing an inherited field default to
make a persisted snippet field required. Neither changes runtime
validation or suppresses a module/rule globally. Evidence is retained in
`work/dev6-schema-contracts.json`, `work/dev6-schema-equivalence.json` and
`work/dev6-png-size.json`.

2026-10-26, batch 4 keyword slice: the four checkpoint-converter modules
now name their actual Hub and recipe dictionary keys, and pass the same
explicit local-only flag. All pins, checksums, filters and serialized
recipe/cache values are unchanged. Their diagnostics fell from 64 to 24
(40 keyword/call findings removed), with no suppressions or casts.
The 111 existing mocked loader tests, focused ruff and independent review
passed. Remaining library-export/model-interface findings are a separate
slice. Receipts: `work/dev6-hubkw-before.json` and
`work/dev6-hubkw-after.json`.

2026-10-26, batches 6 and 8: storage/bundle/job boundaries resolve all
18 initial findings across six checked modules (five changed); developer
tools resolve all 25 findings across six files. Optional guards encode
existing branch invariants, row/mask/file interfaces describe their
actual inputs, and the resolved job run id retains its existing value.
Tool environment setup still precedes app imports; fake events, process
captures, streaming ZIP reads and tokenizer tables have precise contracts.
No suppressions or casts were added.

The 237 existing storage/bundle/job tests and 63 tooling/docs-map tests
passed, with clean scoped checks and ruff. Independent reviews found no
actionable issue. Additional bounded fixtures checked synthetic fake
state, mocked tokenizer snapshots/missing tables, a tiny ZIP hash and
mocked controller start/stop; no storage scale/interruption benchmark or
live process/model run was used. Receipts:
`work/dev6-storage-after.json`, `work/dev6-tools-after.json` and
`work/dev6-tools-fixture-result.json`.

## Initial counts by diagnostic

<!-- Counts below are generated from the retained initial Pyright JSON. -->
| Diagnostic | Errors |
|---|---:|
| `reportArgumentType` | 176 |
| `reportAttributeAccessIssue` | 34 |
| `reportPrivateImportUsage` | 34 |
| `reportOptionalMemberAccess` | 28 |
| `reportCallIssue` | 18 |
| `reportOptionalSubscript` | 10 |
| `reportGeneralTypeIssues` | 8 |
| `reportOperatorIssue` | 5 |
| `reportReturnType` | 5 |
| `reportAssignmentType` | 4 |
| `reportInvalidTypeForm` | 4 |
| `reportOptionalCall` | 3 |
| `reportOptionalIterable` | 3 |
| `reportOptionalOperand` | 2 |

## Counts by module

Every configured source file appears below, including zero-error files.

| Module | Errors |
|---|---:|
| `backend/__init__.py` | 0 |
| `backend/api_bundles.py` | 0 |
| `backend/api_gallery.py` | 0 |
| `backend/api_generate.py` | 15 |
| `backend/api_jobs.py` | 0 |
| `backend/api_masks.py` | 0 |
| `backend/api_more_seeds.py` | 4 |
| `backend/api_png.py` | 7 |
| `backend/api_prompt.py` | 7 |
| `backend/api_refine_batch.py` | 6 |
| `backend/api_snippets.py` | 0 |
| `backend/api_swap.py` | 4 |
| `backend/app_context.py` | 0 |
| `backend/backends/__init__.py` | 0 |
| `backend/backends/anima_loading.py` | 18 |
| `backend/backends/anima_pipeline.py` | 11 |
| `backend/backends/base.py` | 0 |
| `backend/backends/diffusers_backend.py` | 27 |
| `backend/backends/loading.py` | 21 |
| `backend/backends/netayume_loading.py` | 29 |
| `backend/backends/newbie_loading.py` | 14 |
| `backend/backends/newbie_pipeline.py` | 8 |
| `backend/backends/newbie_text_encoder.py` | 1 |
| `backend/backends/newbie_transformer.py` | 6 |
| `backend/backends/schedule.py` | 5 |
| `backend/backends/single_file.py` | 3 |
| `backend/bundle_archive.py` | 3 |
| `backend/bundle_common.py` | 0 |
| `backend/bundle_crypto.py` | 0 |
| `backend/bundle_images.py` | 1 |
| `backend/bundles.py` | 5 |
| `backend/clear.py` | 1 |
| `backend/config.py` | 0 |
| `backend/critic.py` | 1 |
| `backend/critic_context.py` | 0 |
| `backend/critic_reorder.py` | 0 |
| `backend/critic_runtime.py` | 15 |
| `backend/critic_server.py` | 8 |
| `backend/critic_validation.py` | 2 |
| `backend/detached.py` | 0 |
| `backend/encoder_table.py` | 0 |
| `backend/gallery_delete.py` | 0 |
| `backend/gallery_lock.py` | 0 |
| `backend/gallery_reports.py` | 0 |
| `backend/generate.py` | 3 |
| `backend/grid.py` | 13 |
| `backend/guides.py` | 2 |
| `backend/hub_access.py` | 0 |
| `backend/jobs.py` | 2 |
| `backend/latent_preview.py` | 1 |
| `backend/launch.py` | 4 |
| `backend/loader_choices.py` | 0 |
| `backend/loras.py` | 0 |
| `backend/main.py` | 4 |
| `backend/metadata.py` | 0 |
| `backend/model_benchmarks.py` | 9 |
| `backend/model_defaults.py` | 0 |
| `backend/model_info.py` | 1 |
| `backend/model_selection.py` | 0 |
| `backend/more_seeds.py` | 0 |
| `backend/more_seeds_intents.py` | 0 |
| `backend/outpaint.py` | 0 |
| `backend/png_settings.py` | 1 |
| `backend/prompt_check.py` | 2 |
| `backend/prompt_encoding.py` | 0 |
| `backend/prompt_order.py` | 0 |
| `backend/prompt_parts.py` | 0 |
| `backend/prompt_rules.py` | 6 |
| `backend/prompt_sources.py` | 0 |
| `backend/prompt_text.py` | 0 |
| `backend/prompt_weighting.py` | 0 |
| `backend/refine.py` | 4 |
| `backend/refine_batch.py` | 0 |
| `backend/reindex.py` | 2 |
| `backend/schemas.py` | 0 |
| `backend/schemas_gallery.py` | 2 |
| `backend/schemas_generate.py` | 7 |
| `backend/schemas_model.py` | 0 |
| `backend/schemas_prompt.py` | 0 |
| `backend/schemas_rerun.py` | 0 |
| `backend/schemas_snippets.py` | 1 |
| `backend/search.py` | 0 |
| `backend/seeds.py` | 0 |
| `backend/sizes.py` | 0 |
| `backend/snippets.py` | 0 |
| `backend/startup_preferences.py` | 0 |
| `backend/stats.py` | 1 |
| `backend/storage.py` | 6 |
| `backend/submission.py` | 15 |
| `backend/tag_lookup.py` | 1 |
| `backend/tag_swap.py` | 0 |
| `backend/tags.py` | 0 |
| `backend/then_refine.py` | 0 |
| `backend/translation.py` | 0 |
| `tools/check_classic_scripts.py` | 0 |
| `tools/check_run_continue.py` | 3 |
| `tools/ci_changes.py` | 0 |
| `tools/fake_server.py` | 8 |
| `tools/fetch_tokenizers.py` | 2 |
| `tools/run_smoke.py` | 3 |
| `tools/storage_interrupt.py` | 6 |
| `tools/storage_scale.py` | 3 |
| `tools/trial_fixture.py` | 0 |
| `tools/trial_walkthrough.py` | 0 |
