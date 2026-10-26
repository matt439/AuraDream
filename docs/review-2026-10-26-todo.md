# TODO audit — 2026-10-26

Read-only reconciliation of the TODO index, its owners and current code,
with one cached, offline tokenizer reproduction. No image model, Marian
model weights, GPU run, download or live gallery was used. This review
owns the confirmed defect below; TODO owns its priority.

## GEN#26: Marian truncates a full sentence chunk

**Fixed and checked, 2026-10-26.** `MarianTranslator._source_limit` reserves room for
special tokens: the model limit of 512 becomes a source budget of 510.
`_chunks` splits an over-long sentence into pieces within that budget.
`translate` then passes the same 510 as the tokenizer's *total* input
limit, with special tokens enabled. A full piece therefore loses source
content when its EOS is added.

Confirmed with the cached `Helsinki-NLP/opus-mt-zh-en` tokenizer, with both
offline flags set and `local_files_only=True`. This synthetic sentence
has 1,201 source tokens. `_chunks` returns pieces of 510, 510 and 181;
the first two become 511 tokens with EOS, then are truncated to 510.
Each loses its final source token, ID 5667, while EOS remains. No model
inference is needed to establish the loss.

From the repository root, the reproduction's Python body is:

```python
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from backend.translation import MarianTranslator
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Helsinki-NLP/opus-mt-zh-en", local_files_only=True
)
source = (
    "\u8fd9\u662f\u4e00\u53e5\u5173\u4e8e\u98ce\u666f\u7684\u8bf4\u660e" * 200
    + "\u3002"
)
chunks = MarianTranslator._chunks(source, tokenizer)
limit = MarianTranslator._source_limit(tokenizer)
print(len(tokenizer.encode(source, add_special_tokens=False)))
for chunk in chunks:
    raw = tokenizer.encode(chunk, add_special_tokens=False)
    uncut = tokenizer(chunk, truncation=False)["input_ids"]
    actual = tokenizer(chunk, truncation=True, max_length=limit)["input_ids"]
    print(len(raw), len(uncut), len(actual), uncut[-6:], actual[-6:])
```

The output lengths were `1201`, then `510 511 510`, `510 511 510` and
`181 182 182`. A tokenizer warning about the original over-long sentence
is expected: it is deliberately split before any model call. If the
tokenizer is absent, the offline check must stop rather than download it.

Done means:

- Use distinct source-token and total model-input budgets so adding
  special tokens cannot truncate a chunk's source content. Keep every
  input within the actual model limit.
- Add a regression with a tokenizer that faithfully adds special tokens
  and applies truncation, covering the source boundary, a sentence just
  over it and multiple full chunks. The current stub ignores truncation
  and therefore misses this defect. Ordinary tests must need no network
  or weights.
- Run the focused translation tests. Repeat the cached offline tokenizer
  check when available and record the preserved source tokens; this is
  additional dependency evidence, not a requirement to download files.

The fix separates `_input_limit` (the tokenizer's bounded total input,
512 here) from `_source_limit` (510 after reserving special-token room).
`translate` now passes the total limit to the tokenizer. All 18 focused
translation tests pass on Windows/Python 3.13.15, including a truncating
tokenizer that inserts two special tokens and checks actual model input
at, just past and across multiple full source chunks. An independent
review found no blocking issue.

The cached tokenizer reproduction above was repeated with both offline
flags and `local_files_only=True`, without weights or model inference.
The 1,201 source tokens still split into 510, 510 and 181. The actual total
lengths are now 511, 511 and 182, equal to their untruncated inputs; both
full chunks preserve final source token 5667. No download or GPU run was
used. This closes a deterministic input-loss defect, not a
translation-quality claim.

Separately, `GEN#21`'s earlier sentence-splitting fix has only stub
tests after its implementation (`verified.md`, Behaviour, the Marian
row). A later bounded CPU qualification can replay a synthetic
multi-sentence expansion through the real translator using already cached
weights, with networking disabled, and compare coverage with individual
sentence translations. It should record the model/cache identity, input,
outputs and runtime; stop if the cached files are absent. That validation
does not require ERNIE, image generation or the GPU, and does not make a
general claim about translation accuracy.

## Index and owner reconciliation

- NOW said no implementation work remained while seven rows were OPEN.
  Existing tooling proposals still have real gaps; the
  [tooling plan](plan-tooling.md) defines bounded stages, starting with
  `DEV#7`'s portable output. Keep card, human and product-decision work distinct.
- The critic's decision about withholding rewrites that lose user
  content was missing from TODO despite remaining explicit in the design
  notes and `verified.md`. Restore it as `PA#11`; warning-only behavior
  remains implemented until Matt decides otherwise.
- Historical `PA#10` labels name both the 2026-10-25 critic tag-loss
  warning and the 2026-10-26 hybrid-prompt fix. Disambiguate that history
  by the completed work's name; do not reuse the ID for another open task.
- The loader's old measurement paragraph still called its implementation
  decision open. The fix exists; `GEN#22` is the remaining real Windows
  RAM measurement.
- Inpainting's phase text still called the Krea dev outpaint and its
  Nunchaku inpaint unmeasured. Their results are dated 2026-10-21 and
  2026-10-25 in `verified.md`. Dev and schnell still need their indexed
  card checks.
- The plans' original open-measurement lists also retain completed mask
  byte counts and CPU fill/mask timings. Those results already exist in
  `verified.md`, Inpaint and Outpaint; they are not new ready work. Keep
  the design lists as historical scope with a current-status pointer.
- The manual-validation owner also leaves native notification delivery
  and actual Windows display scaling unchecked. `UI#7` should name these
  alongside screen-reader and touch checks.

## Remaining-work audit after the overnight stages

The four OPEN rows still include implementation. They are not a declaration
that every remaining step needs Matt. The original speed review specified a
plan first, then implementation, then CARD; the new plans' "before shipping"
and "before exposing" questions do not prohibit source work or scratch tests.
The continuing implementation request permits those stages while leaving
real model loads/downloads, GPU runs and experimental render activation out
of this session.

| Row | Independent implementation identified by the audit | Separate activation or evidence boundary |
|---|---|---|
| PERF#1 | Loader/selection/status lifecycle and fake-backed placement, failure and teardown checks | Compiler/toolchain installation; opening compiled rendering; GEN#22 and the finite card protocol |
| PERF#3 | Cold request checks, dormant selection/status/processor lifecycle and fake checks | An explicit external-override policy before exposing selection; native/cuDNN card fit and repeatability; external wheels remain later work |
| PERF#4 | Cold request checks, dormant per-call integration and presentation checks | Opening cached rendering and the shared-seed card comparison; Matt's quality acceptance before routine use |
| GAL#6 | Default-off configuration, startup/preload ordering and status using scratch galleries | Explicit unattended protection/activation values; real OS credential integration; verified shared exclusion before retention deletion |
| DEV#4 | Diagnostic/artifact preparation is complete | Linux qualification is blocked by the actual WSL service denial; hosted qualification requires publishing, outside the local-only request |

The first missing cold cuDNN/FBC checks were implemented following this audit,
with all nonempty-render refusals retained. Backup lifecycle work can likewise
require age, count, destination and informed protection as explicit activation
inputs rather than invent values for Matt. Existing draft policies must not
silently become active when lifecycle wiring lands. Conservative empty-gallery
skips, manual-age exclusion and no retry are existing first-version semantics,
not evidence of a personal preference. The OS-secret option still needs a real
provider and recovery design; it cannot fall back to plaintext.

The explicit default-off GAL#6 startup/configuration/status slice was then
implemented and checked on scratch galleries, as recorded in the backup plan.
Explicit Windows shared-mode installation and mode-aware downloads were then
implemented and checked on an owned NTFS process cohort. Published archive
imports, listings, public encrypted helpers and receipt hashing then joined the
same protocol, including original-key quarantine for uncertain input closes.
The unconnected Windows mutation mechanism then gained mandatory installed-mode
exclusion, pinned scope and same-handle hash/disposition with scratch qualification.
V3 then added separately consented retention, current-N protection and durable
pending/retired proofs with bounded retry history. Cleanup warnings preserve
verified backup success/time, and pending/unknown state remains visible while
disabled. V4 then added Windows credential source/setup/recovery, pending-first
provisioning and worker-only lookup with API stubs, without real vault access.
Actual intended-user store/session and independent recovery qualification remain.
No real gallery/destination was activated; older v2 counts remain inert.

The next bounded PERF#1 slice also landed in source: actual eager-load publication
and shared teardown own unpublished pipeline/component references and future
registration state, with strict synthetic failure/weak-reference checks. It
changes no public loader selection or nonempty-render admission. Regional setup,
selection and real execution remain in the compilation owner. The next private
loader/call stage then wired cold preflight and owned regional registration into
the actual load transaction, with one policy boundary for direct/refine forwards.
Synthetic checks preserve eager behavior and every public nonempty refusal;
compiler/cache setup and real qualification remain separate.

Private attention preparation/call ownership then joined the same transaction,
with exact pre/post-forward verification and weak terminal-discard exclusions.
Focused synthetic checks preserve primary exceptions, eager behavior and every
public nonempty refusal. Effective imported environment behavior, public
selection and actual cuDNN/card qualification remain in the attention owner.

Private FBC ownership then joined load/publication/call/teardown, with effective
Native checks and explicit deferred disposal after unsafe cleanup. Synthetic
checks cover scope/lock order, stale callbacks, primary failures and weak terminal
invalidation. Public cached rendering, main-status reconciliation and actual
quality/time/memory qualification remain separate.

Approved workflow work was reconciled against its owners, production paths and
regression fixtures: mixed tag/prose handling and native tag spelling, More seeds
queueing and response retries, queue controls and missed outcomes, deterministic
prompt choices, Bundles password/count clarity, target sizing, button hierarchy,
shortcuts and compact setup are complete. No new implementation defect was
established in that subset. A1111 weighting/chunking is the separate GEN#20
decision, not an unfinished native-parser fix. The historical local-browser
approval language in completed result paragraphs creates no new test gate.

Genuine remaining human/product boundaries are DEV#1's task trials, UI#7's
hands-on accessibility/desktop evidence, GEN#20's weighting behavior, INP#11's
instruction-editing scope, INP#12's detailer/detector scope and PA#11's choice
about withholding content-losing rewrites. PERF#2 retains its PERF#1 dependency;
CARD and explicitly deferred rows keep their existing status. No agent trial,
CPU test or browser emulation supplies those missing human or card results.

## Current boundary origins

The renewed audit used the actual TODO and owners after the credential source
stage. Requirements below are distinguished from ordinary engineering choices;
historical plan-only wording and old local-browser approval receipts do not
create another approval requirement for current source work.

| Remaining item | Requirement and origin | Independent work still allowed |
|---|---|---|
| PERF#1 | The compilation owner's selection/cache/lifecycle stages remain source work. Its exposure section requires an optional compiler/toolchain pin, supported restricted experiment and measured cold/warm trade-off before offering it. The current user scope excludes package installation, compiler/model execution and opening non-Eager rendering. | Private startup cache ownership and cold switch validation; further selection/status/preferences integration and fake tests can proceed with rendering closed. Cache isolation and refusal of unowned environment values are conservative implementation choices, not new user-policy gates. |
| PERF#3 | The attention owner's exposure section leaves nondefault external-environment representation/refusal explicit; actual native/cuDNN fit and repeatability need the card. The current user scope keeps non-Eager rendering closed and excludes external wheel installation. | Shared closed selection contracts, source/status integration and synthetic lifecycle checks. The implemented private Native invariant does not choose public environment precedence. |
| PERF#4 | The cache owner's finite acceptance requires the fixed image pairs, time/memory checks and Matt's quality judgment before routine use. Current user scope excludes model/GPU work and cached-render activation. | Closed selection/admission/status integration and fake checks. New visible opt-in controls have no useful acceptance result while exposure remains closed; that is an implementation sequencing choice, not a denied action. |
| DEV#4 | Linux qualification is blocked by actual `Wsl/EnumerateDistros/Service/E_ACCESSDENIED`; the user explicitly forbids retry. Hosted execution requires publishing, excluded by the local-only request. | Native Node API preflight is now built and checked. No additional established tooling source defect remains; Linux/font/resource and hosted evidence remain unqualified. |
| GAL#6 | The backup owner requires actual intended-user OS credential delivery and independent-password recovery. This session explicitly excludes real vault operations; API stubs cannot supply that evidence. Activating any gallery also requires its protection/age/count/destination inputs and separate deletion consent. | The default-off source, encrypted fake-provider restore and recovery tool are built. No source defect remains established by this audit; no real credential or gallery was activated. |
| DEV#1; UI#7 | The validation owner calls for Matt's observed common-task trials and hands-on NVDA/Narrator, touch, native notification and actual Windows-scaling evidence. These are human/desktop evidence requirements, not missing browser fixtures. | Preparation and the approved browser/layout fixes are built. Keep the trial records blank until observed; do not infer a redesign winner from emulation. |
| GEN#20; INP#11; INP#12; PA#11 | Current TODO marks these DECISION: whether to plan SDXL A1111 weighting/chunking, instruction edits, a detailer with a suitable detector, and whether to withhold content-losing critic rewrites. Their owners specify those product choices. | Preserve current native prompt parsing and warning behavior. No newly authorized feature implementation is implied by the overnight loop. |
| PERF#2 | Current TODO explicitly waits on PERF#1. | Do not broaden into activation-quantization experiments before that dependency. |
| CARD rows; GEN#15/#16/#22 | Repository rules require separately approved real-card runs and the existing RAM/VRAM limits; current user scope excludes model loading/download and GPU workloads. GEN#22 is the cold-load prerequisite for dependent FLUX measurements. | Preserve the implemented CPU contracts and finite measurement protocols; no fabricated device acceptance. |
| INP#5/#6/#9/#10; GEN#6/#8; PA#9 | Current TODO keeps these DEFERRED. GEN#8 specifically waits until Matt raises it. | No new deferred feature is admitted by this audit. |

The small exposure decision set is therefore the restricted performance
experiment/toolchain and environment policy, the four indexed product choices,
and explicit real-data activation inputs if wanted. None of these is a blanket
ban on the remaining performance source stages. Approved mixed-prompt/native-tag
parsing, seed queueing/retries, bundle clarity, shortcuts and compact setup have
no current open row or reproduced regression in this audit.

The last established closed-gate PERF#4 app gap was then repaired: status now
reports a discarded private pipeline as not-ready, and explicit Load consults
actual readiness rather than requiring a preceding GET. Fake checks preserve
loading/error details, selection, preferences and ordinary load coalescing. This
is latent lifecycle groundwork; today's valid public requests still cannot enter
private FBC. After that repair, the renewed bounded audit identified no additional
present correctness change under the closed-gate restrictions. Public selection/
preferences/UI remain unimplemented, rather than falsely qualified or declared
done. Ordinary source preparation remains permitted; the next functional exposure
needs the explicit recipe/toolchain/bootstrap and external-environment choices
above, while actual card/OS/human evidence retains its separate boundaries.
