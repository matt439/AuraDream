# TODO audit — 2026-10-26

Read-only reconciliation of the TODO index, its owners and current code,
with one cached, offline tokenizer reproduction. No image model, Marian
model weights, GPU run, download or live gallery was used. This review
owns the confirmed defect below; TODO owns its priority.

## GEN#26: Marian truncates a full sentence chunk

**OPEN · MED · S.** `MarianTranslator._source_limit` reserves room for
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

This closes a deterministic input-loss defect, not a translation-quality
claim. Separately, `GEN#21`'s earlier sentence-splitting fix has only stub
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
