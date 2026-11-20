# Blind review of the November 18 performance pairs

Open `C:\Projects\AuraDream\work\blind-performance-review\index.html` in Edge or Chrome.
The folder contains all 60 full-resolution images. It works offline with the
generation server stopped. The pack starts with zero judgments.

1. Read the prompt. Inspect both images at **Fit**, then use **1:1** to see
   individual image pixels. Drag either image or focus it and use the arrow keys
   to inspect the same region in both. **2×** magnifies fine details.
2. Choose **Prefer A**, **Tie** or **Prefer B**. Use the optional note for what
   you can see, naming the side. Look for prompt coverage, anatomy and
   interaction, fine detail and readable lettering where the prompt calls for it.
3. Use the arrows or pair picker to move around. Return to the same file in the
   same browser profile to resume. **Save backup** downloads the decisions and
   notes. Keep that JSON and the original pack folder together when moving to
   another computer or clearing browser data. Use **Restore backup** on the
   same pack. Each newly built pack has its own randomized assignment.
4. After all 30 choices, **Reveal methods & times** and confirm **Lock & reveal**.
   Read the preference/tie counts by family and each pair's methods, recorded
   wall times and notes. **Download results** keeps the readable report.

The three `USER` rows await Matt's own assessment. A review records image
preferences. Release, source and artifact-rebuild policy, Models placement
and any further qualification have their own decisions in the three owner plans.

## Source and verification

The 2026-11-18 suite used FLUX.1 Krea [dev], fp8 with model offload, one image
per call, 28 steps and guidance 3.5. The three fixed prompts and two seeds run
at 1024 square for all families, with a second 1360 × 768 aspect for the first
two. Each pair uses the original timed occurrence, including the first image
where applicable. These per-image times include any first-use cost incurred.

| Comparison | Pairs | Original suite directories under `.artifacts/gpu-20261118` | Tracked evidence |
| --- | ---: | --- | --- |
| Eager / Regional | 12 | `perf1/eager-suite-a1`, `perf1/regional-suite-retained-a1` | [Regional matrix](perf1-completed-matrix-2026-11-18.json) |
| Native / cuDNN | 12 | `perf3-native-generate-suite-a1`, `perf3-cudnn-generate-suite-a1` | [Attention qualification](perf3-qualification-2026-11-18.json) |
| Off / On | 6 | `perf4-fbc-generate-suite-a1` | [Step-cache qualification](perf4-qualification-2026-11-18.json) |

The builder authenticates the five raw measurements files against the tracked
receipts. It checks the case, effective method, applied request and dimensions,
and checks SHA-256 on the decoded RGB bytes of both original and copied PNGs.
The review copies contain pixels only. Pair order is shuffled across families,
with each method on the left equally often within its family. The page loads
the method key only on reveal. Opening the key file by hand exposes the labels.

Run these commands from the checkout containing the review tool, substituting
the full paths to the evidence and pack when they are in another checkout.

```powershell
# Check an existing pack against its original receipts and this viewer version.
uv run python -m tools.blind_review verify .artifacts/gpu-20261118 work/blind-performance-review

# Prepare a separate blank review at a new path.
uv run python -m tools.blind_review build .artifacts/gpu-20261118 work/new-blind-review
```

Build refuses an existing destination and publishes after every image verifies.
Copy the entire original folder to retain its pair order and A/B key. A backup
belongs to that exact assignment. The method key also retains each source path,
full RGB digest and original elapsed time for later audit.

## Local verification on November 20

The published pack verified against all 60 original RGB digests. The selected
paths and digests also agreed with the retained Regional pair metrics and the
attention/cache quality-pair indexes. In a separately shuffled scratch pack,
Edge displayed each original-size pair, retained a note and preference across
reload, synchronized 1:1 pan across both images, and revealed three tables with
30 rows only after the last judgment and confirmation. A synthetic scratch
review covered backup restoration and conflict refusal. Both browser checks
used disposable profiles. Matt's review was left blank.
