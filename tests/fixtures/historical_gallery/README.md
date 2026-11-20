# Historical gallery samples

These three ZIPs are the verified output of two pinned AuraDream ancestor
revisions. Each ZIP contains its writer's SQLite snapshot, original PNGs,
thumbnails and checksum manifest. All images are synthetic 512 × 512 RGB
swatches. No model, network, or existing gallery is needed to make them.

| Archive | Bytes | Writer commit | Contents |
|---|---:|---|---|
| `legacy-v1.zip` | 97,491 | `16e9f059ff1ef602545e1cec2f1458222b641e1c` | The parent of the instruction-edit storage change. The Local Diffusion writer exports a starred base, its Refine child and a binned third image in a v1 archive. Its SQLite schema predates the edit column. |
| `edit-v2.zip` | 93,933 | `613ca303ff72a3aa6db701c6ff4bc547107493da` | The AuraDream rename commit. The v2 archive holds a starred base and a Kontext edit with the original filename, dimensions and full RGB SHA-256. |
| `edit-only-v2.zip` | 78,031 | `613ca303ff72a3aa6db701c6ff4bc547107493da` | The same historical writer exports only the selected edit, with its original row ID and the source proof still embedded. |

`regenerate.py` makes a temporary local clone, checks out each full commit
and invokes that revision's `storage.save_image`, `grid.run_id` and
`bundles.export_bundle`. The fixed clock and filename UUIDs make identities
stable. Each process uses a new scratch database. Run from a clone that
contains both commits with the project Python and an output directory.

```powershell
uv run python tests/fixtures/historical_gallery/regenerate.py work/regenerated-galleries
```

Two builds on Windows with Python 3.13.15, SQLite 3.53.1 and Pillow 13.0.0
produced these SHA-256 sums. ZIP platform attributes and the system SQLite
or image encoder can change the bytes on other hosts. The tests read the
checked-in archives and compare restored data with independent expectations.

```text
0a9139901eef561cd7f1f073621c9da17ddb539ff7354ec8cd086b3c113ca96a  legacy-v1.zip
cafb1b744ec722b0273119124bf97978391ea700a06661ba02a2e4d59d5b438f  edit-v2.zip
4f51beef82f68bc908893b69b56c11adda4dd0b2b3604948a0a92738ba16aa40  edit-only-v2.zip
```
