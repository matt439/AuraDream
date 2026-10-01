# Choosing an anime model

Checked against this app's catalogue, profiles and recorded GPU checks on
2026-10-01. Covers the eight dedicated anime checkpoints in the startup menu.

For a quick decision: **Anima Turbo for speed, WAI for a tuned anime look,
Illustrious for deliberate style control and larger images, and NewBie for
scenes with multiple characters.**

## Which should I use?

| What you want | Start with | Why | How to prompt it |
| --- | --- | --- | --- |
| Quickly explore ideas and try several variations | **[Anima Turbo v1.1](https://huggingface.co/circlestone-labs/Anima)** | The fastest in the local recorded samples. Distillation gives it a stronger, more consistent default style, with less diversity than Anima Base. It also handles illustrations beyond anime. | Tags, prose, or both. For pure prose, describe the image in at least two sentences. |
| An everyday anime illustration with a tuned default look | **[WAI Illustrious v17](https://tensor.art/models/990778216270553015/WAI-illustrious-v17)** | A separately tuned anime aesthetic. Its author emphasizes smoother colouring and cleaner subject/background rendering. A useful starting point when you like its default look. | Primarily tags, with prose support. Finish with `masterpiece, best quality, amazing quality`. |
| Direct the artistic style yourself, or work above 1 MP | **[Illustrious XL v2.0-STABLE](https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0)** | An untuned illustration base with less aesthetic preference imposed by the checkpoint. It has stronger documented support for larger native images. A bare subject can produce the dataset's average look. | Tags or concise prose. Specify a style and one clear composition, such as portrait or full body. |
| A known anime character described with familiar tags | **[Animagine XL 4.0 Opt](https://huggingface.co/cagliostrolab/animagine-xl-4.0)** | Character-focused anime generation: identity, series, clothing and poses. The author says Opt improves anatomy, consistency, noise and colours. | Concise Danbooru tags. Put identity and essential details first; finish with its quality/score suffix. The author discourages prose. |
| Mix anime tags with a naturally described scene, or explore styles | **[Kohaku XL Zeta](https://huggingface.co/KBlueLeaf/Kohaku-XL-Zeta)** | Emphasizes character and style fidelity with broad stylistic coverage. Flexible about mixing descriptions and tags. | Tags, prose, or both. Long elaborate prompts are optional; the app still has a short text limit. |
| Describe the character, setting, lighting and composition in sentences | **[Z-Anime Base](https://huggingface.co/SeeSee21/Z-Anime)** | An anime fine-tune of Z-Image oriented toward descriptive control. The full Base version in this app is relatively slow. | Descriptive sentences covering clothing, pose, surroundings and lighting. Use a negative prompt for unwanted features. |
| Try a different anime aesthetic from the Lumina family | **[NetaYume Lumina v4](https://huggingface.co/duongve/NetaYume-Lumina-Image-2.0)** | The author emphasizes sharp outlines, vibrant colours and detailed accessories/clothing. The published guidance still describes v2, so treat those descriptions as family guidance for this v4 selection. | Tags or prose. The app automatically adds the anime system instruction; write only the image description. |
| Keep multiple characters' clothes, colours, actions and positions separate | **[NewBie image Exp0.1](https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1)** | Structured descriptions are its distinctive feature. The authors recommend XML to help keep each character's attributes together. This is an experimental release. | Use one XML block per character and a shared scene block. Ordinary tags and prose also work. The app's Examples menu has the author's XML example. |

These are practical starting choices based on author guidance and the local
checks. The project has not run a controlled eight-model image-quality
comparison. Your preferred subjects and aesthetic will still decide which
look you favour.

WAI and Illustrious v2.0-STABLE are separate choices: WAI has its own tuned
style, prompting recipe and size guidance. Do not assume that Illustrious
v2's native resolution range also describes WAI.

## Starting settings and likely waiting time

The app fills in each checkpoint's starting Steps and CFG automatically.
Leave Mu blank where available to keep its supplied schedule; the SDXL
models and this Anima integration disable Mu.

| Model | Steps | CFG | Recorded time per image | Sample size |
| --- | ---: | ---: | --- | --- |
| Anima Turbo v1.1 | 8 | 1 | About 4-6 seconds | 1024 x 1024 |
| WAI Illustrious v17 | 28 | 6 | About 9-10 seconds | 1024 x 1344 |
| Illustrious XL v2.0-STABLE | 28 | 5 | About 15 seconds | 1024 x 1024 |
| Animagine XL 4.0 Opt | 28 | 5 | No comparable timing recorded | Generation and batch use verified |
| Kohaku XL Zeta | 28 | 5 | About 9-10 seconds | 1024 x 1024 |
| Z-Anime Base | 40 | 4 | About 78-79 seconds | 1024 x 1024 |
| NetaYume Lumina v4 | 40 | 5 | About 39 seconds | 1024 x 1024 |
| NewBie image Exp0.1 | 28 | 4 | About 39 seconds | 1024 x 1024, XML prompt |

These are individual RTX 5080 checks recorded on September 17-26, 2026,
using Windows FP8 and model offload. They indicate likely waiting time;
different prompts, sizes and run conditions prevent a controlled speed
ranking. Loading is excluded. Most samples also exclude PNG saving, while
Illustrious's CLI timings include it. Animagine's earlier successful runs
did not record comparable timing or exact quantization/offload settings.
See [the measurement record](verified.md#checkpoints) for the exact conditions.

## Choosing a resolution

Start around **1 MP** while finding the prompt and style you want. For larger
direct renders, **Illustrious is a sensible first choice around 1.5-2.25 MP**;
**Z-Anime is another option if you can accept the longer wait**.

| Model | Resolution guidance |
| --- | --- |
| Animagine | The author recommends shapes around **1 MP** and warns that sizes such as 1536 x 1536 can degrade. |
| Kohaku | The author recommends **1024 x 1024 or a similar pixel count**. Larger sizes have not been measured locally. |
| Illustrious | The authors document roughly **1-2.25 MP native output**, with possible degradation at 4 MP. The local 1536 x 1536 prose sample retained the requested doctor; at 2048 x 2048, the scene omitted the doctor. That is one observation, not a general quality verdict. |
| WAI | The author's examples start at **1024 x 1344**, then use a separate high-resolution refinement pass. This app generates directly at the requested size and does not perform that pass. No maximum is published. |
| Z-Anime | The author documents approximately **512 x 512 through 2048 x 2048**. Local samples stayed coherent through 4 MP, but the 4 MP image took **8 minutes 39 seconds** and nearly filled the card. Prefer **1.5-2 MP** for iteration. |
| Anima | The authors document **512 x 512 through 1536 x 1536**, about 0.26-2.36 MP. Only 1 MP has been measured locally. |
| NetaYume | Family examples cover **1024 x 1024 and 1024 x 1536**. Those examples do not establish a v4 maximum. Start at 1 MP; larger sizes have not been measured locally. |
| NewBie | The quickstart uses **1024 x 1024**. That is an example, not a published ceiling. Higher resolutions have not been measured locally. |

A successful large render establishes that the image can be produced; check
that the subject, anatomy and composition still follow the prompt. The
Illustrious and Z-Anime observations above used different prompts and are
not a controlled comparison. Details are in [local verification](verified.md).

## Prompt habits that matter when switching models

- **Keep SDXL prompts short.** Animagine, Kohaku, Illustrious and WAI each
  stop at 77 tokens per CLIP text encoder in this app, leaving about 75 for
  content. Tokens are pieces of words, not a word count. Put essential
  details early and use **Check** to see where reading stops. Kohaku's
  advertised 300-token training context does not extend this app's limit.
- **Use the selected model's recipe.** Animagine's recommended suffix is
  `masterpiece, high score, great score, absurdres`; WAI's is
  `masterpiece, best quality, amazing quality`. Illustrious and Kohaku have
  their own guidance. The **Tags**, **Examples** and prompt help reflect the
  selected checkpoint.
- **Give Illustrious a style.** A style description or artist tag helps
  direct its untuned base. Choose a clear composition rather than stacking
  conflicting framing tags.
- **Give Z-Anime descriptive sentences.** Explain what is happening and how
  the scene is lit. Its author favours prose over a flat tag list.
- **For lettering inside an image, avoid Animagine.** Its guide explicitly
  says text rendering is unsupported. The local checks do not establish a
  best anime model for reliable lettering.
- **At Anima's default CFG 1, the negative prompt does nothing.** The
  integration supports negatives above CFG 1, but Turbo's recommended
  starting point is CFG 1. Write tags in lowercase with spaces, and put `@`
  before artist names.
- **Keep NetaYume's system prefix out of the editor.** The app adds it
  automatically. Its 256-token total sequence includes that instruction.
- **For NewBie, group attributes by character.** XML provides separate
  appearance, clothing, expression, action and position fields. Keep the
  main description within its shorter Gemma encoder's 512-token limit,
  even though its Jina encoder accepts 2048.

Anima, Z-Anime, NetaYume and NewBie have longer text contexts than the SDXL
models. That gives you more room for descriptions, but **Check** is still
useful for detecting truncation and model-specific prompt issues.

## Sources and maintenance

The model names in the chooser link to the authors' cards. Additional
Illustrious background comes from the
[author's base-model explanation](https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0)
and [technical report](https://arxiv.org/html/2409.19946v1).

For the exact behaviour of this app, use:

- [Model catalogue](../backend/launch.py) for the versions actually selected.
- [Profiles](../backend/model_defaults.py) for starting settings and controls.
- [Prompt and size guides](../backend/guides.py) for checkpoint-specific advice.
- [Design notes](design-notes.md) for source links and integration differences.
- [Verified on the card](verified.md) for dated performance and quality observations.

When a checkpoint version or integration changes, refresh the relevant row
against those files. Keep author recommendations separate from locally
measured results.
