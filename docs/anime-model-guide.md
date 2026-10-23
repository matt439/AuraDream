# Choosing an anime model

Checked against this app's catalogue, profiles and recorded GPU checks on
2026-10-24. Covers the twelve dedicated anime checkpoints in the startup menu;
Anima 2.9B Preview v1 has not been measured on the card.

For a quick decision: **Anima Turbo for speed, WAI for a tuned anime look,
RouWei for hybrid prompts and broad character/style knowledge, Illustrious
for deliberate style control and larger images, and NewBie for scenes with
multiple characters.**

## Which should I use?

| What you want | Start with | Why | How to prompt it |
| --- | --- | --- | --- |
| Quickly explore ideas and try several variations | **[Anima Turbo v1.1](https://huggingface.co/circlestone-labs/Anima)** | The fastest in the local recorded samples. Distillation gives it a stronger, more consistent default style, with less diversity than Anima Base. It also handles illustrations beyond anime. | Tags, prose, or both. For pure prose, describe the image in at least two sentences. |
| Try Anima's aesthetic fine-tune with full guidance | **[Anima Aesthetic v1.1](https://huggingface.co/circlestone-labs/Anima)** | The authors describe a fine-tune for consistency and a stronger default art style. It uses more steps and CFG than Turbo; local samples are not a quality comparison. | Tags, prose, or both. Quality tags are optional; omit `score_*` from both positive and negative prompts. |
| Try the community-expanded Anima preview | **[Anima 2.9B Preview v1](https://huggingface.co/Gazingstars123/Anima-2.9B)** | Expands the transformer from 28 to 40 layers; locally unmeasured, with no established speed or quality advantage. | Detailed tags, prose, or both. Use `@` artist names and pair named characters with their series; score captions were absent in training, but the author permits their use. |
| An everyday anime illustration with a tuned default look | **[WAI Illustrious v17](https://tensor.art/models/990778216270553015/WAI-illustrious-v17)** | A separately tuned anime aesthetic. Its author emphasizes smoother colouring and cleaner subject/background rendering. A useful starting point when you like its default look. | Primarily tags, with prose support. Finish with `masterpiece, best quality, amazing quality`. |
| Combine familiar anime tags with a concise scene description | **[RouWei 0.8 epsilon](https://huggingface.co/Minthy/RouWei-0.8)** | An Illustrious retrain aimed at prompt adherence and broad character/style knowledge, with an April 2025 dataset cutoff. The author's claims have not been compared locally. | Tags, prose, or both; spaces instead of underscores. Prefix artists with `by ` and put them last in this app. |
| Explore NoobAI's V-prediction release with familiar anime tags | **[NoobAI-XL V-pred 1.0](https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0)** | An SDXL model trained with Danbooru and e621 captions. The app applies its required V-prediction Euler recipe. | Tags, prose, or both. Examples includes the author's complete graphite-style character prompt. Keep the full prompt within the short CLIP limit. |
| Direct the artistic style yourself, or work above 1 MP | **[Illustrious XL v2.0-STABLE](https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0)** | An untuned illustration base with less aesthetic preference imposed by the checkpoint. It has stronger documented support for larger native images. A bare subject can produce the dataset's average look. | Tags or concise prose. Specify a style and one clear composition, such as portrait or full body. |
| A known anime character described with familiar tags | **[Animagine XL 4.0 Opt](https://huggingface.co/cagliostrolab/animagine-xl-4.0)** | Character-focused anime generation: identity, series, clothing and poses. The author says Opt improves anatomy, consistency, noise and colours. | Concise Danbooru tags. Put identity and essential details first; finish with its quality/score suffix. The author discourages prose. |
| Mix anime tags with a naturally described scene, or explore styles | **[Kohaku XL Zeta](https://huggingface.co/KBlueLeaf/Kohaku-XL-Zeta)** | Emphasizes character and style fidelity with broad stylistic coverage. Flexible about mixing descriptions and tags. | Tags, prose, or both. Long elaborate prompts are optional; the app still has a short text limit. |
| Describe the character, setting, lighting and composition in sentences | **[Z-Anime Base](https://huggingface.co/SeeSee21/Z-Anime)** | An anime fine-tune of Z-Image oriented toward descriptive control. The full Base version in this app is relatively slow. | Descriptive sentences covering clothing, pose, surroundings and lighting. Use a negative prompt for unwanted features. |
| Try a different anime aesthetic from the Lumina family | **[NetaYume Lumina v4](https://huggingface.co/duongve/NetaYume-Lumina-Image-2.0)** | The author emphasizes sharp outlines, vibrant colours and detailed accessories/clothing. The published guidance still describes v2, so treat those descriptions as family guidance for this v4 selection. | Tags or prose. The app automatically adds the anime system instruction; write only the image description. |
| Keep multiple characters' clothes, colours, actions and positions separate | **[NewBie image Exp0.1](https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1)** | Structured descriptions are its distinctive feature. The authors recommend XML to help keep each character's attributes together. This is an experimental release. | Use one XML block per character and a shared scene block. Ordinary tags and prose also work. The app's Examples menu has the author's XML example. |

These are practical starting choices based on author guidance and the local
checks. The project has not run a controlled image-quality comparison of all
twelve models. Your preferred subjects and aesthetic will still decide which
look you favour.

WAI and Illustrious v2.0-STABLE are separate choices: WAI has its own tuned
style, prompting recipe and size guidance. Do not assume that Illustrious
v2's native resolution range also describes WAI.

## Compare by total megapixels

**Megapixels measure total image area: width x height / 1,000,000. Aspect
ratio describes its shape.** A recommendation of about 1 MP can include
square, portrait and landscape images. For example, 1024 x 1024 is **1.05 MP**,
768 x 1344 is **1.03 MP**, and 1536 x 640 is **0.98 MP**. All three are around
1 MP despite their different shapes and longest sides.

Choose the **MP tier** for image area, then the **aspect ratio** for framing.
The app adjusts width and height to keep approximately that area. Its tier
names are rounded: **1 MP** presets contain 0.98-1.08 MP, **1.5 MP** contain
1.54-1.67 MP, **2 MP** contain 2.06-2.10 MP, and **4 MP** contain 4.19-4.21 MP.
The ranges below use decimal megapixels; author wording that rounds them
differently is identified.

| Model | Total MP guidance | What that means for shape and larger images |
| --- | --- | --- |
| **Anima Turbo** | **About 0.26-2.36 MP documented** | The **1, 1.5 and 2 MP** presets fit the documented area span. Measured locally up to **2.07 MP**, square, portrait and landscape: coherent at each, under 4.4 GB of card, 15.7 s at 2 MP (one prompt and seed). |
| **Anima Aesthetic** | **About 0.26-2.36 MP documented** | The same family guidance covers the **1, 1.5 and 2 MP** tiers. Measured locally up to **2.07 MP**, square, portrait and landscape: coherent at each, 55 s at 2 MP (one prompt and seed). |
| **Anima 2.9B Preview** | **About 0.99-1.77 MP recommended** | The author lists 812x1216 and 1152x1536; adjust the 812 side to this app's 16-pixel grid. The **1 and 1.5 MP** tiers fit that area span. The separately listed 1536-square (**2.36 MP**) is labelled "iffy". No size has been measured locally for this preview. |
| **WAI** | **1.05-1.38 MP reference area**; **no published maximum** | Reference sizes include a portrait. This is an example span, not an established operating range for every aspect ratio. Showcase images also use a separate 1.5x hires refinement pass; the app's **Refine** pass (below) is the nearest it has, with a plain resize in place of the author's upscaling model. |
| **RouWei** | **About 1-1.5 MP recommended** | The author allows any aspect ratio with dimensions divisible by 64. Start at 1024 x 1024; app presets are multiples of 16, so not every preset follows that recommendation exactly. Refine is a separate pass with a plain resize; it does not reproduce the author's latent or GAN upscaling recipe. |
| **NoobAI V-pred** | **About 1 MP recommended**; author-listed shapes are **1.01-1.05 MP** | The card lists seven square, portrait and landscape sizes, with 832 x 1216 emphasized. The app highlights those framings. Larger presets are available, but the author does not establish a larger native operating range. |
| **Illustrious** | **About 1-2.25 MP native**, in the authors' wording; roughly **4 MP** possible with quality loss | The authors describe varied aspect ratios. Their higher training size is **2.36 MP** when converted exactly; treat the native range as approximate. A long side alone is not a ceiling. Start with the **1, 1.5 or 2 MP** presets. |
| **Animagine** | **About 1 MP recommended**; author-listed shapes are **0.98-1.05 MP** | The author lists square, portrait and landscape shapes. Use those aspect presets where possible. Larger area can degrade; the warning includes a **2.36 MP** square. |
| **Kohaku** | **About 1 MP recommended**; reference area **1.05 MP** | The recommendation explicitly allows a similar total pixel count, so the reference square is not a square-only requirement. Locally, **2.36 and 4.19 MP** squares fit easily (under 6.8 GB of card, 19 and 36 s); their composition was not judged against 1 MP. |
| **Z-Anime** | **About 0.26-4.2 MP documented** | The author explicitly allows **any aspect ratio**. **1.5-2 MP** is the practical local iteration range; the coherent **4.19 MP** sample took **8 minutes 39 seconds** and nearly filled the card. |
| **NetaYume** | **1.05-1.57 MP example area**; **no published v4 range or maximum** | Inherited/v2 examples include square and portrait images. They do not define v4's aspect-ratio coverage or a size ceiling. Start around **1 MP**. Locally a **4.19 MP** square stayed coherent but took **4 minutes**; **2.36 MP** took under 2. |
| **NewBie** | **1.05 MP example area**; **no published minimum or maximum** | The quickstart and local check use a square. That example establishes neither a square-only requirement nor a guarantee for other shapes. Start around **1 MP**. Locally the authors' example held at **2.36 MP** but at **4.19 MP** its two characters repeated into a sheet of about eight. |

For a quick choice by area: **around 1 MP**, start with the model whose style
and prompting you prefer. At **1.5-2 MP**, Illustrious and Z-Anime have local
larger-image evidence; Anima's documented area also covers these tiers, and
Turbo and Aesthetic held together there locally. At **4 MP**, allow for experiments:
Z-Anime's sample retained its scene but was slow, NetaYume's held
and NewBie's broke into repeated figures; Illustrious's **4.19 MP**
sample omitted the requested doctor, which its **2.36 MP** sample retained.
Those used different prompts between models and are not a controlled
comparison. See [local verification](verified.md).

**Equal MP does not guarantee equal composition quality at every aspect
ratio.** Animagine's author-listed shapes are a specific set; other app
presets can have the same area without being on that list. Very wide or tall
framing still changes what the model must compose. Check the subject,
anatomy and layout as well as whether generation completes.

**To get the same picture larger, use Refine rather than a larger render.**
Press ⤢ on a finished image and Refine in the dialog it opens: by default it
is resized by 1.5x and re-denoised part of the way, which keeps the
composition where the same seed at the larger size would draw a new one. **WAI, RouWei, NoobAI V-pred, Illustrious, Animagine, Kohaku and Z-Anime**
have the pass; **Anima Turbo, Anima Aesthetic, Anima 2.9B Preview, NetaYume and NewBie** do not have a compatible
image-to-image sibling in this app. All three Anima selections use the app's custom
pipeline; Diffusers does ship an Anima modular image-to-image workflow, but
it is not integrated here. The dialog's Strength starts
at each checkpoint's own value - **0.4** on the six SDXL models, **0.15**
on Z-Anime - and lower keeps more of the original. In the local checks WAI's
1024 x 1344 became 1536 x 2016 in about 12 seconds and Z-Anime's 1024 x 1024
became 1536 x 1536 in about 33 seconds, both keeping the character and
scene. A pass may go past 4 MP: WAI reached 2560 x 3360 (8.6 MP) in 46
seconds, but Z-Anime fills the 16 GB card at 2048 x 2048 and slows to a crawl
beyond it. RouWei and NoobAI also passed a 1024 x 1024 to 1536 x 1536 check,
each taking about 9 seconds. These are separate samples, not a comparison of
image quality. See the README's [Refine](../README.md#refine) section.

## Starting settings and likely waiting time

The app fills in each checkpoint's starting Steps and CFG automatically.
Leave Mu blank where available to keep its supplied schedule; the SDXL
models and this Anima integration disable Mu.

| Model | Steps | CFG | Recorded time per image | Sample area and shape |
| --- | ---: | ---: | --- | --- |
| Anima Turbo v1.1 | 8 | 1 | About 4-6 seconds | 1.05 MP, square |
| Anima Aesthetic v1.1 | 30 | 4 | About 23 seconds | 1.05 MP, square |
| Anima 2.9B Preview v1 | 30 | 4 | Unmeasured | No local sample |
| WAI Illustrious v17 | 28 | 6 | About 9-10 seconds | 1.38 MP, portrait |
| RouWei 0.8 epsilon | 28 | 6 | About 8 seconds | 1.05 MP, square |
| NoobAI-XL V-pred 1.0 | 28 | 5 | About 8 seconds | 1.05 MP, square |
| Illustrious XL v2.0-STABLE | 28 | 5 | About 11 seconds, first image | 1.05 MP, square |
| Animagine XL 4.0 Opt | 28 | 5 | About 17 seconds, first image | 1.05 MP, square |
| Kohaku XL Zeta | 28 | 5 | About 9-10 seconds | 1.05 MP, square |
| Z-Anime Base | 40 | 4 | About 78-79 seconds | 1.05 MP, square |
| NetaYume Lumina v4 | 40 | 5 | About 39 seconds | 1.05 MP, square |
| NewBie image Exp0.1 | 28 | 4 | About 39 seconds | 1.05 MP, square, XML prompt |

The measured rows are individual RTX 5080 checks recorded in September and
October 2026, using Windows FP8 and model offload. The Animagine and Illustrious
rows above are October 5 first-image samples with other processes keeping the
CPU about 75% busy; they are upper bounds, not warm-loop measurements. RouWei,
NoobAI and Anima Aesthetic were checked on October 7.
They indicate likely waiting time;
different prompts, sizes and run conditions prevent a controlled speed
ranking. Loading is excluded; sample methods and whether PNG saving is included
vary across the retained measurements.
See [the measurement record](verified.md#checkpoints) for the exact conditions.

## Prompt habits that matter when switching models

- **Keep SDXL prompts short.** Animagine, Kohaku, Illustrious, WAI, RouWei and NoobAI each
  stop at 77 tokens per CLIP text encoder in this app, leaving about 75 for
  content. Tokens are pieces of words, not a word count. Put essential
  details early and use **Check** to see where reading stops. Kohaku's
  advertised 300-token training context does not extend this app's limit.
- **Use the selected model's recipe.** Animagine's recommended suffix is
  `masterpiece, high score, great score, absurdres`; WAI's is
  `masterpiece, best quality, amazing quality`. Illustrious and Kohaku have
  their own guidance. The **Tags**, **Examples** and prompt help reflect the
  selected checkpoint.
- **Put RouWei artists last, with `by ` before each name.** Its author prefers
  separate CLIP chunks, with putting artists last as a fallback. This app has
  no prompt chunking: `BREAK` is literal text, and the entire prompt must fit
  the CLIP limit. Keep the negative short; the author's recipe is
  `worst quality, low quality, watermark`.
- **Give Illustrious a style.** A style description or artist tag helps
  direct its untuned base. Choose a clear composition rather than stacking
  conflicting framing tags.
- **Keep NoobAI's V-pred recipe distinct.** The app uses Euler, V-prediction,
  zero-terminal-SNR rescaling and starts at 28 steps/CFG 5. Its quality and date
  vocabulary differs from Illustrious and Animagine. The author's recommended
  negative also suppresses furry subjects; adapt it to your subject.
- **Give Z-Anime descriptive sentences.** Explain what is happening and how
  the scene is lit. Its author favours prose over a flat tag list.
- **For lettering inside an image, avoid Animagine.** Its guide explicitly
  says text rendering is unsupported. The local checks do not establish a
  best anime model for reliable lettering.
- **At Anima Turbo's default CFG 1, the negative prompt does nothing.** The
  integration supports negatives above CFG 1, but Turbo's recommended
  starting point is CFG 1. Write tags in lowercase with spaces, and put `@`
  before artist names.
- **Anima Aesthetic uses its own defaults and advice.** It starts at CFG 4,
  where negative prompts apply. Leave `score_*` out of both prompts; quality
  tags are unnecessary, though `masterpiece, best quality` may stay. Select
  **Anima Aesthetic v1.1** in the menu or use
  `--model circlestone-labs/Anima@aesthetic-v1.1`; the bare Anima id remains Turbo.
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
- [Size presets](../backend/sizes.py) for the actual MP at each aspect ratio.
- [Design notes](design-notes.md) for source links and integration differences.
- [Verified on the card](verified.md) for dated performance and quality observations.

When a checkpoint version or integration changes, refresh the relevant row
against those files. Keep author recommendations separate from locally
measured results.
