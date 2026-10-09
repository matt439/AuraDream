# Testing notes

Things noticed while using the app, written down as they came up and not yet
looked into. Each entry is the observation as reported; when one is picked
up, mark it and name the commit that closed it.

Legend: `[ ]` open, `[x]` done (commit in the note), `[-]` dropped (why in
the note).

## Bugs

- [ ] **Run thumbnails swallow the wheel.** With the pointer over the run's
  thumbnail panel, the page can't be scrolled up or down.
- [ ] **Lightbox arrow hit-boxes show.** Viewing an image full size, the
  hit-boxes of the back and next arrows are visible.
- [ ] **Lightbox hint covers the image.** Viewing an image full size, the
  hover helper ("Scroll to zoom...") gets in the way of looking at the image.
- [ ] **Bundle file names aren't local time.** Gallery bundle file names use
  a time that isn't the local one.

## Layout

- [ ] **Left pane is short of height.** The left pane feels restricted
  vertically: the batch progress bar is almost cut off, and the elapsed text
  is cut off.
- [ ] **References and LoRA below Grid search.** Reference images and LoRA
  are rarely used, so they could move below Grid search.
- [ ] **Megapixels in the lightbox.** Viewing an image full size, give the
  megapixels after the resolution, e.g. `1500x1000 (1.5MP)`.

## Features

- [ ] **Swipe comparison across CFG.** The swipe comparison between images
  that differ in CFG.
- [ ] **Redo with tags swapped.** Select several images and run them again
  with some tags swapped for others - including replacing two tags with one.
- [ ] **Mass refine.** Refine many images in one action.
