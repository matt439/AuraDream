# Testing notes

Things noticed while using the app, written down as they came up and not yet
looked into. Each entry is the observation as reported; when one is picked
up, mark it and name the commit that closed it.

Legend: `[ ]` open, `[x]` done (commit in the note), `[-]` dropped (why in
the note).

## Bugs

- [x] **Run thumbnails swallow the wheel.** With the pointer over the run's
  thumbnail panel, the page can't be scrolled up or down. (d506c9d; the
  docked "Earlier this session" panel too, 3ec1e98)
- [x] **Lightbox arrow hit-boxes show.** Viewing an image full size, the
  hit-boxes of the back and next arrows are visible. (6bc7af0)
- [x] **Lightbox hint covers the image.** Viewing an image full size, the
  hover helper ("Scroll to zoom...") gets in the way of looking at the image.
  (6bc7af0; the help is now on the zoom readout)
- [x] **Bundle file names aren't local time.** Gallery bundle file names use
  a time that isn't the local one. (4c161ad)

## Layout

- [x] **Left pane is short of height.** The left pane feels restricted
  vertically: the batch progress bar is almost cut off, and the elapsed text
  is cut off. (f339e61, fae66bf; fits at 1400x900, 1920x947 and
  2048x1040@1.25; at 1536x864 since 2026-10-20, UI#8, by the panel's own
  gap and field height under 900 px tall)
- [x] **References and LoRA below Grid search.** Reference images and LoRA
  are rarely used, so they could move below Grid search. (f339e61)
- [x] **Megapixels in the lightbox.** Viewing an image full size, give the
  megapixels after the resolution, e.g. `1500x1000 (1.5MP)`. (6bc7af0;
  written `1536×1536 (2.4 MP)` with a space, as the size hints are)

## Features

- [x] **Swipe comparison across CFG.** The swipe comparison between images
  that differ in CFG. (ccd999d, f1079c3, 2db4787)
- [x] **Swipe comparison across steps.** Compare steps, Fewer / More steps
  and the sheet's Swipe steps, on the CFG comparison's ladder code.
- [x] **Redo with tags swapped.** Select several images and run them again
  with some tags swapped for others - including replacing two tags with one.
  (Select on the gallery: 6d259b6, 56216c6; Swap tags: a4b6a19, 9611a8d,
  d7beaa9, ba2fa31, e3172c5)
- [x] **Mass refine.** Refine many images in one action. (481af25, beeeb65,
  abbbcf1, e3172c5)
