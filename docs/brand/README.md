# Tholos logo

The round house's roof line and its one door cut the T of Tholos through a disc.

In Athens the Tholos was the round building where the council's presiding committee stayed day and night, and its
head kept the keys and the state seal (Aristotle, Constitution of the Athenians 44.1). The mark keeps the building to
three pieces: the roof above the T's arm and two walls either side of the door.

## Files

| File | Use |
|---|---|
| `svg/tholos-mark.svg` | colour mark for white and light grounds: navy with a cream T |
| `svg/tholos-mark-on-dark.svg` | cream mark for dark grounds such as `#0D1117` and `#1F2430`; the T shows the ground |
| `svg/tholos-mark-black.svg`, `svg/tholos-mark-white.svg` | one-colour versions; the white one opens both slots by 2 units so the T stays open on dark grounds |
| `svg/tholos-sticker.svg` | mark with its white die-cut border (14 units on the 256 grid, 6.25 % of the mark), transparent outside |
| `svg/tholos-logo-on-light.svg`, `svg/tholos-logo-on-dark.svg` | horizontal logo, height 256 |
| `svg/tholos-logo-stacked.svg` | mark above the wordmark |
| `svg/tholos-appicon.svg` | cream mark at 60 % of a navy rounded-square tile, 1024 grid |
| `png/` | 1024 px mark and app icon, 1600 px logos, 512 px sticker, all rendered from the SVGs |
| `favicon/` | `favicon.svg` (16 px cut, switches to cream in dark mode), `favicon.ico` (16, 32, 48), `favicon-32.png`, `apple-touch-icon.png` (180), `icon-192.png`, `icon-512.png`, `icon-maskable-512.png` |
| `preview-final.png` | everything above on one sheet |

## Colours

| Name | HEX | On `#FFFFFF` | On `#FDF8F1` | On `#0D1117` | Use |
|---|---|---|---|---|---|
| Navy | `#1E2A38` | 14.55:1 | 13.77:1 | 1.30:1 | mark and wordmark on light grounds |
| Cream | `#F6EEE3` | 1.15:1 | 1.09:1 | 16.45:1 | the T on light grounds; mark and wordmark on dark grounds |
| Gold | `#A68A56` | 3.29:1 | 3.11:1 | 5.76:1 | site accent only |

Ratios are WCAG 2 contrast, computed from the HEX values. The cream T inside the navy disc measures 12.65:1. Gold is
kept out of the mark: xdfdet's mark is navy and gold, and Tholos should be told apart by colour as well as shape.

## Clear space

Keep one third of the mark's diameter free on every side. That is the height of the roof above the T's arm, so the
mark carries its own unit. In a lockup, measure it from the mark in that lockup.

## Minimum sizes

- Mark: 16 px with `favicon/favicon.svg`, a cut whose two slots fall on whole pixels at 16 px. Use the regular mark
  from 24 px up.
- Horizontal logo: 24 px tall (about 70 px wide). Below that, use the mark alone.
- Print, as guidance (not proofed on paper): mark 6 mm wide, horizontal logo 8 mm tall.

## Type

The wordmark is Newsreader by Production Type (The Newsreader Project Authors), version 1.003, licensed under the SIL
Open Font License 1.1, taken from the google/fonts repository (`ofl/newsreader`). It is set at weight 540 and optical
size 36, outlined to paths, with one change of ours: a Th ligature in which the T's right arm runs into the h. The
logo files need no font installed. Newsreader is also the display face of the Tholos site.

## Do

- Use `tholos-mark.svg` on white, cream and other light grounds, and `tholos-mark-on-dark.svg` on dark ones.
- Use the black or white file for one-colour print, engraving, embroidery and stamps.
- Use `tholos-sticker.svg` on the coloured sheets of mertkayacs.com; the border is part of the file.
- Give the mark its clear space and keep it level.

## Don't

- Don't recolour `tholos-mark.svg` into one colour. Its T is a filled cream shape and vanishes; use the black or white
  file.
- Don't place the navy mark on navy or near-black grounds.
- Don't fill the T with gold or any other colour, outline the mark, add shadows or rotate it.
- Don't change the slot widths or move the T. The white file and the favicon cut are already tuned for their sizes.
- Don't set "Tholos" in another typeface or retype it; use the outlined wordmark.
