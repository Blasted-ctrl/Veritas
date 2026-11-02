---
name: Veritas
description: A light, precise product interface with a bold green V and readable media evidence.
colors:
  bg: "#f8faf8"
  surface: "#ffffff"
  surface-2: "#f1f5f1"
  line: "#d9e3db"
  ink: "#15382a"
  muted: "#506358"
  quiet: "#576d5f"
  accent: "#1f603e"
  accent-ink: "#ffffff"
  accent-soft: "#eaf4ed"
  warning: "#80500a"
  warning-bg: "#fff3d9"
  error: "#9d342d"
  error-bg: "#fff0ec"
  focus: "#1f603e"
typography:
  display:
    fontFamily: Hanken, Arial, sans-serif
    fontSize: clamp(36px, 4.5vw, 56px)
    fontWeight: 600
    lineHeight: 1.1
    letterSpacing: -0.035em
  body:
    fontFamily: Hanken, Arial, sans-serif
    fontSize: 16px
    fontWeight: 400
    lineHeight: 1.6
rounded:
  control: 8px
  inset: 12px
  panel: 16px
---

# Veritas design

The website helps a visitor upload one image or video, run analysis, and read
a prediction with its limits. The media and result share the work area.
Keep the interface simple, legible, and recognizably Veritas.

## Identity

The logo pairs a solid forest-green tile with a white, geometric V and a
capitalized Hanken wordmark. The mark uses `#174f35` and white. Its source is
`frontend/src/app/icon.svg`; the same asset appears in the header, footer,
and browser tab. The `Brand` component owns both logo placements.

The main mark displays at 40px with a 32px wordmark, reducing to 34px and
26px on small screens. The footer uses a 28px mark and 24px wordmark.
Keep the tile's clear silhouette and surrounding space. Do not add a seal,
authenticity checkmark, decorative period, or thin circle around the V.

## Color and surfaces

The website is **light-only**, at the user's request. It has no theme switch,
theme state, or automatic dark palette. The root declares `color-scheme:
light`, including when the operating system prefers dark mode.

Use semantic CSS properties from `frontend/src/app/globals.css`. A nearly
white green canvas separates the white analysis panel from its surroundings.
The upload area, privacy strip, and score track use a pale inset tone. Panels
use a fine border and no shadow. Amber identifies cautions and red identifies
errors; neither is decorative.

## Typography and spacing

Self-hosted Hanken supplies regular and semibold text. Use the existing 4px
spacing rhythm, tight groups, and clear separation between the main task,
instructions, and FAQs. Body text is 16px; short supporting copy may be
smaller. Headlines use balanced wrapping and tracking no tighter than -0.04em.
The page heading grows to 56px on desktop and is 36px on narrow phones.

The centered shell is at most 1248px wide, including horizontal padding.
The header is 88px tall, reducing to 76px below 768px. Navigation contains
How it works and FAQs, with both links available on phones.

## Upload and results

Keep the file chooser prominent. Images and video share the same explicit
select, preview, and analyze flow. Switching media clears the old selection.
Sample captions sit below their images, with a clear arrow and accessible
button name; their labels describe provenance, not a model prediction.

On desktop the upload target and examples sit side by side. A selected file
reveals the source and assessment panes. Below 768px those groups stack in
the same order. Never crop the uploaded image to imply evidence or obscure
model uncertainty.

The result shows the actual class score and decision threshold without
rescaling. Keep model identity and frame scores in native disclosures.
Failed or unavailable analysis remains an error, not a positive-looking
verdict. Preserve the existing language about scores and model limits.

## Interaction and access

Use Tabler outline icons for controls. Keep visible green keyboard focus,
44px primary action targets, native buttons and disclosures, a skip
link, and announced results. Selection and carets use the accent palette.
Avoid ornamental movement; reduced-motion preference disables transitions
and smooth scrolling.

Verify empty, selected, loading, error, cancellation, offline, and result
states. Test 320, 768, 1024, and 1440px, plus dark system preference: the
website must stay light and retain the same contrast and layout.

Older design explorations in `design/` and `.impeccable/design.json` are
historical references. Current production CSS and this document define the
light-only interface.
