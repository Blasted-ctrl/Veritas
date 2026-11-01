---
name: Veritas
description: 'Conservation study: a quiet green interface for inspecting media and model evidence.'
colors:
  bg: '#eef1ec'
  surface: '#fafbf8'
  surface-2: '#e2e8df'
  line: '#c9d2c8'
  ink: '#203b34'
  muted: '#52635b'
  quiet: '#5c6c61'
  accent: '#416b4b'
  accent-ink: '#fafbf8'
  accent-soft: '#e5eddc'
  warning: '#855917'
  warning-bg: '#f3e8d2'
  error: '#943f36'
  error-bg: '#f6e5e1'
  focus: '#416b4b'
  dark-bg: '#15231d'
  dark-surface: '#1d3025'
  dark-surface-2: '#203b34'
  dark-line: '#385344'
  dark-ink: '#fafbf8'
  dark-muted: '#c9d2c8'
  dark-quiet: '#a6b8a9'
  dark-accent: '#b9d0a5'
  dark-accent-ink: '#182c1c'
  dark-accent-soft: '#182c1c'
  dark-warning: '#e7c48b'
  dark-warning-bg: '#3b301d'
  dark-error: '#edb4a9'
  dark-error-bg: '#452920'
  dark-focus: '#b9d0a5'
typography:
  display:
    fontFamily: Hanken, Arial, sans-serif
    fontSize: clamp(46px, 4.7vw, 64px)
    fontWeight: 600
    lineHeight: 1.04
    letterSpacing: -0.035em
  headline:
    fontFamily: Hanken, Arial, sans-serif
    fontSize: 30px
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: -0.025em
  title:
    fontFamily: Hanken, Arial, sans-serif
    fontSize: 19px
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: -0.015em
  body:
    fontFamily: Hanken, Arial, sans-serif
    fontSize: 16px
    fontWeight: 400
    lineHeight: 1.6
  button:
    fontFamily: Hanken, Arial, sans-serif
    fontSize: 14px
    fontWeight: 600
    lineHeight: 1.6
  label:
    fontFamily: Hanken, Arial, sans-serif
    fontSize: 12px
    fontWeight: 400
    lineHeight: 1.6
  code:
    fontFamily: ui-monospace, monospace
    fontSize: 11px
    fontWeight: 400
    lineHeight: 1.5
rounded:
  chart-bar: 2px
  small: 4px
  control: 8px
  inset: 12px
  panel: 16px
spacing:
  '4': 4px
  '8': 8px
  '12': 12px
  '16': 16px
  '20': 20px
  '24': 24px
  '28': 28px
  '32': 32px
  '36': 36px
  '48': 48px
  '56': 56px
  '64': 64px
  '72': 72px
  '80': 80px
  '88': 88px
  '96': 96px
components:
  button-primary:
    backgroundColor: '{colors.accent}'
    textColor: '{colors.accent-ink}'
    typography: '{typography.button}'
    rounded: '{rounded.control}'
    padding: 10px 22px
  button-secondary:
    backgroundColor: '{colors.surface-2}'
    textColor: '{colors.ink}'
    typography: '{typography.button}'
    rounded: '{rounded.control}'
    padding: 10px 22px
  button-text:
    textColor: '{colors.accent}'
    padding: '0'
    size: 13px
  button-icon:
    textColor: '{colors.ink}'
    rounded: '{rounded.control}'
    height: 36px
    width: 36px
  analysis-panel:
    backgroundColor: '{colors.surface}'
    textColor: '{colors.ink}'
    rounded: '{rounded.panel}'
  upload-area:
    rounded: '{rounded.inset}'
    padding: 30px 16px 23px
  notice-warning:
    backgroundColor: '{colors.warning-bg}'
    textColor: '{colors.warning}'
    rounded: '{rounded.control}'
    padding: 14px 16px
  notice-error:
    backgroundColor: '{colors.error-bg}'
    textColor: '{colors.error}'
    rounded: '{rounded.control}'
    padding: 14px 16px
---

# Design System: Veritas

## Overview

**Creative North Star: "Conservation study"**

Conservation study gives Veritas a calm, precise visual identity: cool green grounds, deep ink, open grotesk lettering, and flat panels. Hierarchy comes from type, spacing, fine rules, and restrained changes in tone. Real media and readable evidence carry the visual interest.

This document records the production website in `frontend/src/app/globals.css`, `page.tsx`, `layout.tsx`, and `components/VerdictCard.tsx`. The separate illustrative Hue kit at `design/veritas-design` is a reference artifact; its examples do not override the website tokens recorded here.

**Key Characteristics:**

- Cool green tonal surfaces in light and dark themes.
- Hanken Grotesk with compact headings and plain sentence-case controls.
- Flat panels, thin separators, and Tabler outline icons.
- Evidence, class scores, and their limits remain readable together.

## Colors

The palette is a cool green tonal field with a restrained botanical accent; amber and red are reserved for meaningful status.

### Primary

- **Botanical accent** (`accent`): primary file and analysis actions, score fills, and small emphasis. `accent-ink` supplies the contrasting foreground and `accent-soft` the dragging ground.
- **Focus** (`focus`): the keyboard outline follows the accent in both themes.

### Secondary

- **Amber caution** (`warning`, `warning-bg`): unavailable-service notices and the generated-pattern prediction icon.
- **Clay error** (`error`, `error-bg`): failed requests and invalid uploads.

### Neutral

- **Cool green ground** (`bg`): the page canvas.
- **Pale study surface** (`surface`): the main analysis panel.
- **Tonal inset** (`surface-2`): media ground, secondary actions, icon backing, and score tracks.
- **Fine green rule** (`line`): panel edges, pane divisions, metadata, and tables.
- **Deep green ink** (`ink`): headings and primary text. `muted` carries explanations; `quiet` carries supporting labels.

The unprefixed frontmatter colors are the light palette. Every `dark-*` token records the corresponding dark override of the same CSS property. Explicit theme choice overrides the system preference; otherwise the system dark preference selects the dark palette. These are production colors, not the synthesized tonal ramps shown in the sidecar.

**The Semantic Theme Rule.** Use the semantic CSS properties for both themes; warning, error, and focus retain their roles when the palette changes.

## Typography

**Display and body font:** Hanken Grotesk, registered locally as `Hanken`, with Arial and sans-serif fallbacks. Regular is self-hosted at weight 400; the semibold face is registered for weights 600–800. The visible hierarchy mainly uses 400 and 600. Code and hashes use the system monospace stack.

The type is open and direct, with tight heading tracking and ordinary sentence-case controls. The frontmatter records the reused roles; responsive and state-specific adjustments follow here.

- **Display:** introduction heading; fluid on desktop and fixed at 46px below the mobile breakpoint. Its selected-file state contracts to 34px with a 1.15 line height, then 30px on mobile.
- **Headline:** explanatory section headings and prominent score numerals. Section headings become 28px on mobile; score values use tabular numerals.
- **Title:** the analysis panel heading; 18px on mobile. Result predictions use 26px, 1.2 line height, and -0.025em tracking, then 25px on mobile.
- **Body:** base reading text. Introductory copy uses 18px, becoming 16px on mobile; selected-file introductory copy uses 15px, then 14px. Paragraphs cap at 70ch, with narrower local explanatory measures.
- **Button and label:** compact action and supporting text. Small score notes, privacy text, and sample attribution use 10–11px where implemented; do not promote these small annotations into a new general body size.

## Layout

The centered shell is 1280px wide including horizontal padding of 56px. At widths of at least 1450px it grows to 1400px with 72px padding. At 1000px and below the padding becomes 32px, then 20px at 720px and below.

The initial desktop hero pairs introduction and upload at 0.9fr / 1.1fr with a 72px gap and 80px / 88px vertical padding. At the intermediate breakpoint it uses 0.85fr / 1.15fr with a 36px gap. At the large breakpoint its gap and top padding become 96px. The header is 112px tall, or 84px on mobile.

Selecting a file creates a compact introduction above the full-width workbench. Source and assessment use adjacent 1.08fr / 1fr panes. The source has 24px padding; the assessment 28px, reduced to 24px at the intermediate breakpoint. Source media uses a contained 4:3 preview. At 720px and below the hero and workbench stack, source precedes assessment, the separating rule becomes horizontal, and the preview becomes 3:2. Mobile source padding is 20px and assessment padding is 24px 20px.

Spacing follows the observed steps in the frontmatter, with 20–28px component insets and larger 36–96px structural intervals. Supporting sections use equal desktop columns with 80px gaps; mobile uses one column and 24px gaps. Sample choices become a single column below 1000px. Mobile navigation retains the final external link and theme control rather than adding a drawer.

## Elevation & Depth

There are no box shadows. The panel, preview, controls, and notices are separated by tonal surfaces, fine borders, and whitespace. Keyboard focus uses a 3px outline in the semantic focus color with a 4px offset; it is an interaction indicator, not elevation.

**The Flat Surface Rule.** Separate surfaces with tone and fine borders. The production interface uses no box shadows.

## Shapes

Outer panels use the panel radius; action controls, notices, and media previews use the control radius. The upload inset and its icon ground use the inset radius. Samples and score tracks use the small radius; chart bars use the smaller chart-bar radius. Ordinary borders are 1px. Upload targeting uses a dashed border; status dots are circles. Tabler SVG outlines use a 1.5px stroke. The existing wordmark is an identity treatment rather than a reusable pictogram construction rule.

## Components

### Buttons

Primary actions use the accent and its contrasting ink; secondary actions use the tonal inset and line border. Both share the button type role, control radius, 10px 22px padding, 46px minimum height, and a 24px icon gap. Primary hover applies brightness(0.92). All buttons use 120ms ease-out transitions for background, color, and transform; pressing an enabled button shifts it down 1px. Disabled buttons reduce opacity to 0.5 and use a not-allowed cursor.

Text actions are underlined, 13px, and accent-colored. Icon buttons are transparent 36px squares, becoming tonal on hover; the theme button also has a fine border and becomes 32px square on mobile.

### Cards / Containers

The main analysis panel combines the surface ground, fine line border, panel radius, and clipped overflow. Its heading uses 24px 28px padding on desktop. A selected file reveals internal source and assessment panes rather than replacing the source with a result-only screen. The source remains steady through request, failure, and result states.

### Inputs / Fields

The actual input is a visually hidden native file input controlled by the visible file-selection button. There is no styled text-field primitive. The upload target has an inset radius, dashed quiet border, and centered prompt; dragging adds the soft accent ground and a 2px accent outline. Keep file selection available as a standard button interaction.

### Navigation

The compact wordmark anchors the header. Text links use the body family at 14px, have no resting underline, and underline on hover. The navigation sits at the right edge with 32px gaps. At 720px and below only its final link remains visible, at 12px, alongside the theme control. The keyboard skip link appears at the top when focused.

### Notices

Caution and error notices share a control radius, 14px 16px padding, 13px text, and an outlined status icon. Their semantic foreground and ground colors change together by theme. Recovery actions remain plain underlined text.

### Assessment and frame evidence

The prediction leads with an outline icon and a short heading. Supporting copy states that the result does not establish authenticity. The generated class score uses tabular numerals, an 8px tonal track, an accent fill, and a fine threshold marker; labels describe the two pattern classes. The score is explicitly described as a class score rather than a calibrated probability.

Video evidence adds a compact bar sequence and an expandable table of real sampled frame scores. Model details use native disclosures and a two-column definition list with wrapping hashes. This evidence does not claim audio or temporal analysis. Result arrival is a single 180ms ease-out fade from 0.7 opacity and a 4px upward settle. Reduced-motion preference removes animation and transitions and disables smooth scrolling.

## Do's and Don'ts

### Do:

- Do use the production semantic properties so light and dark themes retain the same hierarchy.
- Do keep real media visually prominent and place the model assessment beside it on desktop, then below it on mobile.
- Do present a class score with its threshold, model identity, and visible limits.
- Do use Tabler outline icons with the production stroke weight and preserve visible keyboard focus.

### Don't:

- Don't substitute the illustrative Hue kit's example styles for production CSS.
- Don't turn a model score into an authenticity seal or a calibrated probability claim.
- Don't introduce decorative shadows or motion that changes the quiet, flat surface language.
- Don't hide unavailable analysis behind a positive-looking result.
