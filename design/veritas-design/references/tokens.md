# Veritas tokens

## 0. PRIMITIVES

### neutral

| Step | Hex |
| --- | --- |
| 50 | #fafbf8 |
| 100 | #eef1ec |
| 200 | #e2e8df |
| 300 | #c9d2c8 |
| 400 | #9cafa0 |
| 450 | #a6b8a9 |
| 500 | #5c6c61 |
| 600 | #52635b |
| 700 | #2c4839 |
| 750 | #385344 |
| 800 | #203b34 |
| 900 | #1d3025 |
| 950 | #15231d |

### brand

| Step | Hex |
| --- | --- |
| 50 | #e5eddc |
| 100 | #d8e5cb |
| 200 | #b9d0a5 |
| 300 | #95b282 |
| 400 | #6e915c |
| 500 | #416b4b |
| 600 | #355b3e |
| 700 | #294b33 |
| 800 | #223e2b |
| 900 | #1b3223 |
| 950 | #182c1c |

### red

| Step | Hex |
| --- | --- |
| 50 | #f6e5e1 |
| 400 | #edb4a9 |
| 500 | #943f36 |
| 900 | #452920 |

### green

| Step | Hex |
| --- | --- |
| 50 | #eaf3ea |
| 500 | #4b8a65 |
| 900 | #1a3022 |

### amber

| Step | Hex |
| --- | --- |
| 50 | #f3e8d2 |
| 400 | #e7c48b |
| 500 | #855917 |
| 900 | #3b301d |

Spacing: 0, 2, 4, 8, 10, 12, 16, 20, 22, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 96px.

Radii: 4, 8, 12, 16, 999px.

## 1. TYPOGRAPHY

Self-host Hanken Grotesk from assets/. Its OFL license is included. Use ui-monospace only for source code and hashes. Use tabular numerals for scores.

| Token | Size | Line height | Tracking | Weight |
| --- | --- | --- | --- | --- |
| display | 64px | 1.04 | -0.035em | 600 |
| heading | 32px | 1.2 | -0.025em | 600 |
| subheading | 22px | 1.3 | -0.015em | 600 |
| body | 16px | 1.6 | 0em | 400 |
| body-sm | 14px | 1.6 | 0em | 400 |
| caption | 12px | 1.5 | 0em | 400 |
| label | 14px | 1.4 | 0em | 600 |

## 2. COLOR SYSTEM

### light

| Role | Primitive | Hex |
| --- | --- | --- |
| background | {neutral.100} | #eef1ec |
| surface1 | {neutral.50} | #fafbf8 |
| surface2 | {neutral.200} | #e2e8df |
| surface3 | {neutral.300} | #c9d2c8 |
| border | {neutral.300} | #c9d2c8 |
| border_visible | {neutral.500} | #5c6c61 |
| text1 | {neutral.800} | #203b34 |
| text2 | {neutral.600} | #52635b |
| text3 | {neutral.500} | #5c6c61 |
| text4 | {neutral.500} | #5c6c61 |
| accent | {brand.500} | #416b4b |
| accent_subtle | {brand.50} | #e5eddc |
| accent_ink | {neutral.50} | #fafbf8 |

### dark

| Role | Primitive | Hex |
| --- | --- | --- |
| background | {neutral.950} | #15231d |
| surface1 | {neutral.900} | #1d3025 |
| surface2 | {neutral.800} | #203b34 |
| surface3 | {neutral.700} | #2c4839 |
| border | {neutral.750} | #385344 |
| border_visible | {neutral.400} | #9cafa0 |
| text1 | {neutral.50} | #fafbf8 |
| text2 | {neutral.300} | #c9d2c8 |
| text3 | {neutral.450} | #a6b8a9 |
| text4 | {neutral.450} | #a6b8a9 |
| accent | {brand.200} | #b9d0a5 |
| accent_subtle | {brand.950} | #182c1c |
| accent_ink | {brand.950} | #182c1c |

Status tints come from red, green, and amber ramps. Keep body copy in the normal text tokens; use status colors for icons and larger labels.

## 3. SPACING AND RADII

Use 8px for tight groups, 16px inside groups, 24px between controls, and 48 to 96px between sections. Controls 8px; panels 16px; small indicators 999px.

## 4. ELEVATION

Flat. One border, no shadow.

## 5. MOTION

Mechanical feedback, 120ms fast and 180ms normal, ease-out. Content is visible by default. Reduced motion disables transitions and entrance motion.

## 6. ICONOGRAPHY

Tabler outline, 1.5px stroke, soft terminals, geometric forms. It matches the original upload glyph more closely than a heavier or more humanist set. It is a chosen open kit, not a claim about a proprietary brand.

## 7. HERO STAGE

Flat blank ground. No composed subject, texture, glow, or background motion. Uploaded media supplies the image.
