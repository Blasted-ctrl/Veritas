# Veritas components

## 1. BUTTONS

| Variant | Background | Text | Radius | Height |
| --- | --- | --- | --- | --- |
| Primary | accent | accent_ink | 8px | 46px |
| Secondary | surface2 | text1 | 8px | 46px |
| Ghost | transparent | accent | 8px | 36px |

Use Hanken 14px/600, padding 10px 22px. Hover brightness 0.92; pressed translation 1px. Disabled opacity 0.5. Focus outline 3px, offset 4px. A primary action has one clear verb.

## 2. CARDS / SURFACES

The standard panel uses surface1, a 1px border, 16px corners, and 28px inner padding. Featured and compact variants change spacing, not decorative depth. Keep the source pane and assessment inside the same panel, separated by a divider. Avoid nested cards.

## 3. INPUTS

Use a native file input with a visible Choose a file control. The drop target has a dashed border, 12px corners, and a visible drag state. Derived text inputs use surface1, 46px height, 8px corners, and 12px 16px padding. Labels stay visible. Focus uses the shared outline; errors include recovery text. Textareas retain the same anatomy and allow vertical resizing.

## 4. LISTS / DATA ROWS

File rows show media type, an ellipsized filename, file size, and a labeled remove control. Frame tables show sampling order and a percentage; numerals align using tabular figures. Selected rows use accent_subtle and a visible selected label. Long hashes wrap inside their own column.

## 5. NAVIGATION / TAB BAR

A wordmark anchors a compact header. Links use body-sm and persistent underlines on hover and focus. Theme controls retain a text alternative. Mobile hides optional desktop links while the product task and GitHub remain reachable. No product tab bar is currently needed.

## 6. TAGS / STATUS

Status dots are 6px and always accompanied by readable text. Colors alone never communicate availability or a verdict. Derived tags use 4px 12px padding, 28px height, and pill corners.

## 7. OVERLAYS

This flow uses no modal, tooltip, toast stack, or popover. Error messages remain inline with the upload. Native details/summary reveals model metadata and per-frame scores without hiding the media.

## 8. FEEDBACK AND EMPTY STATES

Empty: explain accepted media and provide the file action. Upload: show actual upload progress. Analysis: distinguish completion of transfer from inference. Cancel: keep the selected file ready to retry. Error: name the failure and offer recovery. Result: show server verdict, score, threshold, provenance, and limits. Never invent analysis progress or prefill a synthetic result in the working app.
