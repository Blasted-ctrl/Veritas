# Verification record

The implementation at commit 68440a1a01af8a6964e35e126c879ed0561b518d passed the [CI workflow](https://github.com/Blasted-ctrl/Veritas/actions/runs/41361447171) and [production-container smoke test](https://github.com/Blasted-ctrl/Veritas/actions/runs/41361447181).

| Gate | Recorded result |
| --- | --- |
| Core backend tests | 39 passed; optional modules skipped without their dependencies |
| ML smoke tests | 11 passed; non-ML scopes deselected or unavailable modules skipped |
| API / ONNX tests | 39 passed, 50 outside the API scope deselected |
| Frontend | ESLint, TypeScript, and production build passed |
| Production containers | Both images built; verified model installed as non-root; image and MP4 inference passed |
| Configurable browser API URL | Nondefault URL found in the production client bundle |
| Local API regressions | 22 passed, including actual MP4 sampling, invalid uploads, model-aware cache identity, and cleanup |
| Local Python quality | Ruff checks/format and mypy passed |
| Pretrained numerical parity | Two included images matched the reference processor, logits, and class scores within declared tolerances |

See pretrained-parity.json for exact scores, tolerances, model hashes, and reference revision. The processor difference was below 1.2e-7; maximum raw-logit difference was below 3.6e-7. Numerical agreement verifies conversion and preprocessing, not detection quality.

Browser testing exercised real image and MP4 uploads, sample selection, native video playback controls, explicit analysis, cancellation and retry, invalid previews, corrupt image errors, oversize rejection, keyboard file selection, frame tables, model details, both themes, and mobile stacking. At a 390px CSS viewport, the document content width was 380px with the scrollbar and no horizontal overflow.

The screenshots in ../screenshots are real local browser captures using the pinned default model. A fresh Impeccable review returned ship at the scope of the selected Conservation study card; no approved raster comp existed for this code-led build. The review found no material fixes. DESIGN.md records the finished design system.

The full local optional-dependency test run on Windows reached its time limit after its first API cases passed. The complete scoped ML and API jobs passed on GitHub’s Linux runner; the local timeout is not counted as a full-suite pass. Docker was unavailable locally, so the production-container gate ran on GitHub.
