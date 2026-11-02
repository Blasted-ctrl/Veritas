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

## Repeat the browser checks

From `frontend`, install the locked dependencies and isolated test browser:

```sh
npm ci
npx playwright install chromium
npm run build
npm run test:browser
```

The nine browser regressions use controlled API responses to exercise keyboard selection, dropping a file, explicit analysis, oversize rejection, cancellation and replacement, retry, unavailable audio/API states, mobile themes, and a native request timeout.

Only the timeout test shortens its browser’s timer; the application keeps its three-minute limit.

The suite owns port 3010, refuses an existing server there, and has a three-minute suite deadline.

For real inference, start the API with the downloaded model as described in [development.md](../development.md). Stop your development frontend, then run `npm start` from `frontend` to serve the production build on port 3000. In another terminal in that directory, run these commands one after the other:

```sh
npm run capture:demo
npm run audit
```

The capture command submits the included generated PNG, constructed MP4, and a four-color GIF decoder fixture to the actual API. It compares displayed scores with responses, checks video averages, rejects console errors and horizontal overflow, and saves eight desktop/mobile screenshots in both themes.

The manifest includes responses, model identity, fixture credits, and capture time. These files go to `.turbo/demo-captures`; the production-container workflow publishes them as a run artifact. The README uses the committed captures in [screenshots](../screenshots).

The audit command runs Lighthouse against the initial upload page in mobile and desktop modes. Raw reports and a compact summary go to `.turbo/lighthouse`. Each mode has a five-minute deadline. The command closes its isolated browser when it finishes or fails. These lab measurements cover page loading and automated audits, not inference speed or detector accuracy. They vary with machine load and do not replace keyboard or visual checks.

Both commands accept `-- --url http://127.0.0.1:3000 --output <directory>`. To use an installed Chrome for Playwright, set `PLAYWRIGHT_CHANNEL=chrome`. Lighthouse can use `CHROME_PATH` to select its executable. Neither command attaches to your normal browser profile.

## Recorded production audit

[The saved Lighthouse summary](lighthouse-summary.json) records one final local run per mode with Lighthouse 13.6.0 and Chrome 156 on Windows. The production build used the default local API configuration.

| Mode | Performance | Accessibility | Best practices | SEO | LCP | Blocking time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mobile | 69 | 100 | 100 | 100 | 0.98 s | 1.81 s |
| Desktop | 90 | 100 | 100 | 100 | 0.38 s | 0.26 s |

Mobile JavaScript execution remains a performance cost. These are single lab measurements on a shared machine, not field data or detector accuracy. The summary retains the failed audits and earlier diagnostic scores, including the initial mobile score of 74. The intermediate CLI wrapper wrote reports but timed out during shutdown; it is not counted as a completed audit.

The final runner uses Lighthouse’s programmatic API and closes a directly owned Playwright browser. Its successful run logged no browser errors. The wordmark uses its native text name, and the favicon returns successfully.

[The browser capture manifest](browser-capture.json) records actual PNG, MP4, and GIF predictions with the pinned model. Both video formats produced four frame scores and matching averages. Screenshot filenames in that manifest refer to the generated capture directory. All eight desktop/mobile, light/dark captures were visually inspected. The independent HTTP/asset/manifest check passed 21 cases.

The nine controlled browser regressions passed locally. Independent code review found one audit-cleanup issue, which the direct-browser implementation resolved. The optional Codex CLI peer could not run with the account’s configured model; an independent agent review was used instead.
