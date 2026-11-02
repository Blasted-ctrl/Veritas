---
status: done
---

# Plan: Verify Veritas and redesign its upload website

## Context

Veritas is an existing research and inference project for detecting manipulated or generated media. It prepares datasets with identity-separated train/validation/test splits, trains ViT image and Wav2Vec2 audio models, exports ONNX graphs, and exposes predictions through FastAPI and a Next.js upload site. The requested outcome is a working, attractive website for image and video uploads, an explanation of why the project exists, and a short README with screenshots of actual analysis.

The current repository ships no trained weights. Its committed image and audio metrics describe synthetic fixtures, so they do not establish real-world detection accuracy. The user chose a downloadable pretrained detector while retaining the existing training tools. Downloads should happen during an explicit setup command, not as a hidden response to an upload. Inference should run on the operator's machine/server, with no third-party upload API or paid inference service required.

Initial bounds: a local developer demo and a single small-instance API with a few simultaneous visitors. Process one file per request, retain the existing 25 MiB upload limit and capped frame sampling, and avoid adding accounts, durable raw-upload storage, or a job queue. Scores and optional image overlays can remain in the existing result cache, which must be bounded, expiring and described to users. These are implementation assumptions, not a promise of capacity for an unrestricted public service. Provide runnable local and Docker instructions; paid cloud provisioning and a production-scale accuracy study are outside this change. Preserve existing audio, training, export, and data-preparation capabilities. The user authorized regular commits and pushes directly to main.

## Acceptance criteria

- A new user can install dependencies, explicitly download/export the documented pretrained image detector, start the API and website, and analyze a supported image without training a model.
- The pretrained model has a pinned source revision, license/attribution, training-domain limitations, correct label mapping and preprocessing, and a manifest identifying its input contract and provenance. Missing or invalid artifacts produce an unavailable state, never a fabricated prediction.
- An uploaded image produces a real API-backed prediction, a preview, and an understandable model score. An uploaded MP4 or GIF produces an aggregate prediction and sampled-frame scores. Video output explains that frame analysis does not verify audio or temporal manipulation.
- UI copy describes model predictions and limitations. Model probability is not presented as calibrated confidence, proof of authenticity, or a real-world accuracy rate. Existing synthetic benchmark results stay explicitly identified as pipeline smoke results.
- The website has a distinct, coherent design, an immediately visible upload action, readable results, explicit mobile layouts, light and dark themes, keyboard access, visible focus, and reduced-motion support. It handles selection, uploading, analysis, success, retry, cancellation, and failure without displaying stale results or leaking preview object URLs.
- Empty, unsupported, corrupt, oversized, missing-model, unreachable-API, and timed-out requests have useful errors. A video with no decodable frames cannot return an authenticity verdict. Failed analysis cannot populate a success state or result cache. Cancelled or obsolete requests cannot update browser success state. Browser cancellation is best effort at the server: successfully completed server work may still be cached after the browser disconnects.
- Allowed browser origins can call health and verification endpoints in local and configured deployments. The README, Compose settings, frontend build-time URL, API environment names, and actual request path agree.
- Cached predictions are scoped to media interpretation, model identity, preprocessing, threshold and video sampling settings (FPS and maximum frame count). The in-memory fallback is bounded and expiring, and remains usable when Redis fails. Replacing a model cannot serve predictions from its predecessor as current results.
- README opening explains the purpose in plain language, followed by verified setup, usage, two actual demo screenshots, and a concise limitations section. Detailed training/architecture material moves to linked docs. Image and video screenshots come from the running app and real inference; no mocked verdicts are used as evidence.
- Existing backend checks and training/export/inference tests pass, frontend lint/type/build pass, and browser checks demonstrate the image and video journeys in both desktop and mobile layouts. Record skipped checks and their cause rather than claiming verification that did not run.

## Pattern survey

### Analogous features

- frontend/src/app/page.tsx: Home already coordinates uploads, health, preview and VerdictCard. Redesign this existing page rather than scaffold a separate website.
- frontend/src/components/UploadDropzone.tsx: UploadDropzone supports image/audio/video/GIF through drag, pointer and keyboard input. It currently submits immediately and silently selects the first file.
- frontend/src/components/VerdictCard.tsx and HeatmapView.tsx: Existing result and real Grad-CAM views can be reorganized; preserve optional explanations without inventing heatmaps.
- backend/src/veritas/api/app.py: create_app supplies /health and /verify, with _verify_image, _verify_audio and _verify_video sharing a Verdict response.
- backend/src/veritas/tasks/app.py: analyze_video runs synchronously through .apply in _verify_video, scores image frames and averages probabilities. The current request path does not distribute work to the deployed worker or analyze audio/motion.
- backend/tests/test_api.py: Integration fixtures export tiny random ViT/Wav2Vec2 graphs and run image/audio/GIF requests. They verify plumbing, not real detection quality.

### Reusable utilities

- frontend/src/lib/api.ts: verify handles multipart uploads and progress; getHealth reports readiness. Add bounded requests and cancellation to these utilities.
- frontend/src/lib/types.ts and backend/src/veritas/api/schemas.py: Verdict, Health and frame contracts anchor frontend/backend compatibility.
- backend/src/veritas/api/inference.py: ImageDetector, AudioDetector and verdict_from_probability implement production scoring. ImageDetector currently assumes ImageNet normalization and fake class index 1, which must be checked against any downloaded model.
- backend/src/veritas/export/onnx_export.py: export_image_onnx and export_image_from_dir reuse the existing logits graph contract. Keep the ONNX serving path free of a mandatory Torch runtime.
- backend/src/veritas/api/explainer.py: GradCamExplainer requires a matching saved Torch model. Preserve it as optional and report unavailable explanations plainly.
- backend/src/veritas/tasks/video.py: extract_frame_pngs already samples/caps frames. Reuse it after fixing cleanup and failed decoding.
- backend/src/veritas/api/cache.py: ResultCache and content_hash provide the existing result cache.
- backend/src/veritas/data/prepare.py, splitting.py and synthetic.py: prepare, split_manifest, assert_no_identity_leakage and generate_dataset retain the existing training-data workflow.

### Convention anchors and observed gaps

- Next.js App Router, TypeScript and Tailwind 3 live in frontend; Python stdlib core plus optional ml/api/dev extras live in backend. No icon library or browser test runner is currently installed.
- No tracked AGENTS.md, PRODUCT.md or DESIGN.md was found in the clone.
- frontend calls localhost:8000 directly but FastAPI has no CORS middleware and Next has no proxy. Browser connectivity needs an explicit solution.
- verify reads the entire upload before its limit check. _verify_video lacks decode-error mapping. analyze_video maps an empty frame list to real with 1.0 confidence. extract_frame_pngs does not close its reader on every error path.
- Model readiness tests file existence only. Cache keys omit model/config/modality, memory entries never expire, and failed Redis writes can be invisible to later reads.
- Home initially labels a not-yet-checked API unreachable and never revokes image object URLs. verify has no cancellation or timeout.
- Compose and Render use REDIS_URL, but Settings expects VERITAS_REDIS_URL. NEXT_PUBLIC_API_URL is compiled into the frontend and cannot be fixed by setting only its runtime Docker environment.
- Next.js 14.2.5 is obsolete; current official security guidance names supported patched 15/16 releases. Upgrade the small application before publishing and keep CI, lint scripts and Docker runtime aligned.

### Incumbent visual audit

Observed in the running 1280x720 page: a centered single-column uploader, system sans body, 36px bold gradient wordmark, dark-only background (#0b1020 to #0a0f1d), pale-blue/indigo title and a 16px-radius dashed upload boundary with 48px padding. No navigation, marketing sections, media preview or result is visible before upload; the lower viewport has extensive unused space. Preserve the Veritas name and upload purpose, replace the visual composition and generic gradient treatment, and improve missing-model guidance. The current interface is sparse and restrained, roughly 3/2/2 on variance/motion/density; the proposed Operate direction remains provisional until design selection.

### Proposed alignment

Retain the existing frontend/backend boundary and model-training tools. Add a reproducible pretrained setup path, correct the inference/browser failures, then replace the visual composition around the working upload task. Keep test models and synthetic fixtures distinct from public-facing detector evidence.

## Implementation steps

1. Establish the runnable baseline and dependency set.
   - Inspect .github/workflows/ci.yml, frontend/package.json, frontend/package-lock.json, frontend/Dockerfile, backend/pyproject.toml and existing deployment configuration.
   - Upgrade Next/React and their lint/type dependencies together to supported patched versions from official release guidance at execution time. Replace next lint with the supported ESLint CLI when required. Align Node in CI and Docker and preserve the existing Tailwind 3 integration unless an actual upgrade need appears.
   - Check free space before dependency/model downloads. At planning time the host had about 520 MiB free, which is insufficient to assume a full model cache, Torch environment and exported graph can coexist. Use a suitable existing drive/environment or resolve capacity before downloading; never delete unrelated user files. Record actual download and working-space requirements.
   - Keep backend requirements compatible with the production exporter/inference APIs; use a Python environment with matching optional wheels. Avoid treating import-skipped ML/API suites as successful inference checks.

2. Add explicit pretrained image-model setup and provenance.
   - New backend/src/veritas/models/pretrained.py: implement the pinned download and artifact preparation entry point. New download-model CLI registration in backend/src/veritas/cli.py calls it lazily, preserving the stdlib-only core.
   - Evaluate an appropriately licensed binary image detector before pinning it. dima806/ai_vs_real_image_detection is a compatible ViT candidate with REAL=0/FAKE=1 and an Apache-2.0 model card, but its author warns of old training data and concept drift. Do not treat its published benchmark as Veritas accuracy or assume suitability for all modern generators/deepfakes. Prefer a better documented current candidate when evidence supports it.
   - Pin the chosen repository revision, download size, expected artifact checksums and license; use local inference, safe weight loading and no remote custom-code execution. Record processor resize/resample/rescale/normalization, class order, graph input/output names, revision and source in a model manifest beside the exported ONNX file. Add processor metadata to the existing training/export path as needed for compatibility.
   - Export via export_image_onnx where the architecture matches. Verify downloaded model outputs against the Torch/Transformers reference for the same pixels before installing the artifact as ready. Add explicit settings/metadata to ImageDetector for class index and processor parameters instead of overwriting its existing training-model assumptions silently.
   - Reconcile custom-model preprocessing as well: backend/src/veritas/models/image_model.py make_image_transform uses torchvision bilinear resize, while existing ImageDetector uses Pillow's default bicubic resize. New custom export metadata and the explicit legacy custom-model fallback must reproduce the actual training transform. Thread the same processor contract through explanations. Verify from a nonuniform raw image with a size different from model input size through training/reference, ONNX preprocessing and explanation inputs; tensor-only export parity and uniform-color fixtures cannot detect this difference.
   - New backend/tests/test_pretrained.py covers manifest validation, reversed labels, nondefault normalization and missing/broken artifacts with small fixtures. Actual checkpoint download and reference/ONNX parity remain a separate real integration check, never replaced by a stubbed download.
   - New additive ModelProvenance/model_info response metadata in backend/src/veritas/api/schemas.py and frontend/src/lib/types.ts carries source type (pretrained/custom), source URL, pinned revision, license and artifact identity where known. Keep the existing model string for compatibility. Populate metadata from the configured artifact, not a hardcoded default; legacy custom models without a manifest must be identified as custom/unknown rather than attributed to the downloaded checkpoint. Wire this through image/video responses, cached responses and VerdictCard; cover pretrained, custom and cached cases in API/contract tests.
   - Reuse the same metadata-driven ImageDetector in image and video paths. Optional heatmaps must use the same checkpoint and compatible preprocessing as the prediction or remain unavailable.

3. Repair the browser/API and upload lifecycle.
   - In backend/src/veritas/api/config.py and app.py, add configurable explicit allowed origins and CORS middleware for health and verification, with local frontend defaults. Keep the documented NEXT_PUBLIC_API_URL contract rather than adding a second proxy transport.
   - Read upload content only through the maximum plus one byte, reject oversize input before inference, and keep the temporary-file cleanup guarantee. Preserve existing API status semantics; map corrupt/empty videos to a clear 422 response.
   - In tasks/video.py, close readers in finally. In tasks/app.py, reject no-frame results rather than producing a score. Reuse the already-configured image detector inside the synchronous API video path where practical, preserving the standalone task entry point. Document the synchronous execution path truthfully; do not introduce async job state.
   - In api/inference.py, validate graph/config readiness without exposing sensitive paths in client errors. Unloadable models produce an explicit readiness or service error rather than healthy status or a fake success.
   - Update .env.example, docker-compose.yml, render.yaml and frontend/Dockerfile so Redis prefixes, model manifests/mounts, build-time API URL and allowed origins agree. Model weights remain untracked. Correct worker claims and provide a small-instance run path without requiring an unused distributed worker.
   - Extend backend/tests/test_api.py and new focused video tests for allowed/disallowed origins, oversize boundary, real MP4/GIF decode, corrupt video, zero frames, unavailable/corrupt models and temporary-file/reader cleanup.

4. Make cached results match the active model and request semantics.
   - Update ResultCache and create_app to namespace scores by immutable artifact/model identity, preprocessing, threshold, routed modality, and (for video) video_fps/max_video_frames while retaining the content digest in the response.
   - Bound and expire in-memory entries. On Redis failure, transition reads and writes consistently to memory. Avoid retaining overlays in memory beyond the stated bound and expiry.
   - Extend backend/tests/test_cache.py with deterministic expiry, eviction and Redis-failure cases; add API tests where identical bytes with different model/config interpretations cannot reuse stale scores, including two apps sharing Redis/cache storage but using different video_fps/max_video_frames settings. Do not build a generalized cache/versioning framework.

5. Set the product/design contract and redesign the upload site.
   - Inspect the running incumbent interface on desktop and mobile first. Write PRODUCT.md with confirmed facts and open design choices. Use the requested design workflow to resolve the visual direction before UI edits; write DESIGN.md, a reusable token model and a route-specific surface brief from that choice.
   - Mode: Operate for the uploader/results, Read for explanation/docs. Apply marketing guidance only to explanatory landing content. Provisional dials: DESIGN_VARIANCE 4, MOTION_INTENSITY 3, VISUAL_DENSITY 4, reflecting trust and task clarity. Exact colors/type are not approved yet; use one restrained accent, consistent semantic tokens, limited radii and both themes. Keep the Veritas name.
   - Use Hue to record the agreed design language and Impeccable to review composition and states. Select one licensed icon family after observing the incumbent glyphs; install it before imports. Self-host fonts or use next/font. Generate only useful sample/section imagery with clear provenance; never use generated product screenshots as proof.
   - Redesign frontend/src/app/page.tsx, layout.tsx, globals.css, UploadDropzone.tsx, VerdictCard.tsx, HeatmapView.tsx and ConfidenceBar.tsx as needed. Keep a visible upload action and a useful file preview beside readable results; stack explicitly on mobile. Put technical metadata in a disclosure and explain what the score covers.
   - Add selection validation matching server limits, clear upload versus analysis progress, a cancellable request, retry/replacement, preview cleanup, accurate initial health loading and capability-specific unavailable states. Keep existing audio support without making it appear ready when no audio model exists.
   - Add timeout/abort handling in frontend/src/lib/api.ts. Home owns the active request, cancels before replacement/unmount, and ignores obsolete completions. Revoke each object URL when replaced or unmounted. A cancelled request must not later overwrite the current result; server cancellation is best effort and must not be promised.
   - Render results from actual Verdict values, include filename/media preview, score explanation, model attribution/limitations, sampled-frame evidence and optional real overlay. Add polite live announcements, focus handling and accessible text equivalents for frame charts. Avoid fabricated confidence/accuracy claims or a default model-backed-looking sample.

6. Verify real journeys and capture reproducible demo evidence.
   - Add bounded browser integration coverage, using Playwright or available DevTools, for choosing and dropping a file, image analysis, MP4/GIF analysis, second upload, keyboard control, cancellation/retry, capability errors, timeout and responsive themes. At least one image and one video case must reach the real API and pinned model; mocked errors can supplement them.
   - New scripts/capture-demo.* and docs/demo/ assets reproduce screenshots from the actual running app. Record fixture provenance, model revision, capture procedure, and which outputs were actually observed. Use a small openly licensed camera image and clearly attributed generated sample; any constructed video must be described as constructed from those frames.
   - Capture desktop upload/result and video-result screenshots, plus check 390px mobile and both themes. No browser-console errors, horizontal overflow or inaccessible upload controls. Run Lighthouse on a production build and report measured results without inventing performance numbers.

7. Rewrite the README around purpose and use.
   - Opening: Veritas helps people inspect whether media resembles the kinds of generated/manipulated content its model learned. Explain why inspecting provenance matters, what the tool returns, and why predictions require other evidence.
   - Present one verified first-run path: dependency setup, explicit model download, API start and frontend start, then a Docker alternative. Include Windows PowerShell and POSIX commands where syntax differs.
   - Embed actual image and video screenshots via relative repository paths with concise alt text and captions identifying their model/demo context.
   - Explain actual data handling: files are sent to the configured API; temporary video files are removed; cached scores and optional image overlays expire under the configured policy. Do not claim uploads never leave the browser or that caching stores no media-derived imagery.
   - Keep capabilities and limits close to usage: supported file types/25 MiB, frame caps, no temporal/audio analysis for video, optional audio/heatmaps, no universal authenticity guarantee, and synthetic-only historical metrics.
   - Move detailed data prep/training/export/architecture material to docs/development.md; document model source/license and pipeline behavior in docs/model-card.md. Remove phase checklists, unverified green-CI claims and placeholder live-demo links. Only include a hosted URL if it was actually deployed and verified.

## Verification

- Baseline observed at cloned main commit 076418b: 25 tests in test_datasets.py, test_manifest.py, test_prepare.py and test_splitting.py passed. Ruff lint passed, format check reported 45 formatted files, and mypy passed on 9 core source files. Frontend lint and type checking also passed. The existing page rendered at 1280x720 with no logged browser errors and no horizontal overflow; it displayed API unreachable as expected without a running API. Full inference and upload-journey verification has not yet completed; no trained weights or ONNX Runtime were available at initial inspection. Dependency setup and optional imports were slow on this host. Record remaining baseline outcomes as they become available.
- Core: from backend, run pytest on core files and ruff check src tests, ruff format --check src tests, and mypy. Isolate optional imports and enforce an external timeout so a missing dependency or stuck Redis connection does not look like a green suite.
- ML/export/API: install compatible CPU ml/api dependencies; run pytest -m ml and pytest -m api against existing tiny models. Add pretrained manifest/parity, custom raw-image resize parity, video error/cleanup and cache-isolation tests (including FPS/frame caps). Report collection skips explicitly. Existing synthetic models establish integration only.
- Real checkpoint: run the explicit download/export command, validate provenance, compare reference and ONNX predictions, then POST an image, MP4 and GIF to the running API. Check per-frame aggregate consistency and repeated-upload caching. Record observed predictions without claiming these few samples establish accuracy.
- Connectivity/deployment: browser calls from an allowed origin must succeed; disallowed-origin browser access must fail. Health must distinguish pending, unreachable and model-unavailable states. Test the documented build-time API URL and clean Docker/local startup with model files at the documented mount path. Do not call an unexecuted Render/Vercel blueprint verified.
- Error behavior: empty/unsupported/oversize uploads, corrupt images/videos, zero-frame extraction, missing/broken models, Redis loss, timeout, cancel and replacement. Verify failed analysis cannot become cached success and cancelled/obsolete requests cannot become stale browser success. A successful server prediction may finish and be cached after browser cancellation; exercise this case without promising server interruption. Verify temporary files/object URLs are cleaned up.
- Frontend: run the updated lint, typecheck and production build scripts; inspect live desktop/mobile output in both themes and keyboard/reduced-motion settings. Verify score semantics, live regions, contrast, charts, state transitions and real API/network results. Run Lighthouse after changes settle.
- README/demo: follow the quickstart from a clean environment; run the capture script, inspect every screenshot, and check all relative links/assets. The image and video screenshots must show real inference and be reproducible with the stated checkpoint and fixture provenance.

## Context files

- README.md: current project claims and documentation to simplify.
- backend/src/veritas/api/app.py, config.py, inference.py and schemas.py: current production request and model contracts.
- backend/src/veritas/tasks/app.py and video.py: synchronous video execution, aggregation and decoding.
- backend/src/veritas/export/onnx_export.py and backend/src/veritas/cli.py: model setup/export integration points.
- backend/tests/test_api.py and test_export.py: tiny-model production-counterpart tests.
- frontend/src/app/page.tsx, frontend/src/lib/api.ts and types.ts: browser upload lifecycle and response consumers.
- .github/workflows/ci.yml, docker-compose.yml and frontend/Dockerfile: verification and deployment behavior.

## External references

- FastAPI CORS: https://fastapi.tiangolo.com/tutorial/cors/
- Next.js current security release guidance: https://nextjs.org/blog/nextjs-security-update-october-14-2026
- Next.js upgrade guidance: https://nextjs.org/docs/app/guides/upgrading
- Torchvision resize default: https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html
- Pillow resize default: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize
- Candidate model card and author limitations: https://huggingface.co/dima806/ai_vs_real_image_detection
- Candidate class labels/config: https://huggingface.co/dima806/ai_vs_real_image_detection/raw/main/config.json

## Planning decisions and remaining limits

The user selected Plan mode and approved a downloadable pretrained detector while keeping training tools. Direct pushes to main are authorized. Exact model revision, visual world and hosting capacity are not yet proven or locked; the implementation must resolve and validate them before claiming the site is complete. No code, detector accuracy, public deployment or final website screenshot is certified by this plan alone.

## Completion record

Implemented on main. The default detector is downloadable, pinned, and checksum verified; processor and raw-model parity passed against its Torch reference. Browser image and MP4 analysis, provenance, cancellation, recovery, sampled-frame evidence, and both responsive themes are working. README contains real inference captures and explains the observed NASA false positive. Original training/export tools remain.

CI at 68440a1 passed core, ML, API/ONNX, frontend, and production-container gates. The independent Impeccable finish review returned ship, and the documenter wrote DESIGN.md and .impeccable/design.json. The Hue design skill and four previews are included and validated. See docs/verification/README.md for exact evidence and limits. No public hosting deployment or real-world accuracy benchmark is claimed.
