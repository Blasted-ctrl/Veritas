# Veritas product page

The page helps someone check one image or video without learning the project first. The primary action is choosing a file, followed by an explicit analysis.

Reference: [Deepfake Detection image tool](https://deepfakedetection.io/deepfake-image-detection), inspected November 2, 2026. Its centered tool heading, media selector, visible upload area and sample images make the next action clear. Veritas uses those ideas with its own green identity, existing sample media and working analysis flow. No copied branding, invented usage statistics, testimonials or unsupported authenticity claims.

Design dials: variance 3, motion 2, density 4. White and neutral surfaces, green actions, self-hosted Hanken typography, restrained borders. Keep light and dark themes. Move GitHub and implementation information out of the primary upload flow; preserve the existing how-it-works and about-model anchors.

Required states: empty, selected preview, uploading, analyzing, cancelled, result, invalid/oversize file, unavailable service, unavailable model, failed preview and sample loading. Switching media clears and cancels the previous selection. Samples load into the same explicit preview-and-analyze flow as uploads. Results show the verdict and score first; model identity, thresholds and hashes remain in a disclosure.

At 320 and 768 pixels the upload comes before examples and results stack below the preview. At 1024 and 1440 pixels the tool uses two columns. No horizontal scrolling, keyboard-only dead ends, auto-running samples, fake scores, or decorative dashboard controls. Verify real image/video inference, cancellation, retry, both themes and reduced motion.

## Verification

The production build and ESLint pass. Fourteen browser tests cover explicit upload, drag and drop, oversize rejection, cancellation, stale responses, sample selection, service recovery, timeouts, reduced motion, long filenames and both themes at 320, 768, 1024 and 1440 pixels. Axe reports no WCAG 2 A/AA or 2.1 AA violations on the upload and result screens in either theme. Keyboard analysis moves focus to the result heading.

The [demo responses](demo-responses.json) record real image, MP4 and GIF inference. The README screenshots come from that same live API without request interception. The video fixture is an AI-generated still encoded as a short clip; it verifies sampled-frame analysis and does not demonstrate motion detection. Independent correctness reviews found no remaining issues after the example-button accessible names were fixed.

[Verification details and Lighthouse measurements](verification.json) include the local machine-load limits.
