# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

People who want to inspect an image or video and understand whether a trained detector finds signs associated with generated media. Developers can also train and evaluate their own detectors.

## Product Purpose

Veritas makes media-classifier results accessible through a file upload. A useful result shows the media, the model's prediction and the evidence available, with enough context to understand its limits.

## Positioning

The repository combines identity-separated dataset preparation, custom-model training and ONNX inference with a browser interface. Video analysis samples frames and reuses the image detector.

## Operating Context

A visitor chooses or drops a local file, waits for analysis and reviews the prediction. The operator downloads a pretrained checkpoint explicitly or supplies a custom model. Inference runs on the configured API server.

## Capabilities and Constraints

Image, video/GIF and optional audio inference exist. Model weights are downloaded separately or supplied by the operator. Historical committed accuracy figures are synthetic-fixture smoke results, not validation on modern generated media. Video frame scores do not verify audio or temporal manipulation. Heatmaps require a matching Torch checkpoint and remain optional. The user authorizes additions and changes throughout the repository; existing implementation limits are adjustable.

## Brand Commitments

Keep the Veritas name. The website should be visually pleasing and understandable. Documentation should explain why the project was built in simple language and include real screenshots of working analysis.

## Evidence on Hand

Source inspection, synthetic-fixture test data, a checksum-pinned pretrained detector, successful Torch/ONNX numerical parity checks, real browser analysis captures, and passing CI and production-container checks. The NASA photograph example demonstrates a false positive. No independent real-world accuracy benchmark, testimonial, or public deployment is claimed.

## Product Principles

- Show the actual model output and identify the model that produced it.
- Explain what the detector checked in language a visitor can understand.
- Treat unavailable or failed analysis as an error, never authenticity evidence.
- Keep file selection and results in a short, accessible flow.
