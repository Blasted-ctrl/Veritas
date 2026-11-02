# API and deployment

## Local containers

The README starts the website, API, and Redis. Models live in the veritas-data named volume, initialized with ownership for the non-root API user. The worker is optional under the worker profile; ordinary video requests run synchronously in the API process using a shared image-detector session.

The API image explicitly sets VERITAS_DATA_ROOT=/app/data. To include the default model in a standalone image, build the backend with DOWNLOAD_PRETRAINED_MODEL=1. The default Compose workflow downloads it into the persistent volume instead. Remove a model volume only when you intend to discard its stored artifacts.

## Configuration

Copy .env.example when you need overrides. Important settings:

| Variable | Meaning |
| --- | --- |
| VERITAS_IMAGE_MODEL | ONNX image model path, with optional .json sidecar |
| VERITAS_AUDIO_MODEL | Optional custom audio model path |
| VERITAS_ALLOWED_ORIGINS | JSON array of allowed frontend origins |
| VERITAS_MAX_UPLOAD_BYTES | Per-file limit, default 26214400 bytes |
| VERITAS_VIDEO_FPS | Target sampled frames per second, default 1 |
| VERITAS_MAX_VIDEO_FRAMES | Maximum sampled frames, default 32 |
| VERITAS_FAKE_THRESHOLD | Decision threshold, default 0.5 |
| VERITAS_REDIS_URL | Optional cache connection; empty selects memory |
| NEXT_PUBLIC_API_URL | Browser-reachable API URL, applied at frontend build time |

For Compose, NEXT_PUBLIC_API_URL is passed as a build argument. Rebuild the frontend after changing it. A Docker network name such as api is not normally reachable from a visitor’s browser. For another site origin, configure the API’s allowed origins as well.

The Render file is a deployment template for separate API and frontend services. Set NEXT_PUBLIC_API_URL to the public API address and VERITAS_ALLOWED_ORIGINS to a JSON array containing the public frontend origin. Its API build downloads the pinned model. Render [passes environment variables to Docker build arguments](https://render.com/docs/docker#environment-variable-translation), which the frontend Dockerfile consumes. Choose resources sufficient to load and run the model; the template is not a promise that a free instance can do so.

## Endpoints

GET /health reports whether the actual ONNX sessions load, optional explainer compatibility, the cache backend, and upload/sampling limits. A running service may report image_model=false; checking status alone does not establish detector readiness.

POST /verify accepts multipart form field file. It returns verdict, fake_probability, decision_threshold, modality, model_info, content_sha256, latency_ms, and cached. Videos also return frames_analyzed and frame scores. A request for explain=true only adds a heatmap when a matching Torch ViT checkpoint is configured.

Example:

    curl -F "file=@image.jpg" http://localhost:8000/verify

Empty uploads return 400, oversized uploads 413, unsupported media 415, unreadable media 422, and unavailable models 503. Failed video analysis is not cached as a real verdict.

## Data and hosting

The API does not retain uploaded media. Request uploads may be temporarily spooled by the web framework; video processing also creates a temporary file and removes it after analysis. Scores, file hashes, provenance, and optional explanation images can be cached. Memory caching is bounded and expires; Redis entries use the configured TTL. Logs record modality, verdict, score, and latency.

Browser cancellation stops the client waiting and transferring data; already-running server inference can finish. The service is designed for local use and controlled hosting. A public operator should set appropriate request limits, concurrency, HTTPS, and access controls at its hosting boundary.

The production-container CI job builds both images, installs the real model as the non-root runtime user, verifies image and MP4 inference, and checks that a nondefault API URL was included in the browser bundle. Docker was not installed on the development machine; this gate runs on GitHub’s Linux runner.
