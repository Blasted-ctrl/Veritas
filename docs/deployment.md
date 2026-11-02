# Deploy Veritas online

Veritas runs as a Next.js website and a Python inference API. Inference works on a CPU. Video requests run synchronously, so start with a small audience and measure latency, memory, and concurrency before increasing traffic.

- [Render: dashboard setup](#option-1-render)
- [Linux server: Docker and automatic HTTPS](#option-2-linux-server)
- [Install the artwork CNN](#install-the-artwork-cnn)
- [API reference](#api-reference)

## Choose the model

| Model | Availability | Limits and permissions |
| --- | --- | --- |
| Default CIFAKE-trained ViT | Downloaded by the Render build or the Docker CLI | Apache-2.0 weights. Can misclassify real photos and human-made art. This is not the demo CNN. |
| Artwork CNN | Separate `veritas-artwork-v3.zip` from the training machine; weights are outside Git | The model in the demo. Inherits research/noncommercial training-data restrictions. Its conservative cutoff misses many AI images. |
| Your own model | [Train and export with PyTorch](cnn-training.md) | Check data rights and evaluate on separate test data before use. |

The CNN's conservative cutoff is **0.8770313858985902**. At this cutoff, overall test accuracy was 69.7%, with 46.1% AI recall. Its 80.1% accuracy result used a different cutoff of 0.5. [Full evaluation](evaluation/artwork-v3/README.md).

For commercial hosting, review the default model's license and model card, or train on data you have permission to use. The project's MIT code license does not remove model or dataset restrictions.

## Option 1: Render

You need a Render account connected to the repository and resources sufficient to run the model. This guide does not promise a free instance can run inference reliably. Render manages public URLs and TLS; see its [Docker documentation](https://render.com/docs/docker).

### 1. Deploy the API

Create a **Web Service**, connect `Blasted-ctrl/Veritas`, select `main`, and choose Docker. Leave the repository root directory unset and configure:

| Setting | Value |
| --- | --- |
| Dockerfile path | `./backend/Dockerfile` |
| Docker build context | `./backend` |
| Health check path | `/health` |
| `DOWNLOAD_PRETRAINED_MODEL` | `1` |
| `PORT` | `8000` |
| `VERITAS_REDIS_URL` | Empty string |
| `VERITAS_ALLOWED_ORIGINS` | `["https://YOUR-WEBSITE.onrender.com"]` |

Use the intended website URL for the origin placeholder and update it after creating the website if needed. Origin entries have **no trailing slash**. The build downloads and verifies the pinned default ViT. A disk is unnecessary for these baked-in weights; an empty disk mounted over `/app/data` would hide them.

Deploy, then open `https://YOUR-API.onrender.com/health`. Confirm **`image_model: true`**. HTTP 200 alone does not establish readiness. The API binds to `0.0.0.0:8000`; the explicit `PORT` setting keeps Render aligned with it.

### 2. Deploy the website

Create a second Docker Web Service from the same repository and branch:

| Setting | Value |
| --- | --- |
| Dockerfile path | `./frontend/Dockerfile` |
| Docker build context | `./frontend` |
| `PORT` | `3000` |
| `NEXT_PUBLIC_API_URL` | `https://YOUR-API.onrender.com` |

Set the API URL **before building**, without a trailing slash. Render [passes environment variables to Docker build arguments](https://render.com/docs/docker#environment-variable-translation); the Dockerfile consumes that argument. Changing this URL requires **rebuild and deploy**, not just a restart. A visitor's browser cannot reach your Docker service name `api` or your server's `localhost`.

### 3. Connect and verify

Set the API's `VERITAS_ALLOWED_ORIGINS` to a JSON array containing the website's final HTTPS origin. Redeploy the API. Open the website and confirm **Detector ready**. Upload a known human-made image, a labeled AI image, and a short MP4. Inspect frame scores and model details. A wrong prediction is a model error; a failed request is a deployment error.

If you add custom domains, update allowed origins and rebuild the frontend whenever its API URL changes. [render.yaml](../render.yaml) provides the same services as a Blueprint template.

This route uses the default ViT. To reproduce the CNN demo, use the server instructions below or package authorized weights in a private container image. Check the research checkpoint's data restrictions before distributing it.

## Option 2: Linux server

[deploy/compose.yaml](../deploy/compose.yaml) starts the website, API, and Caddy. Only ports 80 and 443 are published. Caddy stores certificates in a named volume and manages [automatic HTTPS](https://caddyserver.com/docs/automatic-https).

### 1. Prepare the server and DNS

Install Docker Engine and the Compose plugin on a Linux server. As an initial, unbenchmarked estimate, allow 2 CPU cores, 4 GB RAM, and disk space for image builds and model artifacts. Increase resources if builds or inference exceed this budget. No hosting account or server has been provisioned for you.

Point two DNS A records to the server's public IPv4 address:

| Host | Example |
| --- | --- |
| Website | `veritas.example.com` |
| API | `api.veritas.example.com` |

Allow inbound TCP 80 and 443 at the provider firewall. UDP 443 is optional for HTTP/3. Leave 3000 and 8000 closed externally. Add IPv6 records only if the server is configured to serve them.

### 2. Clone and configure

Run on the server:

```sh
git clone https://github.com/Blasted-ctrl/Veritas.git
cd Veritas
cp deploy/.env.example deploy/.env
nano deploy/.env
```

Set `VERITAS_DOMAIN` to the website hostname, such as `veritas.example.com`, without a scheme, path, or trailing slash. Set `ACME_EMAIL` to your contact email. The API hostname becomes `api.veritas.example.com`. Leave the default model path and threshold for the first run, or install the CNN below before starting.

Production reads **deploy/.env**, independently of the root development .env file. Keep real environment files out of Git.

### 3. Build, install, and start

From the repository root:

```sh
docker compose --env-file deploy/.env -f deploy/compose.yaml config --quiet
docker compose --env-file deploy/.env -f deploy/compose.yaml build
docker compose --env-file deploy/.env -f deploy/compose.yaml run --rm --no-deps api veritas download-model
docker compose --env-file deploy/.env -f deploy/compose.yaml up -d
```

The default download is about 328 MiB and is verified by checksum. Models persist in a named volume. Caddy obtains certificates after DNS resolves and ports 80/443 are reachable.

### 4. Verify public access

Replace the example domain:

```sh
curl --fail https://api.veritas.example.com/health
curl --fail -F "file=@frontend/public/samples/earthrise.jpg" https://api.veritas.example.com/verify
docker compose --env-file deploy/.env -f deploy/compose.yaml ps
docker compose --env-file deploy/.env -f deploy/compose.yaml logs --tail=100 api proxy
```

Confirm `image_model: true`, inspect `model_info` in the analysis response, then test uploads in the public website. The API container health check verifies model readiness before the frontend starts.

This deployment has no account system or built-in rate limiter. For open public access, apply request and concurrency limits at your hosting edge. Watch CPU, memory, errors, and latency: long video requests can occupy a worker. The 25 MiB file limit and 32-frame cap bound individual requests, not total demand.

## Install the artwork CNN

Upload the separate `veritas-artwork-v3.zip` securely to the server. It is not downloaded by `veritas download-model`. After building the API image, run from the repository root:

```sh
sha256sum veritas-artwork-v3.zip
unzip veritas-artwork-v3.zip -d model-bundle
sha256sum model-bundle/model/model.onnx model-bundle/model/model.onnx.json
```

Expected SHA-256 values:

| File | SHA-256 |
| --- | --- |
| ZIP | `1f8be01817d901be6cd289c33b5d12e2207c69c86443acf7a5cfe820986cc18a` |
| `model.onnx` | `9bbf46eed19ff157779bd56fee3d98eaf488f9ccfcb039b260ded569c9701a4f` |
| `model.onnx.json` | `68222d23989e1f154b68a341987c1d5b484e3bf10309f7b1e42305a3898c5021` |

Stop if a checksum differs. Read `USE.txt` and the report in the archive. Copy **both** the graph and its metadata/preprocessing sidecar into the persistent volume:

```sh
docker compose --env-file deploy/.env -f deploy/compose.yaml run --rm --no-deps \
  -v "$PWD/model-bundle/model:/bundle:ro" api sh -c \
  'mkdir -p /app/data/models/artwork && cp /bundle/model.onnx /bundle/model.onnx.json /app/data/models/artwork/'
```

Set these values in `deploy/.env`:

```dotenv
VERITAS_IMAGE_MODEL=/app/data/models/artwork/model.onnx
VERITAS_FAKE_THRESHOLD=0.8770313858985902
```

Apply the changes:

```sh
docker compose --env-file deploy/.env -f deploy/compose.yaml up -d
```

Verify a fresh analysis returns the graph checksum above and the expected `decision_threshold`. The website shows the cutoff. It reduces false alarms while missing many AI images; the raw class score stays unchanged. To return to the default model, restore its original path and `0.5` threshold in deploy/.env, install the default if needed, and run `up -d` again.

## Updates and rollback

Record the old commit before updating:

```sh
git rev-parse HEAD
git pull --ff-only
docker compose --env-file deploy/.env -f deploy/compose.yaml build
docker compose --env-file deploy/.env -f deploy/compose.yaml up -d
```

Repeat health and browser checks after each deployment. If it fails, check out the recorded previous commit, rebuild, and run `up -d`. Keep the graph, sidecar, checksums, threshold, and environment file together for restoring a matching configuration. Avoid `down -v`: it deletes the stored models and certificates.

On Render, use rollback/redeploy controls and restore matching environment settings. Verify readiness and an upload afterward.

## API reference

`GET /health` reports ONNX readiness, optional explainer compatibility, cache backend, and upload/sampling limits. A running service may report `image_model: false`.

`POST /verify` accepts multipart field `file`. Responses include `verdict`, `fake_probability` (the historical name for the class score), `decision_threshold`, `modality`, `model_info`, `content_sha256`, `latency_ms`, and `cached`. Videos add `frames_analyzed` and frame scores. Verdicts compare the score with the cutoff. `explain=true` adds a heatmap only with a matching Torch ViT checkpoint.

```sh
curl -F "file=@image.jpg" http://localhost:8000/verify
```

Empty uploads return 400, oversized uploads 413, unsupported media 415, unreadable media 422, and unavailable models 503. Failed video analysis is not cached as a real verdict.

| Variable | Meaning |
| --- | --- |
| `VERITAS_IMAGE_MODEL` | ONNX path; keep its `.json` sidecar alongside it |
| `VERITAS_FAKE_THRESHOLD` | Decision cutoff; default 0.5 |
| `VERITAS_ALLOWED_ORIGINS` | JSON array of browser origins, without trailing slashes |
| `VERITAS_MAX_UPLOAD_BYTES` | Per-file limit; default 26214400 bytes |
| `VERITAS_VIDEO_FPS` | Target sampled frames per second; default 1 |
| `VERITAS_MAX_VIDEO_FRAMES` | Maximum frames analyzed; default 32 |
| `VERITAS_REDIS_URL` | Optional Redis connection; empty selects memory caching |
| `VERITAS_AUDIO_MODEL` | Optional custom audio model; not used in the website's video analysis |
| `NEXT_PUBLIC_API_URL` | Browser-reachable API URL, applied at frontend build time |

## Data and hosting

The API does not retain uploaded media after processing. The web framework may spool uploads temporarily; video processing creates a temporary file removed after analysis. Scores, hashes, provenance, and optional explanation images can be cached. Memory caching is bounded and expires; optional Redis entries use a TTL. Logs record modality, verdict, score, and latency. Hosting providers and proxies may have their own retention policies.

Browser cancellation stops the client waiting and transferring data; already-running inference can finish. The website analyzes sampled still frames, not motion or audio. Scores do not establish authenticity.

Production-container CI builds both images, installs the default model as the non-root user, verifies image and MP4 inference, and checks the API URL in the browser bundle. CI also validates production Compose and Caddy configuration. Docker is unavailable on the development machine; public DNS and automatic TLS have not been exercised here.
