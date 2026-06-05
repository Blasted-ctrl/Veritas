# Veritas — Deepfake & AI-Content Detector

Veritas accepts an **image, video, or audio** upload and returns an authenticity
verdict with a confidence breakdown and an interpretability heatmap. It
fine-tunes a Vision Transformer (images) and Wav2Vec2 (audio), exports both to
ONNX, and serves them through a FastAPI inference API with Redis caching and
Celery-based video frame fan-out.

> **Build status:** All six phases complete — identity-safe data prep, fine-tuned
> ViT + Wav2Vec2 detectors, an ONNX inference API (Redis cache + Celery video
> fan-out), and a Next.js upload UI with a Grad-CAM interpretability overlay;
> four-job CI green. The one outstanding item is the **live demo URL**, which
> needs deploying under your own Vercel/Render account (config included).

---

## Architecture

```mermaid
flowchart LR
    subgraph Client
        UI[Next.js + TypeScript<br/>drag-drop upload UI]
    end

    subgraph API["FastAPI inference service"]
        V[/POST /verify/]
        H[content-hash cache]
    end

    subgraph Workers
        C[Celery worker<br/>video -> frames -> aggregate]
    end

    subgraph Models["ONNX Runtime"]
        IMG[ViT image detector]
        AUD[Wav2Vec2 audio detector]
    end

    R[(Redis<br/>cache + broker)]

    UI -->|image / audio / video| V
    V --> H
    H <--> R
    V -->|image| IMG
    V -->|audio| AUD
    V -->|video| C
    C -->|frame batches| IMG
    C --> R
    IMG -->|verdict + Grad-CAM| V
    AUD -->|verdict| V
    V -->|verdict + confidence + heatmap| UI
```

### Data preparation pipeline (Phase 1)

```mermaid
flowchart LR
    DS[Dataset adapter<br/>FF++ / DFDC / ASVspoof / dir / synthetic] --> M[Manifest<br/>path, label, subject_id]
    M --> B[Class balancing]
    B --> S[Identity-safe split<br/>group by subject_id]
    S --> G{{Leakage gate<br/>assert no subject crosses splits}}
    G --> O[train.csv / val.csv / test.csv + summary.json]
```

---

## Why identity-safe splitting matters

In deepfake datasets, the *same person* appears in both authentic and
manipulated clips. If a subject's real frames land in **train** and their faked
frames land in **test**, a model can score well by memorising identity instead
of learning manipulation artefacts — inflating test metrics and collapsing in
the real world.

Veritas prevents this **structurally**: every sample carries a `subject_id`,
samples are grouped by it, and each group is assigned *in its entirety* to one
split. A deterministic greedy packer hits the requested split proportions while
keeping each split class-balanced. The pipeline re-verifies leak-freedom before
writing anything, and a test (`tests/test_splitting.py`) asserts the guarantee.

The dataset adapters encode the per-dataset notion of identity:

| Dataset          | Modality | `subject_id` derivation                                  |
|------------------|----------|----------------------------------------------------------|
| FaceForensics++  | image    | target identity of the clip (`033` from `033_097`)       |
| DFDC             | image    | the `original` source video a fake was derived from      |
| ASVspoof         | audio    | speaker id from the protocol file                        |
| generic `directory` | both  | filename stem up to the last `_` (e.g. `subject_0007`)   |

---

## Quickstart (Phase 1)

Requires Python 3.10+. The data-prep core is **standard-library only** — no
heavy ML wheels needed to run it or its tests.

```bash
cd backend
python -m venv .venv && . .venv/Scripts/activate   # Windows
# python -m venv .venv && source .venv/bin/activate  # macOS/Linux
pip install -e ".[dev]"

# Build balanced, identity-safe splits from generated fixtures (no download):
veritas prepare-data --modality image --output-dir ../data/processed/image
veritas prepare-data --modality audio --output-dir ../data/processed/audio

# Run the test suite (includes the no-identity-leakage assertion):
pytest
```

### Using real datasets

Download the gated datasets yourself (each requires accepting a licence), then
point an adapter at the extracted directory:

```bash
veritas prepare-data --modality image --source faceforensics --input-dir /data/FaceForensics++
veritas prepare-data --modality image --source dfdc          --input-dir /data/dfdc
veritas prepare-data --modality audio --source asvspoof      --input-dir /data/ASVspoof2019/LA
# Or a generic real/ + fake/ folder layout:
veritas prepare-data --modality image --source directory     --input-dir /data/my_images
```

Output per run: `train.csv`, `val.csv`, `test.csv` and a `summary.json`
recording split sizes, class balance, subject counts and `identity_leakage:
false`.

---

## Image detector (Phase 2): fine-tuned ViT

`veritas train-image` fine-tunes a pretrained Vision Transformer
(`google/vit-base-patch16-224`) for real-vs-manipulated classification:

* **Deterministic** — all RNGs seeded; reproducible given a seed + split.
* **Staged unfreezing** — stage 1 trains only the classifier head over the
  frozen backbone; stage 2 unfreezes the top encoder layers for end-to-end
  fine-tuning, with discriminative learning rates (higher for the head).
* **Linear warmup + decay** LR schedule.
* **Honest evaluation** — the model is selected on the validation split and
  reported on the untouched **test** split, written to `metrics.json`.

```bash
# Fine-tune on prepared splits (auto-detects GPU; falls back to CPU):
veritas train-image \
  --data-dir ../data/processed/image \
  --output-dir ../data/models/image \
  --epochs 5 --batch-size 16

# Fast, network-free smoke run (small randomly-initialised ViT):
veritas train-image --no-pretrained --image-size 32 --epochs 2 --limit 64
```

### Measured metrics

Source of truth: [`backend/artifacts/image/metrics.json`](backend/artifacts/image/metrics.json),
produced by `veritas train-image` (ViT-base, 3 epochs, seed 1337) on a held-out,
identity-safe test split.

| Metric | Value |
|--------|-------|
| Accuracy | 1.000 |
| Precision | 1.000 |
| Recall | 1.000 |
| F1 | 1.000 |
| AUC | 1.000 |
| Test samples (held-out) | 36 (18 real / 18 fake) |

The two-stage freeze→unfreeze strategy is visible in the per-epoch trainable
parameter count (from `metrics.json` → `meta.history`):

| Epoch | Stage | Trainable params | Val loss trend |
|-------|-------|------------------|----------------|
| 0 | head only (backbone frozen) | 1,538 | loss 0.394 |
| 1 | top-4 encoder blocks unfrozen | 28,354,562 | loss 0.0057 |
| 2 | top-4 encoder blocks unfrozen | 28,354,562 | loss 0.0001 |

> These numbers are **measured** by an actual evaluation run (the source of
> truth is `metrics.json`), not hand-picked. The figures below are from the
> synthetic fixture dataset used for the reproducible CPU smoke benchmark — they
> validate the pipeline end-to-end. To produce real benchmark numbers, run the
> same command against FaceForensics++/DFDC splits on a GPU; `metrics.json` is
> regenerated unchanged.

---

## Audio detector (Phase 3): fine-tuned Wav2Vec2

`veritas train-audio` fine-tunes a pretrained Wav2Vec2 (`facebook/wav2vec2-base`)
for bona-fide-vs-synthetic voice classification:

* **Preprocessing** — audio is loaded, mixed to mono, resampled to 16 kHz
  (`torchaudio`) and per-utterance normalized.
* **Staged unfreezing** — the convolutional feature encoder stays frozen
  throughout (standard Wav2Vec2 practice); stage 1 trains the projector +
  classifier head, stage 2 unfreezes the top transformer blocks.
* **Linear warmup + decay**, validation-based selection, honest **test**-split
  evaluation → `metrics.json`.

```bash
veritas train-audio \
  --data-dir ../data/processed/audio \
  --output-dir ../data/models/audio \
  --epochs 5 --batch-size 8

# Fast, network-free smoke run (small randomly-initialised Wav2Vec2):
veritas train-audio --no-pretrained --epochs 2 --max-seconds 1.0 --limit 64
```

### Measured metrics

Source of truth: [`backend/artifacts/audio/metrics.json`](backend/artifacts/audio/metrics.json),
produced by `veritas train-audio` (Wav2Vec2-base, 3 epochs, seed 1337) on a
held-out, identity-safe test split.

| Metric | Value |
|--------|-------|
| Accuracy | 0.722 |
| Precision | 0.722 |
| Recall | 0.722 |
| F1 | 0.722 |
| AUC | 0.750 |
| Test samples (held-out) | 36 (18 real / 18 fake) |
| Confusion matrix | TP 13 · FP 5 · TN 13 · FN 5 |

Staged unfreeze (CNN feature encoder stays frozen throughout), from
`metrics.json` → `meta.history`:

| Epoch | Stage | Trainable params | Val AUC |
|-------|-------|------------------|---------|
| 0 | head only (projector + classifier) | 197,378 | 0.901 |
| 1 | top-4 transformer blocks unfrozen | 33,269,890 | 0.892 |
| 2 | top-4 transformer blocks unfrozen | 33,269,890 | 0.886 |

> Measured on the synthetic-voice fixture test split (reproducible CPU
> benchmark). Point `--source asvspoof --input-dir ...` at ASVspoof on a GPU for
> real benchmark numbers; `metrics.json` regenerates unchanged.

---

## Inference API (Phase 4): ONNX serving

Both models are exported to ONNX and served via ONNX Runtime (no torch at serve
time — the API image is lean). A FastAPI service exposes `POST /verify`:

```bash
# 1. Export trained models to ONNX:
veritas export --modality image --model-dir ../data/models/image --output ../data/models/image/model.onnx
veritas export --modality audio --model-dir ../data/models/audio --output ../data/models/audio/model.onnx

# 2. Run the stack (Redis + API + Celery worker):
docker compose up        # API on http://localhost:8000

# 3. Verify a file:
curl -F file=@suspicious.jpg http://localhost:8000/verify
```

Response (`Verdict`):

```json
{
  "verdict": "fake",
  "confidence": 0.97,
  "fake_probability": 0.97,
  "modality": "image",
  "model": "model.onnx",
  "latency_ms": 12.4,
  "cached": false,
  "content_sha256": "…"
}
```

* **Routing** — `image` → ONNX ViT, `audio` → ONNX Wav2Vec2, `video` → a Celery
  task that samples frames (`VERITAS_VIDEO_FPS`), scores each with the image
  detector, and aggregates per-frame probabilities into one verdict (with a
  `frames` breakdown).
* **Caching** — results are keyed by SHA-256 of the upload in Redis; an
  unreachable Redis transparently degrades to an in-memory cache.
* **Latency** — measured server-side, returned as `latency_ms` and logged.
* **Graceful degradation** — missing models return `503`; `GET /health` reports
  per-model availability and the active cache backend.

```mermaid
sequenceDiagram
    participant C as Client
    participant A as FastAPI /verify
    participant R as Redis cache
    participant W as Celery worker
    participant O as ONNX Runtime
    C->>A: upload (image/audio/video)
    A->>R: get(sha256)
    alt cache hit
        R-->>A: cached verdict
    else miss
        alt video
            A->>W: analyze_video(frames)
            W->>O: score each frame
            O-->>W: per-frame probs
            W-->>A: aggregated verdict
        else image / audio
            A->>O: run ONNX model
            O-->>A: logits → probability
        end
        A->>R: set(sha256, verdict)
    end
    A-->>C: verdict + confidence + latency
```

---

## Frontend (Phase 5): upload UI + Grad-CAM overlay

A Next.js + TypeScript single-page app (`frontend/`):

* **Drag-and-drop** (or click) upload for image / audio / video, with an upload
  progress bar.
* **Verdict display** — real/manipulated with animated **confidence bars** and
  the fake-probability, model, measured latency and cache status.
* **Grad-CAM overlay** — for images, toggle between the original and the heatmap
  showing which regions drove the verdict.
* **Video** — a per-frame fake-probability bar chart from the Celery aggregation.
* A live **health banner** reflects which models/heatmaps the API has loaded.

Run the whole stack locally:

```bash
docker compose up          # redis + api + worker + frontend
# UI:  http://localhost:3000      API: http://localhost:8000
```

Or develop the frontend against a running API:

```bash
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

**Deployment** — `frontend/` deploys to Vercel (`vercel.json`); the full stack
(Redis + API + worker + frontend) deploys to Render via the `render.yaml`
blueprint. Set `NEXT_PUBLIC_API_URL` to the deployed API URL.

> **Live demo:** _to be added_ — deploy via Vercel/Render (configs included) and
> drop the URL here.

---

## Repository layout

```
veritas/
├── .github/workflows/ci.yml      # backend (lint+type+test), ml, api, frontend jobs
├── docker-compose.yml            # redis + api + worker + frontend
├── render.yaml                   # Render blueprint (full-stack deploy)
├── .env.example
├── LICENSE
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml            # core = stdlib; ml/api/dev = optional extras
│   ├── src/veritas/
│   │   ├── cli.py                # `veritas` CLI (argparse)
│   │   ├── config.py
│   │   ├── data/                 # adapters, manifest, identity-safe splitting
│   │   ├── models/               # ViT + Wav2Vec2
│   │   ├── training/             # fine-tuning + metrics
│   │   ├── export/               # ONNX export
│   │   ├── explain/              # Grad-CAM heatmaps
│   │   ├── api/                  # FastAPI service + ONNX inference + cache
│   │   └── tasks/                # Celery video frame batching
│   ├── artifacts/                # committed metrics.json (source of truth)
│   └── tests/
└── frontend/
    ├── Dockerfile                # standalone Next.js image
    ├── vercel.json
    └── src/{app,components,lib}   # upload UI, verdict card, Grad-CAM overlay
```

---

## Roadmap

- [x] **Phase 1** — Scaffold + identity-safe dataset preparation.
- [x] **Phase 2** — Fine-tune ViT image detector → `metrics.json`.
- [x] **Phase 3** — Fine-tune Wav2Vec2 audio detector → `metrics.json`.
- [x] **Phase 4** — ONNX export + FastAPI `/verify` with Redis cache & Celery video fan-out.
- [x] **Phase 5** — Next.js upload UI + Grad-CAM overlay; deploy config (Vercel + Render).
- [x] **Phase 6** — Hardening: expanded tests, multi-job CI, complete docs.

All reported metrics come from `metrics.json` produced by real evaluation runs
on held-out test sets — never hardcoded.

## Testing & CI

CI runs four jobs (see [.github/workflows/ci.yml](.github/workflows/ci.yml)):

| Job | What it checks |
|-----|----------------|
| `backend` | ruff lint + format, mypy (stdlib core), stdlib-only tests |
| `ml-tests` | CPU torch — tiny ViT/Wav2Vec2 training, metrics, Grad-CAM (`-m ml`) |
| `api-tests` | ONNX export round-trips + FastAPI `/verify` image/audio/video (`-m api`) |
| `frontend` | `next lint`, `tsc --noEmit`, `next build` |

The backend suite is **55 tests**: the stdlib-only core (identity-safe
splitting, manifests, cache) runs everywhere; the `ml`/`api` markers gate tests
that need torch/onnxruntime. Tiny randomly-initialised models keep the ML/API
jobs fast and network-free.

```bash
cd backend
ruff check src tests && ruff format --check src tests   # lint + format
mypy                                                    # type-check (stdlib core)
pytest                       # all tests (needs .[ml,api,dev])
pytest -m "not ml and not api"   # stdlib-only subset (no heavy deps)
```

## Status, honesty & limitations

This repo is built to be **real and runnable**, with a few honest caveats:

- **Metrics are measured, never invented.** The committed `metrics.json` files
  are from actual evaluation runs. The headline figures are on the *synthetic
  fixture* datasets (a reproducible CPU benchmark), not on FaceForensics++ /
  DFDC / ASVspoof — those are gated, hundreds of GB, and need a GPU. The
  training/eval code regenerates `metrics.json` unchanged when pointed at the
  real datasets (`--source faceforensics|dfdc|asvspoof`).
- **Live demo URL & screenshots** require deploying with your own Vercel/Render
  accounts. The deploy config (`vercel.json`, `render.yaml`, Dockerfiles) is
  included; once deployed, drop the URL into the [Frontend](#frontend-phase-5-upload-ui--grad-cam-overlay)
  section and capture screenshots of a verdict + heatmap.
- **Grad-CAM** needs gradients, so it uses the torch ViT (not ONNX); enable it by
  setting `VERITAS_IMAGE_TORCH_DIR` and installing the `ml` extra. The ONNX
  verdict path stays torch-free.

## License

[MIT](LICENSE)
