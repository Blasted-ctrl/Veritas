# Development

Use Python 3.11 or newer and Node 24. The API can serve the downloaded image detector without Torch. Training and Grad-CAM use the optional ML dependencies.

## Native setup

From the repository root, create and activate a virtual environment:

    python -m venv .venv

On Windows PowerShell:

    .venv\Scripts\Activate.ps1

On macOS or Linux:

    source .venv/bin/activate

Install and start the API:

    cd backend
    python -m pip install -e ".[api,dev]"
    veritas download-model
    uvicorn veritas.api.app:app --host 127.0.0.1 --port 8000

Keep that terminal running. In another terminal:

    cd frontend
    npm ci
    npm run dev

Open http://localhost:3000. The default allowed origins include localhost and 127.0.0.1 on port 3000. Redis is optional; an unreachable server falls back to an expiring, bounded memory cache. Native model files default to data/models under the repository root when installed editable.

## Train your own models

Install the additional ML dependencies in the same environment:

    python -m pip install -e "backend[ml]"

Prepare local data, then train and export. Run these commands from backend:

    veritas prepare-data --modality image --source directory --input-dir /path/to/images
    veritas train-image --data-dir ../data/processed/image --output-dir ../data/models/cnn-image
    veritas export --modality image --model-dir ../data/models/cnn-image

The default image trainer is a PyTorch ConvNeXt-Tiny CNN; --architecture vit selects ViT. See [CNN training and evaluation](cnn-training.md) for data provenance, generator holdouts, GPU recipes and measured pilot limits.

The directory adapter expects real/ and fake/ folders. FaceForensics++, DFDC, and ASVspoof adapters also accept local datasets; obtain them under their own access terms. Subject identities stay within a single train, validation, or test split. Verify the identity labels in your own source data.

The default prepare-data source is synthetic fixtures for smoke testing. Its metrics do not measure real-world detection accuracy. Network-free training smoke commands:

    veritas prepare-data --modality image
    veritas train-image --no-pretrained --image-size 32 --epochs 2 --limit 64
    veritas prepare-data --modality audio
    veritas train-audio --no-pretrained --epochs 2 --max-seconds 1.0 --limit 64

Back up your models before exporting over them. A custom image export writes model.onnx.json with the matching normalization, graph hash, and Torch state hash. The optional ViT explainer requires that same saved Torch checkpoint. CNN heatmaps are unavailable. The API identifies the configured graph using its sidecar metadata.

## Checks

From backend:

    ruff check src tests scripts
    ruff format --check src tests scripts
    mypy
    pytest

Install all extras to run the ML and API tests. Without them, optional test modules skip at collection; the standard-library data pipeline remains usable. CI separates core, ML, API, frontend, and production-container checks.

From frontend:

    npm run lint
    npm run typecheck
    npm run build

The [recorded pretrained parity check](verification/pretrained-parity.json) compares serving tensors and raw ONNX logits against the upstream Torch checkpoint, including a non-square generated image and NASA’s photograph. It is a numerical conversion check, not a quality benchmark.

To reproduce it, download config.json, preprocessor_config.json, and model.safetensors from capcheck/ai-image-detection at revision a6661e07d38f1a097bba07ca9415538819278f09 into a reference directory, install the ML extras, then run:

    python scripts/verify_pretrained.py --model ../data/models/image/model.onnx --reference-dir /path/to/reference --images ../frontend/public/samples/generated-lakehouse.png ../frontend/public/samples/earthrise.jpg --output parity.json

## Screenshots

Start the real API with the default model and the website. Select Generated image, then Analyze file. For the video view, upload frontend/public/samples/generated-demo.mp4, analyze it, and expand individual frame scores. Capture the actual browser after analysis completes. The demo MP4 is a four-second still-image clip at four frames per second; it is not independently generated footage.

The checked-in screenshots are real browser captures. Each has an origin sidecar. Sample image prompts, NASA credits, and font licenses are saved alongside their assets.
