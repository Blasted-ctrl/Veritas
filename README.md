# Veritas

A place to take a second look at images and videos that might be AI-generated.

Veritas makes a trained detector easy to try: upload a file, keep the original in view, and inspect the prediction and model details. It was built to make detection understandable and reproducible, including the times a model gets it wrong. The original data preparation, image/audio training, and ONNX export tools are still here.

## See it working

These are real screenshots from the local website using the downloadable detector, not mock results.

![Generated image analyzed in Veritas](docs/screenshots/image-analysis.jpg)

The labeled AI-generated landscape scored about 98.8% for the AI-generated class.

![Video analysis with individual sampled-frame scores](docs/screenshots/video-analysis.jpg)

A four-second MP4 made from that image was sampled into four frames, with an average score around 98.7%. [Sample files and credits](frontend/public/samples/credits.json) are included so you can try the same inputs.

## Run it locally

Install Docker with Compose, then run:

    git clone https://github.com/Blasted-ctrl/Veritas.git
    cd Veritas
    docker compose build
    docker compose run --rm --no-deps api veritas download-model
    docker compose up

Open [localhost:3000](http://localhost:3000). Choose an image or video, check the preview, then select **Analyze file**. The first model download is about 328 MiB. Weights stay in a Docker volume and are verified by checksum. No API key is needed.

Prefer Python and Node directly? See [native setup and development](docs/development.md).

## Read the result carefully

The percentage is a model class score, not proof or a calibrated probability of authenticity. The default CIFAKE-trained model can miss new generators and misclassify genuine photographs. In our demo, NASA’s real Earthrise photo incorrectly scored about **89.5% AI-generated**. Check the source even when a score is high.

Videos are assessed through sampled image frames, up to 32 at one frame per second by default. Audio and motion are not analyzed. Optional audio training and inference require your own audio model.

See [the detector, its source, and its limits](docs/model.md) for the pinned revision, license, preprocessing, and custom-model support.

## What’s inside

- **Website:** responsive Next.js interface, previews, light/dark themes, upload progress, cancellation, frame scores, and model attribution.
- **API:** FastAPI and CPU ONNX Runtime. Temporary video uploads are removed after analysis. Scores are cached with model and processing settings included in the key; Redis is optional.
- **Training tools:** identity-separated dataset preparation, ViT image and Wav2Vec2 audio training, evaluation, ONNX export, and optional matching-checkpoint Grad-CAM.

[Development and verification](docs/development.md) · [API and deployment](docs/deployment.md) · [Design language](design/veritas-design/SKILL.md)

MIT for the project code. The default pretrained model uses Apache-2.0; sample and font credits are included alongside their files.
