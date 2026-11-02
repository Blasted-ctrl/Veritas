# Veritas

Check images and video frames for signs of AI generation.

Veritas was built to make AI detection easy to use and its results easier to question. Choose an image or video, preview it, then run the analysis. You get a prediction and an AI score, with the model details available when you need them.

## See it working

The website has image and video modes, drag-and-drop uploads, and examples you can try. These screenshots show real responses from the locally trained artwork CNN. The Docker quickstart below downloads a separate default model; [the CNN results](docs/evaluation/artwork-v3/README.md) explain the demo checkpoint and its limits.

![Veritas image upload page](docs/screenshots/site-overview.jpg)

![Generated image analyzed in Veritas](docs/screenshots/image-analysis.jpg)

The labeled AI-generated landscape scored about 99.2% for the AI-generated class.

![Video analysis with individual sampled-frame scores](docs/screenshots/video-analysis.jpg)

A four-second MP4 made from that image was sampled into four frames, with an average score around 99.0%. [Sample files and credits](frontend/public/samples/credits.json) are included so you can try the same inputs.

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

The percentage is a model class score, not proof or a calibrated probability of authenticity. The default CIFAKE-trained model can miss new generators and misclassify genuine photographs. In an earlier check with that default model, NASA’s real Earthrise photo incorrectly scored about **89.5% AI-generated**. Human-made anime, drawings and digital paintings are non-AI too; this photo-focused detector can falsely flag them. Check the source even when a score is high.

Videos are assessed through sampled image frames, up to 32 at one frame per second by default. Audio and motion are not analyzed. Optional audio training and inference require your own audio model.

See [the detector, its source, and its limits](docs/model.md) for the pinned revision, license, preprocessing, and custom-model support.

## What’s inside

- **Website:** responsive Next.js interface, previews, light/dark themes, upload progress, cancellation, frame scores, and model attribution.
- **API:** FastAPI and CPU ONNX Runtime. Temporary video uploads are removed after analysis. Scores are cached with model and processing settings included in the key; Redis is optional.
- **Training tools:** group-separated dataset preparation, PyTorch EfficientNet-B0/ConvNeXt CNN and ViT image training, Wav2Vec2 audio training, held-out evaluation, ONNX export, and optional matching-ViT Grad-CAM.

The artwork CNN experiment adds human illustrations and paintings alongside AI artwork. On 256 held-out artwork images at the same 0.5 cutoff, human-art false alarms fell from 68.0% with the website detector to 10.9%, while AI-art recall was 87.5%. On the full 512-image test at 0.5, CNN accuracy was **80.1%; the 95% target was not reached**. A conservative cutoff reduces false alarms further but misses more AI images, and the photo regression cohort exposes remaining errors.

The trained research checkpoint stays outside Git. The first-run downloadable model remains the older ViT.

[Train the PyTorch CNN](docs/cnn-training.md) · [Artwork results and limits](docs/evaluation/artwork-v3/README.md) · [Earlier diverse-data experiment](docs/evaluation/diverse-v2/README.md)

[Development and verification](docs/development.md) · [API and deployment](docs/deployment.md) · [Product page design](docs/design/product-redesign.md)

MIT for the project code. The default pretrained model uses Apache-2.0; sample and font credits are included alongside their files.
