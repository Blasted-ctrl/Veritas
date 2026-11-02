<div align="center">

<img src="frontend/src/app/icon.svg" alt="" width="48" height="48" />

# Veritas

Check images and video frames for signs of AI generation.

[Demo video](https://github.com/Blasted-ctrl/Veritas/raw/refs/heads/main/docs/demo/veritas-demo.mp4) · [Run locally](#run-locally) · [Training results](docs/evaluation/artwork-v3/README.md)

</div>

## Why we built it

Veritas makes an AI detector usable through a simple website. Upload a file,
preview it, and see the model's prediction, score, and decision cutoff.
For videos, you can inspect the scores of individual sampled frames.

Human-made anime, paintings, and illustrations belong in the non-AI class.
We train on both human-made and generated images to reduce false alarms,
and keep separate test data to measure what actually improves.

## Stack

| Part               | What we used                                                                  |
| ------------------ | ----------------------------------------------------------------------------- |
| Model training     | **PyTorch** and **TorchVision**                                               |
| Image classifier   | **CNN**, specifically **EfficientNet-B0** for the artwork model               |
| Learning algorithm | **Gradient descent using AdamW**, with backpropagation and cross-entropy loss |
| Website            | **TypeScript**, **React**, **Next.js**, CSS and Tailwind CSS                  |
| Backend            | **Python**, **FastAPI** and Uvicorn                                           |
| Model inference    | **ONNX Runtime** on CPU                                                       |
| Video processing   | **FFmpeg** for extracting frames                                              |
| Cache              | **Redis**, with an in-memory fallback                                         |
| Local setup        | **Docker Compose**                                                            |

## Demo

[![Veritas demo preview](docs/demo/preview.gif)](https://github.com/Blasted-ctrl/Veritas/raw/refs/heads/main/docs/demo/veritas-demo.mp4)

Click the preview to download the 45-second demo. It shows real image and
video analysis. The recording predates the latest visual polish; these
screenshots show the current light-only website.

![Veritas upload page](docs/screenshots/site-overview.jpg)

<details>
<summary>Image and video results</summary>

![Image analysis](docs/screenshots/image-analysis.jpg)

![Video analysis and sampled-frame scores](docs/screenshots/video-analysis.jpg)

</details>

[Demo details and credits](docs/demo/README.md)

## Run locally

Install Docker with Compose, then run:

```sh
git clone https://github.com/Blasted-ctrl/Veritas.git
cd Veritas
docker compose build
docker compose run --rm --no-deps api veritas download-model
docker compose up
```

Open **[localhost:3000](http://localhost:3000)**. Choose an image or video,
then select **Analyze file**. No API key is needed.

The quickstart downloads a separate pretrained **CIFAKE ViT**, not the
research CNN shown in the demo. The trained CNN weights stay outside Git.
See [native setup and model configuration](docs/development.md) to use your
own checkpoint.

## Understand the result

A score is not proof or a calibrated probability. The detector can miss AI
images and incorrectly flag human-made media. Videos are checked through
sampled still frames; motion and audio are not analyzed by the website.

The artwork CNN reached **80.1% accuracy** on a separate 512-image test at
the 0.5 cutoff. The demo uses a more conservative 87.7% cutoff, which scored
69.7% accuracy and detected 46.1% of AI images on that test. **The 95% goal
has not been reached.** [Full results and limits](docs/evaluation/artwork-v3/README.md).

## More information

- [Development, API and model setup](docs/development.md)
- [Train and export a CNN](docs/cnn-training.md)
- [Default model and limitations](docs/model.md)

## License

The code is [MIT](LICENSE). The downloadable default model is Apache-2.0.
The research CNN inherits research/noncommercial training-data restrictions.
[Sample media credits](frontend/public/samples/credits.json) and the
[font license](frontend/public/fonts/OFL.txt) are included.
