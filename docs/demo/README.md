# Veritas demo

[Download the 45-second MP4](https://github.com/Blasted-ctrl/Veritas/raw/refs/heads/main/docs/demo/veritas-demo.mp4)

The demo shows real uploads of NASA's Earthrise photograph, a generated
landscape, and a four-second video made from that still image. It predates
the latest logo and light-only polish; current screenshots are in the main
README. The video is silent, with instructions on screen.

Scores come from the research EfficientNet-B0 CNN at its conservative
cutoff of `0.8770313858985902`. [recording.json](recording.json) contains
the actual API responses and model checksum. These examples demonstrate
the upload flow, not detector accuracy. [Full evaluation](../evaluation/artwork-v3/README.md).

To record the current site, start the website and API, then run from
`frontend`:

```sh
node scripts/record-demo.mjs --output ../.turbo/demo-recording
```

The script captures real browser interactions without mocked predictions.

Credits: [NASA / William Anders, Apollo 8, 1968](https://science.nasa.gov/resource/image-earthrise/)
for Earthrise; OpenAI image generation for the landscape. The sample MP4
uses that generated still and does not demonstrate motion detection.
[Sample credits](../../frontend/public/samples/credits.json).
