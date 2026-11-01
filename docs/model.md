# Default detector

The default is [onnx-community/ai-image-detection-ONNX](https://huggingface.co/onnx-community/ai-image-detection-ONNX), a CPU-compatible export from [capcheck](https://huggingface.co/capcheck/ai-image-detection), derived from [dima806’s CIFAKE ViT](https://huggingface.co/dima806/ai_vs_real_image_detection). Its Apache-2.0 license is separate from this repository’s MIT license. No paid API or account is required. Download once; analysis then runs on your server.

| Artifact | Pinned value |
| --- | --- |
| Revision | e3cfe99f2841930a040a6281682c10c989965603 |
| File | onnx/model.onnx |
| Bytes | 343401688 |
| SHA-256 | 44cb205f596f2530686bf4895e3678f08f9483c926b6bdc0e02998d5b54978a2 |
| Input | RGB, 224 × 224, bilinear resize |
| Normalize | Scale 1/255, channel mean 0.5, standard deviation 0.5 |
| Labels | 0 = real, 1 = fake |

The installer verifies size and checksum before replacing the artifact, then writes processor settings and provenance to model.onnx.json. It preserves a different existing model unless you explicitly pass --overwrite.

## Scores and limitations

The percentage is the softmax score for the AI-generated class, not a calibrated probability of authenticity. The default threshold is 0.5. High scores can still be wrong.

The upstream author warns about concept drift: the training data represents older generators. New generators, unfamiliar subjects, screenshots, recompression, and mixed-origin images may confuse it. Veritas has no independent representative accuracy benchmark for this checkpoint.

In our demonstration, the generated lakehouse scored about 98.8%. NASA’s genuine Earthrise photograph incorrectly scored about 89.5% AI-generated. These examples show behavior and a false positive, not an accuracy evaluation. The older synthetic-fixture training metrics in this repository also do not establish real-world accuracy.

Video scores average a bounded sample of image-frame scores. They do not analyze audio, temporal consistency, face identity, or all frames. No readable frames produces an error, never a real verdict.

## Custom models and heatmaps

The original image/audio training and export tools remain. Image exports save processor metadata, the ONNX artifact hash, and the Torch state hash. A custom model without a sidecar uses the historical training normalization and bilinear resize, and responses identify it as custom.

Optional Grad-CAM requires a configured Torch checkpoint matching the state hash recorded by the image export. No matching explanation checkpoint ships with the downloadable detector, so the default site does not offer a heatmap. A heatmap would not prove manipulation.
