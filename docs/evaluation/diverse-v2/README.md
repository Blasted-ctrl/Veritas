# Diverse CNN results

The new PyTorch CNN reduced false alarms on real images and improved balanced test accuracy. **It did not reach 95% accuracy.** On the primary 1,024-image generator/source holdout, accuracy rose from **47.66% to 73.34%** at threshold 0.5; real-image false alarms fell from **69.73% to 14.45%**.

This is a bounded CPU transfer-learning experiment: 4,096 training images, two eight-epoch candidates, and independent tuning, calibration and final tests. Training includes 63 generator IDs/variants and genuine images from LAION and Pexels. The primary test holds out 20 generator IDs and uses DOCCI/ImageNet for genuine images. Exact versions include GPT Image 2, GPT Image 1.5, Midjourney 7 and Flux.2 Klein 9B. These are curator labels, not independently verified identities.

## Primary generator/source holdout

Both CNNs scored the same 512 genuine and 512 generated images. Accuracy equals balanced accuracy because the classes are balanced. Recall below means the share of AI images flagged.

| Model | Decision policy | Accuracy | Real-image false alarms | AI recall | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Previous ConvNeXt | Threshold 0.5 | 47.66% | 69.73% | 65.04% | 0.5182 |
| Previous ConvNeXt | Calibration-selected | 50.98% | 1.76% | 3.71% | 0.5182 |
| Diverse EfficientNet-B0 | Threshold 0.5 | 73.34% | 14.45% | 61.13% | 0.8083 |
| Diverse EfficientNet-B0 | Calibration-selected | 61.13% | 1.37% | 23.63% | 0.8083 |

At 0.5, the new CNN falsely flagged **74 of 512** real images, compared with **357** for the old CNN. It also missed **199 of 512** AI images, compared with **179** for the old CNN. The accuracy gain came with slightly lower AI recall at this operating point. Of the paired examples, 398 changed from wrong to correct and 135 changed from correct to wrong.

The new CNN’s accuracy has a 95% Wilson interval of **70.55–75.96%** at 0.5. This describes uncertainty for these sampled images, with the dependence limits described in the methods. It is not a claim of general accuracy on every generator.

## Fewer false alarms means more missed AI images

A separate 512-image calibration set selected each threshold before either final test was scored. The rule maximizes balanced accuracy within a measured 5% false-positive budget. The locked thresholds are **0.7739539146423341** for the previous CNN and **0.8961252570152284** for the new CNN. This calibrates decisions, not probabilities.

At its conservative threshold, the new CNN flagged **7 of 512** real images and caught **121 of 512** AI images on the primary test. It missed **391** AI images. Accuracy was **61.13%** (95% Wilson interval **58.11–64.07%**). The observed false-alarm rate was **1.37%**, with interval **0.66–2.79%**. A calibration budget is not a future-source guarantee.

The selected model hashes and thresholds were [published before the generator/source holdout was scored](https://github.com/Blasted-ctrl/Veritas/commit/713cf43f478495b96c7f368d20504f61e0ade86a). Test predictions did not select or retune the model.

## Separate same-distribution test

These 512 images use the training sources and generator mix, with separate groups and content. They were also kept out of checkpoint and threshold selection.

| Model | Policy | Accuracy | Real-image false alarms | AI recall |
| --- | --- | ---: | ---: | ---: |
| Previous ConvNeXt | 0.5 | 52.34% | 60.94% | 65.62% |
| Previous ConvNeXt | Calibration-selected | 50.39% | 1.56% | 2.34% |
| Diverse EfficientNet-B0 | 0.5 | 77.15% | 24.61% | 78.91% |
| Diverse EfficientNet-B0 | Calibration-selected | 67.38% | 3.52% | 38.28% |

Do not compare these percentages directly with the original 64-image CIFAKE pilot’s 82.8%. That was a different, much smaller test set. The tables here compare old and new CNNs on identical examples.

## Compression and resizing

These are transformed copies of the primary test, not additional independent samples. The full report includes both CNNs under both policies.

| New CNN variant | Accuracy at 0.5 | False alarms at 0.5 | AI recall at 0.5 | Conservative accuracy | Conservative false alarms | Conservative AI recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 73.34% | 14.45% | 61.13% | 61.13% | 1.37% | 23.63% |
| jpeg75 | 73.34% | 14.65% | 61.33% | 61.04% | 1.37% | 23.44% |
| downscale50 | 73.34% | 12.11% | 58.79% | 60.16% | 1.17% | 21.48% |

## Withheld generators

Each row contains AI images only, so these values measure recall. Counts are small and uneven; exact held-out versions can share generator families with training.

| Curator generator ID | Images | New CNN recall at 0.5 | Conservative recall |
| --- | ---: | ---: | ---: |
| aurora-20-1-25 | 4 | 50.00% | 0.00% |
| ernie-image | 4 | 100.00% | 50.00% |
| ernie-image-turbo | 5 | 100.00% | 80.00% |
| flux.2-klein-9b | 93 | 51.61% | 10.75% |
| frames-23-1-25 | 1 | 0.00% | 0.00% |
| gpt-image-1.5 | 64 | 54.69% | 18.75% |
| gpt-image-2 | 10 | 40.00% | 0.00% |
| halfmoon-4-4-25 | 5 | 60.00% | 20.00% |
| ideogram-2.0 | 3 | 0.00% | 0.00% |
| illustrious | 80 | 80.00% | 56.25% |
| lumina-17-2-25 | 6 | 66.67% | 16.67% |
| midjourney-7 | 26 | 46.15% | 7.69% |
| nano-banana-pro | 4 | 50.00% | 50.00% |
| recraft-v2 | 3 | 0.00% | 0.00% |
| recraft-v3 | 17 | 29.41% | 0.00% |
| seedream-v5.0 | 4 | 50.00% | 0.00% |
| sora-2 | 10 | 90.00% | 20.00% |
| veo-3 | 27 | 44.44% | 33.33% |
| wan-video-2.5 | 18 | 66.67% | 27.78% |
| z-image-turbo | 128 | 70.31% | 20.31% |

## Genuine sources in the primary test

| Source | Real images | Previous false alarms at 0.5 | New false alarms at 0.5 | New conservative false alarms |
| --- | ---: | ---: | ---: | ---: |
| openfake:docci | 256 | 61.72% | 19.14% | 2.34% |
| openfake:imagenet | 256 | 77.73% | 9.77% | 0.39% |

## Inspect and reproduce

The [methods](methods.md) describe data selection, grouping, initialization, gradient-based AdamW training, checkpoint selection, and limits. The backbone, data and fitting schedule changed together; this comparison does not isolate an architecture advantage. OpenFake declares CC-BY-NC-4.0 research terms, separate from the MIT project code.

- [Protocol](protocol.json), [data counts](data-source.json), [exact split assignments](split-assignments.csv), and [data audit](data-audit.json).
- [Training history](training-history.json), [selected checkpoint](selection.json), [evaluation lock](evaluation-lock.json), and [thresholds](thresholds.json).
- [Complete metrics and confidence intervals](summary.json).
- Per-image scores: [calibration](calibration-predictions.csv), [same-distribution test](id-test-predictions.csv), [generator/source holdout](test-predictions.csv).
- [Exported-score checks](artifact-check.json), [Torch/ONNX parity](export-parity.json), and [software/API verification](verification.json).
- [Training and serving commands](../../cnn-training.md#diverse-data-research-workflow).

All 2,048 exported prediction rows were checked against provenance, and all final aggregate metrics were recomputed from the raw scores. The website’s downloadable default remains a separate pretrained ViT. This experiment does not establish that the research CNN is ready to replace it. Known diagnostic images are not an accuracy benchmark; the conservative CNN threshold misses both the generated cat and lakehouse examples.

Use these published tests for reporting and regression checks. Further model or threshold selection needs fresh tuning/calibration data and a new sealed final test.
