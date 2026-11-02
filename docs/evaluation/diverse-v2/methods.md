# Diverse CNN experiment

This experiment compares a PyTorch EfficientNet-B0 trained on broader data with the previous CIFAKE-trained ConvNeXt-Tiny. The goals are fewer false alarms on genuine images and an attempt at 95% balanced test accuracy.

## Data

The subset comes from [OpenFake at revision 3247b942](https://huggingface.co/datasets/ComplexDataLab/OpenFake/tree/3247b942f18d39d1fd476b1077852af033d35d8c). Seeded, noncontiguous metadata pages and fixed source quotas determine which images enter the experiment.

| Role | Images | Real | AI-generated | Use |
| --- | ---: | ---: | ---: | --- |
| Training | 4,096 | 2,048 | 2,048 | Weight updates |
| Tuning | 512 | 256 | 256 | Checkpoint and candidate selection |
| Calibration | 512 | 256 | 256 | Decision thresholds |
| Same-distribution test | 512 | 256 | 256 | Final comparison |
| Generator/source holdout | 1,024 | 512 | 512 | Primary final comparison |

Training includes 63 generator IDs and variants. Examples include Flux.2-dev (32 images), Qwen Image (32), SD3.5 (256), GPT Image 1 (101), and Imagen 4 (23). Genuine training images come equally from LAION and Pexels.

The primary holdout contains 20 generator IDs, including GPT Image 2, GPT Image 1.5, Midjourney 7, and Flux.2 Klein 9B. Those exact IDs are absent from the other four roles. Its genuine images come equally from DOCCI and ImageNet. Generator families can share earlier versions across training and test.

Normalized prompt groups stay in one role. The importer also checks original source hashes, downloaded bytes, decoded RGB content, and cross-role perceptual hashes. These checks reduce duplicate leakage; curator identities and labels remain unverified.

[Exact assignments and hashes](split-assignments.csv) and [source counts](data-source.json) were committed before final test predictions. Downloads are viewer JPEG derivatives. Original publisher hashes and downloaded-file hashes are recorded separately. JSON and CSV evidence is stored without Git line-ending conversion so raw file hashes remain verifiable across platforms.

## Training and selection

EfficientNet-B0 starts from TorchVision's ImageNet weights. The frozen lower blocks produce two deterministic augmentation views of each training image. AdamW and cross-entropy train the final MBConv block, final convolution, and classifier after one epoch of head warmup. A CPU probe measured the cached upper-block training step at about 0.0245 seconds per image for EfficientNet-B0 and 0.2101 for ConvNeXt-Tiny, using two threads.

The [published protocol](protocol.json) fixes two candidates, each with eight epochs. They differ in learning rates and the loss weight for genuine images. Checkpoints rank by tuning standardized partial ROC-AUC at a maximum 5% false-positive rate, then full AUC and earlier epoch. Test predictions have no role in that choice. The [selected model hashes and fixed thresholds](https://github.com/Blasted-ctrl/Veritas/commit/713cf43f478495b96c7f368d20504f61e0ade86a) were published before the generator/source holdout was scored.

A separate calibration set chooses each CNN's threshold: maximize balanced accuracy while keeping its measured genuine-image false-positive rate at or below 5%. Both thresholds are locked before either final test is scored. This procedure chooses a decision threshold; the displayed class scores remain uncalibrated.

Both CNNs are evaluated on identical examples at threshold 0.5 and at their calibration-selected thresholds. Fixed JPEG quality 75 and half-resolution variants test sensitivity to common image changes. These variants are transformed copies of the same observations.

## Interpreting the measurements

The test sets are balanced, so ordinary accuracy and balanced accuracy agree. Real-image false alarms and AI-image recall expose the tradeoff that overall accuracy can hide. Generator groups contain only AI images, so their accuracy is recall.

The backbone, data and fitting schedule changed together, so this comparison measures the combined update. It does not isolate an architecture advantage.

The sample represents these source quotas and curator labels. Confidence intervals treat images as independent observations; related content and generators can correlate errors. ImageNet initialization may already have exposed the backbone to some ImageNet holdout photographs. Separation here concerns binary detector fitting, and perceptual hashes do not establish human-identity separation.

OpenFake declares CC-BY-NC-4.0. Dataset and research-model terms are separate from the repository's MIT code license. Raw dataset images and trained weights stay outside git.

See the [CPU training and evaluation commands](../../cnn-training.md#diverse-data-research-workflow) and [software verification](verification.json).
