# Teaching Veritas about human-made artwork

An anime frame is not an AI image simply because it is illustrated. This experiment adds human-made illustrations and paintings to the non-AI class, and AI-generated artwork to the AI class. There is no rule that automatically exempts drawings from detection.

## Data and separation

| Role | Human illustration | Human painting | AI illustration | AI painting | General images | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Training | 256 | 256 | 256 | 256 | 512 | 1,536 |
| Validation | 32 | 32 | 32 | 32 | 128 | 256 |
| Calibration | 32 | 32 | 32 | 32 | 128 | 256 |
| Test | 64 | 64 | 64 | 64 | 256 | 512 |

The human illustration source is [Pixiv daily illustrations from 2018](https://www.kaggle.com/datasets/stevenevan99/pixiv-top-daily-illustration-2018), version 1. Its curator describes collecting the dataset in January 2021. This gives us dated human artwork rather than treating every current image tagged “anime” as non-AI. The bounded sample uses the first 12 metadata pages, with seeded selection within that pool; it is not a uniform sample of all Pixiv.

Human paintings come from [ArtBench public-domain artwork](https://huggingface.co/datasets/alfredplpl/artbench-pd-256x256), pinned to `4056b70025f0e3f916fb271c7a792633ba637484`. Its AI-generated captions are ignored when assigning origin labels. Historical artwork and artist metadata supply the non-AI label.

Generated artwork comes from [Lunara Art Eval](https://huggingface.co/datasets/moonworks/lunara-art-eval), pinned to `5a38a59769a6146e8d1b223c58533d12c60fd7d6`. It includes eight generators: Lunara, HiDream-I1, Stable Diffusion 3.5, Z-Image-Turbo, FLUX.2 klein 4B, AuraFlow, Qwen-Image and GPT Image 1 Mini. Explicit prompt descriptions assign coarse illustration/painting categories. A prompt saying “hand-drawn” still receives the AI label when its published origin is an AI generator.

All classes use the same RGB, 256 × 256 bilinear resize and JPEG quality 95 preparation. This reduces obvious resolution/file-format shortcuts, but does not eliminate source-specific shortcuts. The dataset labels are curator assertions, not independently authenticated origin certificates.

Artist groups stay together for ArtBench, artwork IDs for Pixiv, and normalized prompts across generators for AI images. Exact encoded/pixel duplicates are excluded, and cross-role dHash distances of four bits or fewer are rejected. Pixiv artist identities are unavailable, so artist separation is **not** established for that cohort. The painting test contains 13 artist groups; 64 images do not mean 64 independent artists.

The general-image replay uses the previous OpenFake experiment's corresponding training, validation and calibration roles. Its test subset comes from the previously published same-distribution test. It is a **regression check**, not a newly blind photography benchmark. The new artwork test uses newly withheld groups. The same eight artwork generators occur across roles, so this is not an unseen-generator test.

The reported Broly image is excluded from fitting, validation, calibration and every test aggregate. Its SHA-256 is `400d0d9e1d5f01d2d3a33a8aadc1858d3748787c079247625cfdd16b53703241`. It is a separate, user-reported non-AI diagnostic, scored only after the final test.

[Protocol](protocol.json) · [Actual cohort counts and groups](cohort.json) · [Source identities](source.json) · [Image/provenance index](dataset-index.csv) · [Data audit](data-audit.json)

## Training and evaluation

The PyTorch EfficientNet-B0 CNN starts from the completed diverse-data CNN. AdamW updates the final MBConv block, final convolution and classifier using cross-entropy, with a 1.25 loss weight for non-AI examples. The lower convolution blocks remain frozen; two cached augmentation views keep the local CPU run bounded. This is fine-tuning with gradient descent, not a hand-written style filter.

The registered run has eight epochs and one learning-rate configuration. Validation partial ROC-AUC at 5% false-positive rate selects the epoch, with full ROC-AUC and earlier epoch as tie-breakers. Test predictions never select a checkpoint.

Three fixed models are compared: the website's downloadable CIFAKE ViT, the previous diverse CNN, and the artwork-adapted CNN. Each gets its own threshold from the same separate calibration set, targeting an empirical false-positive rate at or below 5%. All thresholds and model/processor identities are saved before test inference. Results also report the common 0.5 threshold so calibration cannot conceal a weak classifier.

The test includes original prepared images, JPEG quality 75 recompression, and half-resolution variants. These are transformations of the same examples, not three independent test sets. False-positive rate measures human images incorrectly labeled AI; recall measures AI images detected. Small, correlated source groups limit the accuracy estimates. No result establishes 95% accuracy across arbitrary uploads, generators, resolutions or styles.

## Reproduce

First complete the [diverse CNN workflow](../../cnn-training.md) to obtain its data and selected checkpoint, and download the default detector. With the Python ML/API extras installed, run from `backend`:

```text
python scripts/prepare_artwork_pilot.py --output ../data/artwork-v3 --photo-replay ../data/openfake-research
python scripts/train_diverse_cnn.py --data-dir ../data/artwork-v3 --output-dir ../data/models/artwork-v3 --cache-dir ../data/artwork-features --protocol ../docs/evaluation/artwork-v3/protocol.json --initialize-from ../data/models/diverse-cnn/selected
python scripts/evaluate_artwork_cnn.py --data-dir ../data/artwork-v3 --models-dir ../data/models/artwork-v3 --previous-model ../data/models/diverse-cnn/selected/model.onnx --previous-data ../data/openfake-research --default-model ../data/models/image/model.onnx --protocol ../docs/evaluation/artwork-v3/protocol.json --output ../data/reports/artwork-v3 --phase calibrate
```

Preserve the calibration output and model bundles before running the identical evaluation command with `--phase test`. Use fresh output directories for a new experiment. The evaluator refuses a second test start in the same directory. Acquisition validates source revisions and refreshes expiring image links; changed source identities stop the run.

The image index records identities and attribution, not raw artwork. Original artists retain their image rights. ArtBench declares public-domain artworks with MIT dataset metadata; Lunara declares Apache-2.0 for its dataset. The inherited OpenFake data/checkpoint carries CC-BY-NC-4.0 research restrictions. These are separate from Veritas's MIT code license; this pilot does not grant commercial rights to the images or resulting weights.
