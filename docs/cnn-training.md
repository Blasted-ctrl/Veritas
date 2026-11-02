# Train the PyTorch CNN

Veritas includes a ConvNeXt-Tiny CNN for learning spatial differences between genuine and generated images. It uses ImageNet initialization, a two-class head, cross-entropy loss and AdamW. AdamW updates the weights using gradients from backpropagation. The CNN is the architecture; AdamW is the optimizer.

This is a practical transfer-learning baseline, not a claim that one architecture wins on every generator. [ConvNeXt](https://arxiv.org/abs/2201.03545) is a convolutional network. The implementation uses [TorchVision's ConvNeXt-Tiny](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html) and [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW).

The website's downloadable default remains the pinned CIFAKE-trained ViT. The small CNN pilot produced too many false positives to justify replacing it. See [measured results](evaluation/README.md).

## Prepare data

From the repository root, install the training and serving extras:

    python -m pip install -e "backend[ml,api,dev]"
    cd backend

Prepare licensed images in real/ and fake/ folders, or supply train.csv, val.csv and test.csv directly:

    veritas prepare-data --modality image --source directory --input-dir /path/to/images --output-dir ../data/processed/image

A manifest uses these columns:

    path,label,subject_id,modality,source,split,generator_id,source_url,license
    images/camera-01.jpg,0,scene-01,image,camera-set,train,,https://example.org/source,CC0
    images/generated-02.png,1,scene-02,image,generated-set,train,generator-family,https://example.org/source,permitted-use

Paths resolve relative to the CSV directory; existing working-directory paths also remain supported. Labels are real=0 and fake=1. Old manifests without the three provenance columns still load. Fill those columns when you know the origin; do not infer ground truth from a detector's prediction.

Use subject_id as a grouping key. Frames from one video, the same person, source-photo edits, and related generated variants belong together. Keep each group in one split. The directory adapter cannot infer all of these relationships, so review its groups before training. Before downloading facial benchmarks such as FaceForensics++ or DFDC, check their access and usage terms.

Training rejects groups and identical encoded or decoded images that cross splits. Each split must contain both classes. Exact hashes do not catch near duplicates or verify curator identities.

For a generator-family holdout, label every fake's generator_id and pass --heldout-generator; that family may occur only in test.

Balance subjects, sources, resolution and compression across classes so the model cannot succeed by learning the collection process.

Newly generated images are useful known-positive examples. Reserve some generators and prompts for evaluation before training. Keep edits and variations with their source group. Add genuine camera negatives from comparable subjects and conditions. A few fresh generated pictures are diagnostics, not an accuracy benchmark. [UniversalFakeDetect](https://arxiv.org/abs/2302.10174) and [GenImage](https://arxiv.org/abs/2306.08571) discuss the importance of generalizing beyond the training generators.

## Train

This is a starting GPU recipe; tune on validation data and keep test data sealed:

    veritas train-image --architecture convnext_tiny --data-dir ../data/processed/image --output-dir ../data/models/cnn-image --device cuda --epochs 10 --batch-size 16 --freeze-epochs 1 --unfreeze-stages 1

The first run downloads ImageNet weights. Choose a batch size that fits the GPU. Use --device cpu for a slower local run.

The trainer first learns the classification head, then fine-tunes the selected top CNN stages. --unfreeze-stages accepts 0 through 4. Defaults use a head learning rate of 0.001 and a backbone rate of 0.00005, with warmup, decay and gradient clipping.

Only training images receive random crops, horizontal flips and JPEG recompression. Evaluation uses deterministic full-image bilinear resizing to the configured image size (224×224 by default) and ImageNet normalization. This serving contract is saved in the ONNX sidecar; it differs from the original ImageNet weights' evaluation crop recipe.

The checkpoint with the highest validation ROC-AUC is evaluated on test once.

The output includes model.pt, config.json, data-audit.json and metrics.json. Metrics record the seed, configuration, selected sample counts, dataset fingerprint and completed Torch state hash. Export checks that the saved weights and audit match that completed run. Use a fresh output directory for each experiment.

For a generator holdout, append, for example:

    --heldout-generator openai-image-generation

This requires that family to be present in test and absent from train and validation. It does not obtain data automatically. Use --architecture vit to train ViT. Train audio with train-audio.

## Export, evaluate and try the CNN in the site

From backend:

    veritas export --modality image --model-dir ../data/models/cnn-image
    veritas evaluate-image --model ../data/models/cnn-image/model.onnx --manifest ../data/processed/image/test.csv --training-dir ../data/processed/image --output ../data/reports/cnn-test.json

Export uses the checkpoint's image size and writes model.onnx plus model.onnx.json. The evaluator uses the API's actual processor and ONNX Runtime. It reports confusion counts, accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, average precision and the genuine-image false-positive rate. Per-source and per-generator results expose uneven performance. A generator group contains only fakes, so its accuracy measures recall; its balanced accuracy and ROC-AUC are undefined.

Every report includes individual predictions and the same examples under fixed JPEG quality 75 and half-resolution variants. Variants are correlated observations.

The evaluator rejects duplicate test pixels and non-test rows. --training-dir also checks train/validation group and exact-content separation. It cannot verify overlap with an upstream model's undisclosed training data.

Choose a decision threshold using validation data before evaluating test; the default is 0.5.

To serve your CNN, set VERITAS_IMAGE_MODEL to the exported graph before starting the API. In Windows PowerShell, from backend:

    $env:VERITAS_IMAGE_MODEL = (Resolve-Path ../data/models/cnn-image/model.onnx).Path
    uvicorn veritas.api.app:app --host 127.0.0.1 --port 8000

On macOS or Linux:

    VERITAS_IMAGE_MODEL=../data/models/cnn-image/model.onnx uvicorn veritas.api.app:app --host 127.0.0.1 --port 8000

The site reads model attribution from the API. Check its model details after restarting. CNN heatmaps are unavailable; the existing explainer supports matching ViT checkpoints only. Keep weights and datasets outside git.

## Reproduce the bounded pilot

From backend:

    python scripts/prepare_cifake_pilot.py --output ../data/cifake-pilot
    veritas train-image --data-dir ../data/cifake-pilot --output-dir ../data/models/cnn-pilot --device cpu --epochs 2 --batch-size 4 --freeze-epochs 1 --unfreeze-stages 1 --seed 1337

The importer selects 128 training, 32 validation and 64 test images from the author's pinned CIFAKE revision. It verifies downloaded Git blobs and keeps source URLs and the author's license declaration. Selection is seeded within bounded GitHub listings, not a uniform sample of the entire dataset.

CIFAKE has 32-pixel images from an older generator; this pilot checks the pipeline and reveals errors. It does not validate current deepfake detection.

## Video scope

The CNN evaluates images and sampled video frames. It does not model motion, blinking, audio or lip synchronization. A temporal network would need labeled video sequences, consistent preprocessing and video/identity-separated evaluation. Face cropping suits a facial-manipulation dataset; applying it to general AI-image detection would discard landscapes and other useful content.
