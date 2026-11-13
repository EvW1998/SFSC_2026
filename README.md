# SFSC_2026

A two-stage baseline for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026). YOLO26m detection proposes high-recall regions on the original 2048 × 2048 images. An image-only U-Net segments each crop, and connected-component post-processing converts its foreground probabilities into individual filament masks.

```text
Image → YOLO26m boxes → expanded native-resolution crops → U-Net probabilities
      → connected components → quality filtering → mask deduplication
      → non-overlapping COCO RLE instances
```

One crop can produce multiple instances. The second stage uses only the crop image, not a coarse segmentation mask. Box and probability caches are kept separate so that mask post-processing can be compared without rerunning the models.

## Setup

Use Python 3.10–3.12. Create and activate an environment, then install the dependencies. For GPU use, install a matching CUDA-enabled PyTorch/torchvision pair in that environment before installing the requirements.

```sh
python -m venv .venv
# Activate .venv for your shell before installing dependencies.
python -m pip install -r requirements.txt
```

The commands below use a single GPU. Use `--device cpu` for CPU checks; full-resolution training on CPU is not the intended setup. The example batch sizes and thresholds are starting points, not measured optima.

Obtain the competition data through Kaggle and extract the original directory at the repository root. Supply the following local model assets:

| Local file | Purpose |
| --- | --- |
| `model_assets/weights/yolo26m.pt` | Trusted YOLO26m **detection** checkpoint, not `yolo26m-seg.pt`. |
| `model_assets/weights/resnet18_imagenet/model.safetensors` | [Pinned SMP ResNet18 ImageNet encoder](https://huggingface.co/smp-hub/resnet18.imagenet/resolve/3f2325ff978283d47aa6a1d6878ca20565622683/model.safetensors), used to initialize U-Net training. |

The encoder SHA-256 is `b1d1d36f298c29c07afa9d5be76d888b01d174824177026613bbd64a26612d19`. A trained U-Net checkpoint contains the encoder and does not require the initialization file during inference. The repository does not include data, weights, caches, or submissions.

```text
MAGFiLO_1.0_Kaggle_2026/
  train/
    MAGFiLO_1.0_Annotations_kaggle2026_train.json
    train_images/
  test/
    test_images/
model_assets/weights/
  yolo26m.pt
  resnet18_imagenet/model.safetensors
```

## Run the cascade

Run the examples from the repository root. Each command writes to a new output directory and refuses to overwrite existing results. Training requires the explicit `--run-training` flag.

### 1. Build the fixed folds and train the detector

```sh
python -m dataset_tools.prepare build --source MAGFiLO_1.0_Kaggle_2026 --output datasets/magfilo_max_3fold --folds 3 --seed 2026 --gap-hours 72
python -m server_tools.detect_unet_cascade prepare-det --fold 0 --out runs/fold0/dataset
python -m server_tools.detect_unet_cascade train-det --dataset runs/fold0/dataset/dataset.yaml --weights model_assets/weights/yolo26m.pt --imgsz 2048 --epochs 100 --batch 2 --device 0 --out runs/fold0/detector --run-training
```

The fixed split groups nearby observations and keeps all annotation versions of the same image together. Training selects one complete annotation version per image (largest instance count, with a deterministic tie break). Detection targets are the rasterized instance-mask bounding boxes. Data preparation copies images by default, so allow disk space for both the prepared data and the detection view.

### 2. Select a high-recall operating point

```sh
python -m server_tools.detect_unet_cascade detect --fold 0 --split val --weights runs/fold0/detector/weights/last.pt --conf 0.01 --max-det 300 --out runs/fold0/boxes_val
python -m server_tools.detect_unet_cascade coverage --fold 0 --boxes runs/fold0/boxes_val/boxes.json --thresholds 0.01 0.03 0.05 0.10 0.20 0.35 --out runs/fold0/coverage
```

`coverage.json` compares box recall at IoU ≥ 0.5, GT instances with at least 90% of their mask inside one expanded crop, and the number of crops per image. Choose the confidence threshold using both coverage and cost. A filament outside every crop cannot be recovered by U-Net. Lowering a threshold below the cache floor requires new detection. If the detector is running in its NMS-free mode, the `--iou` setting is not a conventional NMS threshold.

### 3. Train the crop U-Net

```sh
python -m server_tools.detect_unet_cascade detect --fold 0 --split train --weights runs/fold0/detector/weights/last.pt --conf 0.01 --out runs/fold0/boxes_train
python -m server_tools.detect_unet_cascade train-unet --fold 0 --train-boxes runs/fold0/boxes_train/boxes.json --context 1.5 --min-side 96 --side 384 --epochs 50 --batch 8 --device cuda:0 --out runs/fold0/unet --run-training
```

Targets contain all visible foreground from the selected annotation version within the crop. Training uses GT-centered and jittered crops, background negatives, and optionally detector boxes from the **training partition of the same fold**. Those detector boxes are in-sample predictions, not out-of-fold predictions. Omit `--train-boxes` for a GT-and-background-only control.

The U-Net has a ResNet18 encoder and uses equal-weight BCE and soft Dice losses. Crops are expanded in native image coordinates, letterboxed for U-Net, and restored before mask extraction. Oracle crop Dice is a diagnostic, not an end-to-end PQ result.

### 4. Cache probabilities and compare mask rules

Replace the example `--box-conf 0.05` with the value selected in step 2.

```sh
python -m server_tools.detect_unet_cascade segment --fold 0 --split val --boxes runs/fold0/boxes_val/boxes.json --box-conf 0.05 --weights runs/fold0/unet/last.pt --out runs/fold0/prob_val
python -m server_tools.detect_unet_cascade postprocess --probabilities runs/fold0/prob_val --out runs/fold0/post_base
python -m server_tools.detect_unet_cascade postprocess --probabilities runs/fold0/prob_val --min-area 128 --out runs/fold0/post_area128
python -m server_tools.detect_unet_cascade postprocess --probabilities runs/fold0/prob_val --min-area 200 --out runs/fold0/post_area200
python -m server_tools.detect_unet_cascade postprocess --probabilities runs/fold0/prob_val --peak 0.90 --mean 0.70 --out runs/fold0/post_quality
python -m server_tools.detect_unet_cascade postprocess --probabilities runs/fold0/prob_val --closing 3 --out runs/fold0/post_close3
```

The default is `p > 0.5`, without closing or area/quality filtering. Each crop is processed independently: threshold, optionally apply elliptical closing, extract 8-connected components, and filter by area and the original probability map's peak and mean. Area thresholds and kernel sizes refer to original-image pixels. Closing may bridge a gap in one filament or accidentally join two separate filaments, so retain the no-closing control.

After mapping candidates back to the image, the pipeline removes cross-crop duplicates by mask IoU and assigns any remaining shared pixels to the instance with the highest mean probability. Empty masks are removed. The validation report computes pooled PQ, SQ and RQ over **all annotation versions**, using the competition's strict IoU > 0.5 all-pairs matching. It also distinguishes misses with no crop overlap, partial coverage, and at least 90% coverage. Run `segment --oracle` in a separate validation directory to diagnose the second stage with GT-derived boxes; do not report this as detector-based PQ.

For test export, repeat `detect` and `segment` with `--split test` and new output directories, then run `postprocess` on the test probability cache. It writes and validates `submission.csv` locally. Images without instances are recorded in the audit; no placeholder masks are invented. The entry points do not upload files to Kaggle. A single-fold export is not a full-data retrain or a fold ensemble.

## Tests and status

```sh
python -B -m unittest discover -s tests -v
```

Tests cover coordinate restoration, per-crop components, mask filters, cross-crop deduplication, zero overlap, recall/coverage, and cache-to-PQ/CSV integration. The real-fold test runs when the prepared competition data are present. The new detector/U-Net cascade has not been formally trained or benchmarked; this repository does not claim a leaderboard result for it.
