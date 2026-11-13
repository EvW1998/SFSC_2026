#!/usr/bin/env python3
"""Build fixed max-instance MAGFiLO folds and single-instance refinement crops.

Run from the repository root: python -m dataset_tools.prepare --help
Image operations use Pillow, numpy and the official COCO mask rasterizer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


SCHEMA = "magfilo-max-instances-3fold-v1"
CATEGORY = {"id": 1, "name": "filament", "supercategory": "filament"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts):
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)


def materialize(source, target, mode):
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "symlink":
        target.symlink_to(source.resolve())
    elif mode == "hardlink":
        os.link(source, target)
    else:
        raise ValueError(mode)


def observation_time(name):
    # GONG filenames encode UTC, independently of the host timezone.
    return datetime.strptime(Path(name).stem[:14], "%Y%m%d%H%M%S")


def safe_name(name):
    if Path(name).name != name or not name or "\\" in name or "/" in name:
        raise ValueError(f"Unsafe source filename: {name!r}")
    return name


def polygon_line(annotation, image):
    segments = annotation["segmentation"]
    if not isinstance(segments, list) or len(segments) != 1:
        raise ValueError(f"Expected one polygon for {annotation['id']}")
    flat = segments[0]
    if len(flat) < 6 or len(flat) % 2:
        raise ValueError(f"Invalid polygon: {annotation['id']}")
    width, height = image["width"], image["height"]
    points = []
    for x, y in zip(flat[::2], flat[1::2]):
        x, y = float(x), float(y)
        if not math.isfinite(x + y) or not (0 <= x <= width and 0 <= y <= height):
            raise ValueError(f"Out-of-bounds coordinate: {annotation['id']}")
        if not points or points[-1] != (x, y):
            points.append((x, y))
    if points[0] == points[-1]:
        points.pop()
    if len(set(points)) < 3:
        raise ValueError(f"Degenerate polygon: {annotation['id']}")
    return "0 " + " ".join(f"{v:.8f}" for x, y in points for v in (x / width, y / height))


def temporal_components(names, gap_hours):
    """Connected chronological groups: any cross-group gap exceeds gap_hours.

    This is transitive, unlike calendar bins which leak at their boundaries.
    """
    ordered = sorted(names, key=lambda name: (observation_time(name), name))
    groups, current, previous = [], [], None
    for name in ordered:
        moment = observation_time(name)
        if previous is not None and (moment - previous).total_seconds() > gap_hours * 3600:
            groups.append(current)
            current = []
        current.append(name)
        previous = moment
    if current:
        groups.append(current)
    return groups


def assign_folds(records, groups, n_folds, seed):
    """Seeded greedy balancing of image count, instances, years and stations."""
    by_name = {row["file_name"]: row for row in records}
    features = []
    for group in groups:
        counts = Counter()
        for name in group:
            counts["images"] += 1
            counts["instances"] += by_name[name]["instance_count"]
            counts["year:" + name[:4]] += 1
            counts["station:" + Path(name).stem[-2:]] += 1
        features.append(counts)
    totals = sum(features, Counter())
    targets = {key: value / n_folds for key, value in totals.items()}
    weights = {key: (6.0 if key == "images" else 2.0 if key == "instances" else 1.0) for key in targets}
    order = list(range(len(groups)))
    rng = random.Random(seed)
    rng.shuffle(order)
    order.sort(key=lambda index: (-len(groups[index]), -features[index]["instances"]))
    fold_features = [Counter() for _ in range(n_folds)]
    assignments = {}
    for group_index in order:
        group_features = features[group_index]

        def cost(fold):
            delta = 0.0
            for key, value in group_features.items():
                target = targets[key]
                before = (fold_features[fold][key] - target) / max(target, 1.0)
                after = (fold_features[fold][key] + value - target) / max(target, 1.0)
                delta += weights[key] * (after * after - before * before)
            return delta, fold_features[fold]["images"], fold

        fold = min(range(n_folds), key=cost)
        fold_features[fold].update(group_features)
        for name in groups[group_index]:
            assignments[name] = (fold, f"time_group_{group_index:04d}")
    if any(not feature["images"] for feature in fold_features):
        raise ValueError("A fold is empty; use a smaller temporal gap")
    return assignments


def normalize_licenses(raw):
    licenses = raw.get("licenses", [])
    return [licenses] if isinstance(licenses, dict) else licenses


def coco_selected(raw, records, annotation_by_image):
    images, annotations, maps = [], [], []
    next_annotation_id = 1
    for row in records:
        images.append({"id": row["image_id"], "file_name": row["file_name"],
                       "width": row["width"], "height": row["height"], "license": 1})
        for source_annotation in annotation_by_image[row["selected_image_id"]]:
            annotations.append({"id": next_annotation_id, "image_id": row["image_id"],
                                "category_id": 1, "segmentation": source_annotation["segmentation"],
                                "area": source_annotation["area"], "bbox": source_annotation["bbox"],
                                "iscrowd": int(source_annotation.get("iscrowd", 0))})
            maps.append({"annotation_id": next_annotation_id, "source_annotation_id": source_annotation["id"],
                         "source_category_id": source_annotation["category_id"]})
            next_annotation_id += 1
    return {"info": {"description": "MAGFiLO maximum-instance version, one class"},
            "licenses": normalize_licenses(raw), "categories": [CATEGORY],
            "images": images, "annotations": annotations}, maps


def subset_coco(document, image_ids):
    return {**document,
            "images": [row for row in document["images"] if row["id"] in image_ids],
            "annotations": [row for row in document["annotations"] if row["image_id"] in image_ids]}


def write_runtime_paths(dataset, final_root=None):
    """Regenerate host paths after copying a prepared dataset to a server."""
    dataset = Path(dataset)
    final_root = (Path(final_root) if final_root else dataset).resolve()
    manifest = read_json(dataset / "manifest.json")
    if manifest["schema"] != SCHEMA:
        raise ValueError("Not a supported fixed-fold dataset")
    for fold in range(manifest["n_folds"]):
        directory = dataset / "folds" / f"fold_{fold}"
        for split in ("train", "val"):
            rows = [row for row in manifest["images"] if (row["fold"] == fold) == (split == "val")]
            (directory / f"{split}.txt").write_text(
                "".join((final_root / "images" / "train" / row["file_name"]).as_posix() + "\n" for row in rows),
                encoding="utf-8")
        (directory / "dataset.yaml").write_text(
            f"path: {json.dumps(final_root.as_posix())}\n"
            f"train: folds/fold_{fold}/train.txt\nval: folds/fold_{fold}/val.txt\n"
            "test: images/test\nnames:\n  0: filament\n", encoding="utf-8")


def build(args):
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new output directory")
    if output == source or output.is_relative_to(source):
        raise ValueError("Output must be outside the original dataset")
    source_json = source / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    raw = read_json(source_json)
    by_image, by_file = defaultdict(list), defaultdict(list)
    known_image_ids = set()
    for image in raw["images"]:
        image_id = str(image["id"])
        if image_id in known_image_ids:
            raise ValueError(f"Duplicate source image id: {image_id}")
        known_image_ids.add(image_id)
        by_file[safe_name(image["file_name"])].append(image)
    seen_annotation_ids = set()
    for annotation in raw["annotations"]:
        if str(annotation["image_id"]) not in known_image_ids:
            raise ValueError("Unknown annotation image id")
        if annotation["id"] in seen_annotation_ids:
            raise ValueError("Duplicate annotation id")
        seen_annotation_ids.add(annotation["id"])
        by_image[str(annotation["image_id"])].append(annotation)
    records = []
    for numeric_id, (name, versions) in enumerate(sorted(by_file.items()), 1):
        for version in versions:
            if (version["width"], version["height"]) != (versions[0]["width"], versions[0]["height"]):
                raise ValueError("Versions have different dimensions")
        ranked = sorted(versions, key=lambda row: (-len(by_image[str(row["id"])]), str(row["id"])))
        selected = ranked[0]
        records.append({"image_id": numeric_id, "file_name": name,
                        "width": selected["width"], "height": selected["height"],
                        "selected_image_id": str(selected["id"]),
                        "instance_count": len(by_image[str(selected["id"])]),
                        "versions": [{"image_id": str(row["id"]), "instance_count": len(by_image[str(row["id"])])}
                                     for row in sorted(versions, key=lambda row: str(row["id"]))]})
    groups = temporal_components(by_file, args.gap_hours)
    if len(groups) < args.folds:
        raise ValueError("Too few temporal groups for the requested folds")
    assignments = assign_folds(records, groups, args.folds, args.seed)
    for row in records:
        row["fold"], row["temporal_group"] = assignments[row["file_name"]]
    dataset, annotation_map = coco_selected(raw, records, by_image)
    temp = output.with_name(output.name + f".partial-{os.getpid()}")
    temp.mkdir(parents=True, exist_ok=False)
    for subdirectory in ("images/train", "images/test", "labels/train", "labels/versions", "annotations"):
        (temp / subdirectory).mkdir(parents=True, exist_ok=True)
    for row in records:
        name = row["file_name"]
        materialize(source / "train" / "train_images" / name, temp / "images" / "train" / name, args.image_mode)
        image = {"width": row["width"], "height": row["height"]}
        selected_lines = [polygon_line(item, image) for item in by_image[row["selected_image_id"]]]
        (temp / "labels" / "train" / (Path(name).stem + ".txt")).write_text(
            "\n".join(selected_lines) + ("\n" if selected_lines else ""), encoding="utf-8")
        for version in row["versions"]:
            version_id = safe_name(version["image_id"])
            lines = [polygon_line(item, image) for item in by_image[version_id]]
            (temp / "labels" / "versions" / (version_id + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    test_files = sorted((source / "test" / "test_images").glob("*.jpeg"))
    for path in test_files:
        materialize(path, temp / "images" / "test" / path.name, args.image_mode)
    write_json(temp / "annotations" / "selected_all.json", dataset)
    write_json(temp / "annotations" / "annotation_id_map.json", annotation_map)
    # Preserve complete alternatives for official multi-annotator evaluation and E5.
    all_versions = {**raw, "licenses": normalize_licenses(raw), "categories": [CATEGORY],
                    "annotations": [{**item, "category_id": 1, "source_category_id": item["category_id"]}
                                    for item in raw["annotations"]]}
    write_json(temp / "annotations" / "all_versions.json", all_versions)
    stats = []
    for fold in range(args.folds):
        directory = temp / "folds" / f"fold_{fold}"
        directory.mkdir(parents=True)
        val = [row for row in records if row["fold"] == fold]
        train = [row for row in records if row["fold"] != fold]
        for split, subset in (("train", train), ("val", val)):
            write_json(directory / f"instances_{split}.json", subset_coco(dataset, {row["image_id"] for row in subset}))
        val_names = {row["file_name"] for row in val}
        val_source_ids = {str(image["id"]) for image in raw["images"] if image["file_name"] in val_names}
        write_json(directory / "validation_all_versions.json", subset_coco(all_versions, val_source_ids))
        stats.append({"fold": fold, "train_images": len(train), "val_images": len(val),
                      "train_instances": sum(row["instance_count"] for row in train),
                      "val_instances": sum(row["instance_count"] for row in val),
                      "val_year_counts": dict(sorted(Counter(row["file_name"][:4] for row in val).items())),
                      "val_station_counts": dict(sorted(Counter(Path(row["file_name"]).stem[-2:] for row in val).items()))})
    manifest = {"schema": SCHEMA, "source_annotation_sha256": digest_file(source_json),
                "seed": args.seed, "n_folds": args.folds, "temporal_gap_hours": args.gap_hours,
                "grouping": "transitive chronological components across all observatories",
                "selection": "largest instance count; ties use ascending original image_id",
                "image_mode": args.image_mode, "n_images": len(records), "n_test_images": len(test_files),
                "n_selected_instances": len(dataset["annotations"]), "n_source_instances": len(raw["annotations"]),
                "n_temporal_groups": len(groups), "largest_group": max(map(len, groups)),
                "selection_ties": sum(sum(version["instance_count"] == row["instance_count"] for version in row["versions"]) > 1 for row in records),
                "fold_statistics": stats, "images": records}
    write_json(temp / "manifest.json", manifest)
    write_runtime_paths(temp, output)
    write_json(temp / "checksums.json", {str(path.relative_to(temp)).replace("\\", "/"): digest_file(path)
                                        for path in sorted(temp.rglob("*.json"))})
    temp.rename(output)
    print(json.dumps({key: manifest[key] for key in ("n_images", "n_selected_instances", "n_temporal_groups", "largest_group", "fold_statistics")}, indent=2))
    print(f"Dataset ready: {output}", flush=True)


def decode_mask(annotation, width, height):
    import numpy as np
    from pycocotools import mask as mask_utils

    segmentation = annotation["segmentation"]
    rles = mask_utils.frPyObjects(segmentation, height, width)
    return np.asarray(mask_utils.decode(mask_utils.merge(rles)), dtype=bool)


def crop_box(bbox, width, height, context, min_side):
    x, y, w, h = map(float, bbox)
    side = min(max(width, height), max(min_side, math.ceil(max(w, h) * context)))
    # Square crop, clipped by translation where possible. No forced resizing.
    crop_w, crop_h = min(side, width), min(side, height)
    left = max(0, min(width - crop_w, math.floor(x + w / 2 - crop_w / 2)))
    top = max(0, min(height - crop_h, math.floor(y + h / 2 - crop_h / 2)))
    return left, top, left + crop_w, top + crop_h


def simulated_seed(mask_image, seed):
    """Training-only proxy for a coarse detector mask, never an OOF prediction."""
    from PIL import Image, ImageFilter

    rng = random.Random(seed)
    small = mask_image.resize((max(1, mask_image.width // 4), max(1, mask_image.height // 4)), Image.Resampling.NEAREST)
    blurred = small.resize(mask_image.size, Image.Resampling.BILINEAR)
    threshold = rng.randint(85, 170)
    result = blurred.point(lambda value: 255 if value >= threshold else 0)
    result = result.filter(ImageFilter.MaxFilter(3) if rng.random() < 0.5 else ImageFilter.MinFilter(3))
    shifted = Image.new("L", result.size, 0)
    shifted.paste(result, (rng.randint(-3, 3), rng.randint(-3, 3)))
    return shifted


def emit_crop(destination, image, mask, box, sample_id, meta, resize, seed, negative=False):
    import numpy as np
    from PIL import Image, ImageDraw

    left, top, right, bottom = box
    rgb_crop = image.crop(box)  # Grayscale PNG; loaders replicate it for RGB pretrained networks.
    gt_crop = Image.fromarray(mask[top:bottom, left:right].astype(np.uint8) * 255)
    if resize:
        rgb_crop = rgb_crop.resize((resize, resize), Image.Resampling.BILINEAR)
        gt_crop = gt_crop.resize((resize, resize), Image.Resampling.NEAREST)
    if negative:
        coarse = Image.new("L", gt_crop.size, 0)
        draw = ImageDraw.Draw(coarse)
        w, h = coarse.size
        draw.line([(w // 4, h // 3), (w // 2, h // 2), (3 * w // 4, 2 * h // 3)], fill=255, width=max(2, w // 30))
    else:
        coarse = simulated_seed(gt_crop, stable_seed(seed, sample_id))
    paths = {"image": f"images/{sample_id}.png", "gt": f"gt/{sample_id}.png", "coarse_simulated": f"coarse_simulated/{sample_id}.png"}
    for key, crop in (("image", rgb_crop), ("gt", gt_crop), ("coarse_simulated", coarse)):
        crop.save(destination / paths[key])
    return {**meta, "sample_id": sample_id, **paths,
            "crop_xyxy": list(box), "output_wh": list(rgb_crop.size),
            "scale_xy": [rgb_crop.width / (right - left), rgb_crop.height / (bottom - top)],
            "gt_foreground_pixels": int((np.asarray(gt_crop) > 0).sum()),
            "seed_is_simulated": True}


def make_crops(args):
    import numpy as np
    from PIL import Image

    dataset, output = args.dataset.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    manifest = read_json(dataset / "manifest.json")
    selected = read_json(dataset / "annotations" / "selected_all.json")
    by_image = defaultdict(list)
    for annotation in selected["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    rows = manifest["images"][:args.limit_images] if args.limit_images else manifest["images"]
    temp = output.with_name(output.name + f".partial-{os.getpid()}")
    temp.mkdir(parents=True, exist_ok=False)
    for subdirectory in ("images", "gt", "coarse_simulated", "folds"):
        (temp / subdirectory).mkdir()
    samples = []
    for image_index, row in enumerate(rows):
        image = Image.open(dataset / "images" / "train" / row["file_name"]).convert("L")
        if image.size != (row["width"], row["height"]):
            raise ValueError("JPEG size disagrees with source COCO")
        union = np.zeros((image.height, image.width), dtype=bool)
        for annotation in by_image[row["image_id"]]:
            mask = decode_mask(annotation, image.width, image.height)
            union |= mask
            box = crop_box(annotation["bbox"], image.width, image.height, args.context, args.min_side)
            left, top, right, bottom = box
            if int(mask[top:bottom, left:right].sum()) != int(mask.sum()):
                raise ValueError(f"Crop truncates GT instance {annotation['id']}")
            sample_id = f"{Path(row['file_name']).stem}__ann_{annotation['id']:06d}"
            samples.append(emit_crop(temp, image, mask, box, sample_id,
                                    {"kind": "positive", "file_name": row["file_name"], "fold": row["fold"],
                                     "image_id": row["image_id"], "source_image_id": row["selected_image_id"],
                                     "annotation_id": annotation["id"]}, args.resize, args.seed))
        rng = random.Random(stable_seed(args.seed, row["file_name"], "negative"))
        for negative_index in range(args.negatives_per_image):
            side = min(args.negative_side, image.width, image.height)
            box = None
            for _ in range(500):
                x = rng.randint(0, image.width - side)
                y = rng.randint(0, image.height - side)
                # Every crop corner lies inside a conservative solar disk.
                radius = min(image.width, image.height) * 0.40
                if any(math.hypot(px - image.width / 2, py - image.height / 2) > radius
                       for px, py in ((x, y), (x + side, y), (x, y + side), (x + side, y + side))):
                    continue
                if not union[y:y + side, x:x + side].any():
                    box = (x, y, x + side, y + side)
                    break
            if box is None:
                raise ValueError("Could not find an annotated-empty negative crop")
            sample_id = f"{Path(row['file_name']).stem}__negative_{negative_index:02d}"
            samples.append(emit_crop(temp, image, np.zeros_like(union), box, sample_id,
                                    {"kind": "negative", "file_name": row["file_name"], "fold": row["fold"],
                                     "image_id": row["image_id"], "source_image_id": row["selected_image_id"],
                                     "annotation_id": None}, args.resize, args.seed, negative=True))
        if (image_index + 1) % 50 == 0:
            print(f"Crops: {image_index + 1}/{len(rows)} images, {len(samples)} samples", flush=True)
    write_jsonl(temp / "samples.jsonl", samples)
    for fold in range(manifest["n_folds"]):
        for split in ("train", "val"):
            subset = [row for row in samples if (row["fold"] == fold) == (split == "val")]
            write_jsonl(temp / "folds" / f"fold_{fold}_{split}.jsonl", subset)
    metadata = {"schema": SCHEMA + "-crops", "source_manifest_sha256": digest_file(dataset / "manifest.json"),
                "context": args.context, "min_side": args.min_side, "resize": args.resize,
                "seed": args.seed, "negatives_per_image": args.negatives_per_image,
                "negative_side": args.negative_side, "source_images": len(rows),
                "samples": len(samples), "sample_counts": dict(Counter(row["kind"] for row in samples)),
                "ground_truth": "single instance only; neighbor filaments remain background",
                "coarse_masks": "synthetic training proxies; not detector predictions",
                "validation": "GT-box crops measure oracle refinement only, never end-to-end PQ",
                "rasterizer": "pycocotools.frPyObjects + merge + decode",
                "versions": {name: importlib.metadata.version(name) for name in ("numpy", "Pillow", "pycocotools")},
                "python": platform.python_version()}
    write_json(temp / "metadata.json", metadata)
    temp.rename(output)
    print(json.dumps(metadata, indent=2))


def choose_epoch_version(row, seed, epoch):
    """One version per physical image; balanced shuffled cycles, worker independent."""
    versions = sorted(version["image_id"] for version in row["versions"])
    cycle, position = divmod(epoch, len(versions))
    rng = random.Random(stable_seed(seed, row["file_name"], cycle))
    rng.shuffle(versions)
    return versions[position]


def verify_crops(args):
    import numpy as np
    from PIL import Image

    root, dataset = args.crops.resolve(), args.dataset.resolve()
    metadata = read_json(root / "metadata.json")
    manifest = read_json(dataset / "manifest.json")
    selected = read_json(dataset / "annotations" / "selected_all.json")
    annotations = {row["id"]: row for row in selected["annotations"]}
    images = {row["file_name"]: row for row in manifest["images"]}
    samples = [json.loads(line) for line in (root / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    errors = []
    if metadata["source_manifest_sha256"] != digest_file(dataset / "manifest.json"):
        errors.append("Crop source manifest mismatch")
    if len(samples) != metadata["samples"] or len({row["sample_id"] for row in samples}) != len(samples):
        errors.append("Invalid sample count or duplicate ids")
    kind_counts, empty_seeds = Counter(), 0
    for row in samples:
        kind_counts[row["kind"]] += 1
        source = images[row["file_name"]]
        if row["fold"] != source["fold"] or row["source_image_id"] != source["selected_image_id"]:
            errors.append(f"Crop provenance/fold mismatch {row['sample_id']}")
        with Image.open(root / row["image"]) as image, Image.open(root / row["gt"]) as gt, Image.open(root / row["coarse_simulated"]) as coarse:
            if image.size != gt.size or coarse.size != gt.size or list(image.size) != row["output_wh"]:
                errors.append(f"Shape mismatch {row['sample_id']}")
            gt_array, seed_array = np.asarray(gt), np.asarray(coarse)
            if not np.all((gt_array == 0) | (gt_array == 255)) or not np.all((seed_array == 0) | (seed_array == 255)):
                errors.append(f"Nonbinary mask {row['sample_id']}")
            area = int((gt_array > 0).sum())
            if area != row["gt_foreground_pixels"] or (row["kind"] == "negative" and area != 0):
                errors.append(f"Incorrect GT area {row['sample_id']}")
            if row["kind"] == "positive" and not area:
                errors.append(f"Empty positive GT {row['sample_id']}")
            empty_seeds += int(not seed_array.any())
    if dict(kind_counts) != metadata["sample_counts"]:
        errors.append("Sample kind totals disagree")
    sample_by_id = {row["sample_id"]: row for row in samples}
    all_sample_ids = set(sample_by_id)
    val_seen = Counter()
    fold_counts = []
    for fold in range(manifest["n_folds"]):
        split_ids = {}
        for split in ("train", "val"):
            indexed = [json.loads(line) for line in (root / "folds" / f"fold_{fold}_{split}.jsonl").read_text(encoding="utf-8").splitlines()]
            ids = {row["sample_id"] for row in indexed}
            if len(ids) != len(indexed):
                errors.append(f"Duplicate crop in fold {fold} {split}")
            if any(row != sample_by_id.get(row["sample_id"]) or ((row["fold"] == fold) != (split == "val")) for row in indexed):
                errors.append(f"Incorrect crop fold {fold} {split}")
            split_ids[split] = ids
        if split_ids["train"] & split_ids["val"] or split_ids["train"] | split_ids["val"] != all_sample_ids:
            errors.append(f"Incorrect crop partition fold {fold}")
        val_seen.update(split_ids["val"])
        fold_counts.append({"fold": fold, "train": len(split_ids["train"]), "val": len(split_ids["val"])})
    if set(val_seen.values()) != {1} or set(val_seen) != all_sample_ids:
        errors.append("Each crop must appear in validation exactly once")
    positive_samples = [row for row in samples if row["kind"] == "positive"]
    chosen = random.Random(2026).sample(positive_samples, min(args.geometry_samples, len(positive_samples)))
    for row in chosen:
        source = images[row["file_name"]]
        annotation = annotations[row["annotation_id"]]
        mask = decode_mask(annotation, source["width"], source["height"])
        left, top, right, bottom = row["crop_xyxy"]
        expected = Image.fromarray(mask[top:bottom, left:right].astype(np.uint8) * 255)
        if expected.size != tuple(row["output_wh"]):
            expected = expected.resize(tuple(row["output_wh"]), Image.Resampling.NEAREST)
        with Image.open(root / row["gt"]) as gt:
            if not np.array_equal(np.asarray(expected), np.asarray(gt)):
                errors.append(f"COCO raster/crop mismatch {row['sample_id']}")
    report = {"status": "PASS" if not errors else "FAIL", "samples": len(samples),
              "kinds": dict(kind_counts), "empty_simulated_seeds": empty_seeds,
              "geometry_samples_checked": len(chosen), "fold_counts": fold_counts, "errors": errors}
    write_json(root / "verification_report.json", report)
    print(json.dumps(report, indent=2))
    if errors:
        raise ValueError("Crop verification failed")


def epoch_manifest(args):
    dataset = args.dataset.resolve()
    manifest = read_json(dataset / "manifest.json")
    if args.fold not in range(manifest["n_folds"]) or args.epoch < 0:
        raise ValueError("Invalid fold or epoch")
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for row in manifest["images"]:
        if row["fold"] != args.fold:
            version = choose_epoch_version(row, args.seed, args.epoch)
            rows.append({"file_name": row["file_name"], "version_id": version,
                         "image": f"images/train/{row['file_name']}", "label": f"labels/versions/{version}.txt"})
    write_json(args.output, {"fold": args.fold, "epoch": args.epoch, "seed": args.seed,
                            "policy": "balanced shuffled cycles; training only; custom loader required", "samples": rows})
    print(f"Wrote {len(rows)} training selections to {args.output}")


def verify(args):
    root = args.dataset.resolve()
    manifest = read_json(root / "manifest.json")
    selected = read_json(root / "annotations" / "selected_all.json")
    if manifest["schema"] != SCHEMA:
        raise ValueError("Unrecognized schema")
    errors = []
    groups_to_fold = defaultdict(set)
    line_count = 0
    val_seen = Counter()
    for row in manifest["images"]:
        name = row["file_name"]
        if not (root / "images" / "train" / name).is_file():
            errors.append(f"Missing image {name}")
        maximum = max(version["instance_count"] for version in row["versions"])
        tie_choice = min(version["image_id"] for version in row["versions"] if version["instance_count"] == maximum)
        if row["instance_count"] != maximum or row["selected_image_id"] != tie_choice:
            errors.append(f"Bad maximum selection {name}")
        lines = (root / "labels" / "train" / (Path(name).stem + ".txt")).read_text().splitlines()
        line_count += len(lines)
        if len(lines) != row["instance_count"]:
            errors.append(f"Bad label count {name}")
        for line in lines:
            fields = line.split()
            if fields[0] != "0" or len(fields) < 7 or len(fields) % 2 != 1:
                errors.append(f"Invalid YOLO polygon {name}")
                continue
            if not all(0 <= float(value) <= 1 for value in fields[1:]):
                errors.append(f"Invalid normalized coordinate {name}")
        for version in row["versions"]:
            path = root / "labels" / "versions" / (version["image_id"] + ".txt")
            if len(path.read_text().splitlines()) != version["instance_count"]:
                errors.append(f"Bad alternative label count {version['image_id']}")
        groups_to_fold[row["temporal_group"]].add(row["fold"])
    if any(len(value) != 1 for value in groups_to_fold.values()):
        errors.append("Temporal group crosses folds")
    ordered = sorted(manifest["images"], key=lambda row: observation_time(row["file_name"]))
    for first, second in zip(ordered, ordered[1:]):
        gap = (observation_time(second["file_name"]) - observation_time(first["file_name"])).total_seconds() / 3600
        if gap <= manifest["temporal_gap_hours"] and first["fold"] != second["fold"]:
            errors.append("Nearby observations cross folds")
    all_ids = {row["image_id"] for row in manifest["images"]}
    for fold in range(manifest["n_folds"]):
        directory = root / "folds" / f"fold_{fold}"
        train = read_json(directory / "instances_train.json")
        val = read_json(directory / "instances_val.json")
        train_ids = {image["id"] for image in train["images"]}
        val_ids = {image["id"] for image in val["images"]}
        if train_ids & val_ids or train_ids | val_ids != all_ids:
            errors.append(f"Invalid image split for fold {fold}")
        val_seen.update(val_ids)
        for split, doc in (("train", train), ("val", val)):
            ids = {image["id"] for image in doc["images"]}
            if any(annotation["image_id"] not in ids or annotation["category_id"] != 1 for annotation in doc["annotations"]):
                errors.append(f"Invalid COCO references for fold {fold}")
            paths = (directory / (split + ".txt")).read_text().splitlines()
            if len(paths) != len(ids) or not all(Path(path).is_file() for path in paths):
                errors.append(f"Invalid runtime paths for fold {fold}; run rebase")
    if set(val_seen.values()) != {1} or set(val_seen) != all_ids:
        errors.append("Each image must appear in validation exactly once")
    if line_count != manifest["n_selected_instances"] or line_count != len(selected["annotations"]):
        errors.append("Global instance count mismatch")
    checksums = read_json(root / "checksums.json")
    for relative, expected in checksums.items():
        if digest_file(root / relative) != expected:
            errors.append(f"Checksum mismatch: {relative}")
    report = {"status": "PASS" if not errors else "FAIL", "images": len(all_ids),
              "instances": line_count, "folds": manifest["n_folds"], "errors": errors}
    write_json(root / "verification_report.json", report)
    print(json.dumps(report, indent=2))
    if errors:
        raise ValueError("Dataset verification failed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build", help="Choose maximum-count versions and build fixed folds")
    build_parser.add_argument("--source", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--folds", type=int, default=3)
    build_parser.add_argument("--seed", type=int, default=2026)
    build_parser.add_argument("--gap-hours", type=float, default=72)
    build_parser.add_argument("--image-mode", choices=("copy", "hardlink", "symlink"), default="copy")
    build_parser.set_defaults(handler=build)
    crop_parser = sub.add_parser("crops", help="Create single-instance image/GT/synthetic-coarse crops")
    crop_parser.add_argument("--dataset", type=Path, required=True)
    crop_parser.add_argument("--output", type=Path, required=True)
    crop_parser.add_argument("--context", type=float, default=1.5)
    crop_parser.add_argument("--min-side", type=int, default=96)
    crop_parser.add_argument("--resize", type=int, default=0, help="0 keeps native resolution")
    crop_parser.add_argument("--negatives-per-image", type=int, default=2)
    crop_parser.add_argument("--negative-side", type=int, default=128)
    crop_parser.add_argument("--seed", type=int, default=2026)
    crop_parser.add_argument("--limit-images", type=int, default=0, help="Smoke-test subset only")
    crop_parser.set_defaults(handler=make_crops)
    rebase_parser = sub.add_parser("rebase", help="Rebuild YAML and image lists on the destination host")
    rebase_parser.add_argument("--dataset", type=Path, required=True)
    rebase_parser.set_defaults(handler=lambda args: write_runtime_paths(args.dataset))
    verify_parser = sub.add_parser("verify", help="Check selection, folds, paths and immutable JSON checksums")
    verify_parser.add_argument("--dataset", type=Path, required=True)
    verify_parser.set_defaults(handler=verify)
    verify_crop_parser = sub.add_parser("verify-crops", help="Check crop masks, provenance, partitions and COCO geometry")
    verify_crop_parser.add_argument("--dataset", type=Path, required=True)
    verify_crop_parser.add_argument("--crops", type=Path, required=True)
    verify_crop_parser.add_argument("--geometry-samples", type=int, default=64)
    verify_crop_parser.set_defaults(handler=verify_crops)
    epoch_parser = sub.add_parser("epoch", help="Prototype balanced per-epoch alternative-label selection")
    epoch_parser.add_argument("--dataset", type=Path, required=True)
    epoch_parser.add_argument("--fold", type=int, required=True)
    epoch_parser.add_argument("--epoch", type=int, required=True)
    epoch_parser.add_argument("--seed", type=int, default=2026)
    epoch_parser.add_argument("--output", type=Path, required=True)
    epoch_parser.set_defaults(handler=epoch_manifest)
    args = parser.parse_args(argv)
    if args.command == "build" and (args.folds < 2 or args.gap_hours < 0):
        parser.error("Need at least two folds and a nonnegative time gap")
    if args.command == "crops" and (args.context < 1 or args.min_side < 1 or args.resize < 0 or args.negatives_per_image < 0 or args.negative_side < 1 or args.limit_images < 0):
        parser.error("Invalid crop dimensions/counts")
    if args.command == "verify-crops" and args.geometry_samples < 0:
        parser.error("Geometry sample count must be nonnegative")
    args.handler(args)


if __name__ == "__main__":
    main()
