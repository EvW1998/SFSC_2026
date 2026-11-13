"""Validate local test predictions and the compressed COCO RLE CSV."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from server_tools.evaluate_pq import sha256
from server_tools.mask_postprocess import assert_disjoint

FIELDS = ["filament_id", "segmentation_rle"]


def test_manifest(test_dir, expected_count):
    files = sorted(Path(test_dir).glob("*.jpeg"))
    if len(files) != expected_count or len({p.stem for p in files}) != expected_count:
        raise ValueError("Unexpected or duplicate test image IDs")
    rows = []
    for path in files:
        if path.is_symlink() or not path.stem.isalnum():
            raise ValueError("Unsafe or noncanonical test image name")
        with Image.open(path) as image:
            if image.size != (2048, 2048):
                raise ValueError("Test image must be original 2048x2048")
        rows.append({"file_name": path.name, "image_id": path.stem,
                     "height": 2048, "width": 2048, "sha256": sha256(path)})
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema": "magfilo-test-manifest-v1", "image_count": len(rows),
            "images": rows, "content_sha256": digest}


def validate_counts(counts, shape):
    from pycocotools import mask as mask_util
    if (not isinstance(counts, str) or not counts
            or any(not 48 <= ord(c) <= 111 for c in counts)):
        raise ValueError("CSV must contain raw compressed RLE counts, without quotations")
    rle = {"size": list(shape), "counts": counts.encode("ascii")}
    decoded = mask_util.decode(rle)
    if (decoded.shape != tuple(shape) or decoded.dtype != np.uint8
            or int(decoded.max()) > 1 or not decoded.any()):
        raise ValueError("Invalid dimensions, values or empty instance mask")
    canonical = mask_util.encode(np.asfortranarray(decoded, dtype=np.uint8))["counts"].decode("ascii")
    if canonical != counts:
        raise ValueError("RLE round-trip did not preserve counts")
    return int(decoded.sum())


def validate_csv(path, manifest, predictions):
    images = {r["image_id"]: r for r in manifest["images"]}
    if len(predictions) != len(images) or {p["file_name"] for p in predictions} != {
            r["file_name"] for r in images.values()}:
        raise ValueError("Inference did not cover each test image exactly once")
    if len({p["file_name"] for p in predictions}) != len(predictions):
        raise ValueError("Duplicate image prediction records")
    expected = {i["filament_id"]: i["segmentation_rle"] for p in predictions for i in p["instances"]}
    if len(expected) != sum(len(p["instances"]) for p in predictions):
        raise ValueError("Duplicate IDs in prediction cache")
    counts, ids, image_rles = Counter(), set(), {name: [] for name in images}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError("Expected exactly filament_id,segmentation_rle columns")
        for row in reader:
            if set(row) != set(FIELDS):
                raise ValueError("Wrong CSV column count")
            identifier = row["filament_id"]
            image_id, suffix = identifier.rsplit("_", 1)
            if image_id not in images or not suffix.isdigit() or int(suffix) < 1 or identifier in ids:
                raise ValueError("Invalid/duplicate filament ID or wrong test image prefix")
            if expected.get(identifier) != row["segmentation_rle"]:
                raise ValueError("CSV differs from the inference cache")
            validate_counts(row["segmentation_rle"], (2048, 2048))
            image_rles[image_id].append({"size": [2048, 2048],
                                       "counts": row["segmentation_rle"].encode("ascii")})
            ids.add(identifier)
            counts[image_id] += 1
    if ids != set(expected):
        raise ValueError("CSV omitted prediction instances")
    for rles in image_rles.values():
        assert_disjoint(rles)
    zero = [name for name in sorted(images) if counts[name] == 0]
    return {"status": "PASS", "csv_rows": len(ids), "inferred_images": len(images),
            "images_with_predictions": len(images) - len(zero), "zero_prediction_images": zero,
            "rle_decoded_and_reencoded": len(ids), "csv_sha256": sha256(path),
            "overlapping_images": 0, "overlapping_pairs": 0,
            "checks": ["exact_two_column_header", "unique_ids", "test_image_prefixes",
                       "all_test_images_inferred_once", "csv_matches_prediction_cache",
                       "nonempty_binary_2048x2048_masks", "compressed_RLE_roundtrip",
                       "same_image_instances_share_zero_pixels"]}
