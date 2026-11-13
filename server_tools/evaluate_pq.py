"""Local MAGFiLO scoring: strict float32 IoU > 0.5, all pairs, pooled counts."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def overlap_statistics(iou_matrix):
    """Return the official ALL-pairs counts for one annotation version.

    Both zero-GT and zero-prediction matrices retain their explicit (G, P)
    shape. Conversion to float32 reproduces the official notebook's decoded
    mask dtype, including its strict threshold boundary.
    """
    matrix = np.asarray(iou_matrix, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite two-dimensional IoU matrix")
    if np.any(matrix < 0) or np.any(matrix > 1):
        raise ValueError("IoUs must be in [0, 1]")
    hits = matrix > np.float32(0.5)
    return {
        "TP": int(hits.sum()),
        "FP": int((hits.sum(axis=0) == 0).sum()),
        "FN": int((hits.sum(axis=1) == 0).sum()),
        # Python sum matches the notebook's list accumulation, not float32 sum.
        "sum_IoU": sum(matrix[hits].tolist()),
    }


def metrics_from_counts(counts):
    """SQ/RQ are diagnostic decompositions of this all-pairs PQ definition."""
    tp, fp, fn = (int(counts[name]) for name in ("TP", "FP", "FN"))
    overlap = float(counts["sum_IoU"])
    if min(tp, fp, fn) < 0 or not math.isfinite(overlap) or not 0 <= overlap <= tp:
        raise ValueError("Invalid pooled counts")
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {
        "TP": tp, "FP": fp, "FN": fn, "sum_IoU": overlap,
        "PQ": overlap / denominator if denominator else 0.0,
        "SQ": overlap / tp if tp else 0.0,
        "RQ": tp / denominator if denominator else 0.0,
        "denominator": denominator,
    }


def pool_statistics(rows):
    return metrics_from_counts({
        "TP": sum(row["TP"] for row in rows),
        "FP": sum(row["FP"] for row in rows),
        "FN": sum(row["FN"] for row in rows),
        "sum_IoU": sum(row["sum_IoU"] for row in rows),
    })


def pairwise_rle_iou(gt_rles, predicted_rles):
    """RLE overlap avoids holding 300 full-resolution float mask layers."""
    from pycocotools import mask as mask_util

    if not gt_rles or not predicted_rles:
        return np.zeros((len(gt_rles), len(predicted_rles)), dtype=np.float32)
    # All masks are ordinary instances, never COCO crowd regions. The public
    # notebook likewise uses standard intersection / union for every layer.
    return np.asarray(mask_util.iou(predicted_rles, gt_rles, [0] * len(gt_rles)),
                      dtype=np.float32).T


def polygon_rle(annotation, height, width):
    from pycocotools import mask as mask_util

    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        return mask_util.merge(mask_util.frPyObjects(segmentation, height, width))
    if isinstance(segmentation, dict):
        if segmentation["size"] != [height, width]:
            raise ValueError("GT RLE dimensions do not match the physical image")
        if isinstance(segmentation["counts"], list):
            return mask_util.frPyObjects(segmentation, height, width)
        return {**segmentation, "counts": segmentation["counts"].encode("ascii")
                if isinstance(segmentation["counts"], str) else segmentation["counts"]}
    raise ValueError("Unsupported source segmentation")


def validate_manifest(manifest, fold):
    """Fail closed if physical images or temporal groups cross fixed folds."""
    if manifest.get("schema") != "magfilo-max-instances-3fold-v1":
        raise ValueError("Not the fixed max-instances data manifest")
    if fold not in range(manifest["n_folds"]):
        raise ValueError("Invalid fold")
    images = manifest["images"]
    names = [row["file_name"] for row in images]
    if len(names) != len(set(names)) or len(images) != manifest["n_images"]:
        raise ValueError("Duplicate or missing physical images in manifest")
    groups = defaultdict(set)
    for row in images:
        if Path(row["file_name"]).name != row["file_name"] or "\\" in row["file_name"]:
            raise ValueError("Unsafe physical filename")
        groups[row["temporal_group"]].add(row["fold"])
        if row["fold"] not in range(manifest["n_folds"]):
            raise ValueError("Out-of-range manifest fold")
    if any(len(folds) != 1 for folds in groups.values()):
        raise ValueError("A temporal group crosses training and validation folds")
    validation = [row for row in images if row["fold"] == fold]
    training = [row for row in images if row["fold"] != fold]
    if not validation or not training:
        raise ValueError("Empty validation or training partition")
    return validation, training
