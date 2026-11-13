"""Shared, deterministic mask ownership for validation and Kaggle export.

Only shared pixels are reassigned. No GT, mask merging, area threshold,
component splitting, model changes, or confidence threshold tuning is used.
"""
from __future__ import annotations

import math

import numpy as np

POSTPROCESS_PROTOCOL = {
    "name": "confidence_priority_disjoint_v1",
    "pixel_owner": "highest_object_confidence",
    "equal_confidence_tie_break": "original_prediction_order",
    "empty_instance": "remove",
    "instance_ids": "preserve_original_ids_in_original_order",
    "foreground_union": "preserved_exactly",
    "minimum_area": None,
    "split_components": False,
    "uses_ground_truth": False,
}


def instance_rles(entry):
    return [{"size": list(entry["size"]), "counts": instance["segmentation_rle"].encode("ascii")}
            for instance in entry["instances"]]


def overlapping_pairs(rles):
    """Any shared pixel is invalid, not merely an IoU above a threshold."""
    from pycocotools import mask as mask_util
    if len(rles) < 2:
        return 0
    ious = np.asarray(mask_util.iou(rles, rles, [0] * len(rles)))
    if ious.shape != (len(rles), len(rles)) or not np.isfinite(ious).all():
        raise ValueError("Invalid cross-instance IoU matrix")
    return int(np.count_nonzero(np.triu(ious, 1) > 0))


def assert_disjoint(rles):
    pairs = overlapping_pairs(rles)
    if pairs:
        raise ValueError(f"Overlapping instance masks: {pairs} pair(s) share pixels")


def disjoint_instances(entry):
    """Return a new prediction record plus audit; never mutate the raw cache."""
    from pycocotools import mask as mask_util
    shape = entry["size"]
    if len(shape) != 2 or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError("Expected positive integer height/width")
    instances = entry["instances"]
    ids = [instance["filament_id"] for instance in instances]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate input instance IDs")
    for instance in instances:
        confidence = instance["confidence"]
        if not isinstance(confidence, (float, int)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid object confidence")
        counts = instance["segmentation_rle"]
        if not isinstance(counts, str) or not counts or any(not 48 <= ord(c) <= 111 for c in counts):
            raise ValueError("Invalid compressed RLE counts")
    raw_rles = instance_rles(entry)
    areas = [int(mask_util.area(rle)) for rle in raw_rles]
    if any(int(instance.get("area", area)) != area for instance, area in zip(instances, areas)):
        raise ValueError("Input cache area differs from RLE")
    before = overlapping_pairs(raw_rles)
    audit = {"file_name": entry["file_name"], "input_instances": len(instances),
             "output_instances": 0, "overlapping_pairs_before": before,
             "overlapping_pairs_after": 0, "removed_empty_ids": [], "changed_instances": [],
             "pixels_removed_from_instance_layers": 0, "foreground_union_preserved": True}
    # Non-overlapping records need no raster rewrite; retain their exact RLEs.
    if not before and all(areas):
        result = {**entry, "size": list(shape), "instances": [dict(i) for i in instances]}
        audit["output_instances"] = len(instances)
        return result, audit
    claimed = np.zeros(tuple(shape), dtype=bool, order="F")
    owned = {}
    for index in sorted(range(len(instances)), key=lambda i: (-instances[i]["confidence"], i)):
        instance = instances[index]
        mask = mask_util.decode(raw_rles[index])
        if mask.shape != tuple(shape) or mask.dtype != np.uint8 or int(mask.max()) > 1:
            raise ValueError("Invalid decoded mask")
        mask[claimed] = 0
        rle = mask_util.encode(np.asfortranarray(mask, dtype=np.uint8))
        area = int(mask_util.area(rle))
        lost = areas[index] - area
        if lost < 0:
            raise AssertionError("Postprocessing added pixels")
        audit["pixels_removed_from_instance_layers"] += lost
        if lost:
            audit["changed_instances"].append({"filament_id": ids[index],
                "confidence": instance["confidence"], "original_area": areas[index],
                "area": area, "pixels_removed": lost})
        if not area:
            audit["removed_empty_ids"].append(ids[index])
            continue
        np.logical_or(claimed, mask, out=claimed)
        owned[index] = {**instance, "segmentation_rle": rle["counts"].decode("ascii"), "area": area}
    result = {**entry, "size": list(shape), "instances": [owned[i] for i in sorted(owned)]}
    corrected_rles = instance_rles(result)
    assert_disjoint(corrected_rles)
    if raw_rles:
        raw_union = mask_util.merge(raw_rles)
        if corrected_rles:
            preserved = raw_union["counts"] == mask_util.merge(corrected_rles)["counts"]
        else:
            preserved = int(mask_util.area(raw_union)) == 0
        if not preserved:
            raise AssertionError("Foreground union changed")
    audit["output_instances"] = len(result["instances"])
    return result, audit
