"""Native crop geometry, instance extraction, and official scoring helpers."""
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from PIL import Image

from dataset_tools.prepare import crop_box
from server_tools.evaluate_pq import overlap_statistics, pairwise_rle_iou
from server_tools.mask_postprocess import disjoint_instances, instance_rles

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


@dataclass(frozen=True)
class CropConfig:
    context: float = 1.5
    min_side: int = 96
    side: int = 384

    def __post_init__(self):
        if not math.isfinite(self.context) or self.context < 1 or self.min_side < 1:
            raise ValueError('Context must be >=1 and min-side positive')
        if self.side < 32 or self.side % 32:
            raise ValueError('U-Net input side must be a positive multiple of 32')


@dataclass(frozen=True)
class MaskRules:
    low: float = .5
    peak: float = 0.
    mean: float = 0.
    min_area: int = 0
    closing: int = 0
    duplicate_iou: float = .7

    def __post_init__(self):
        if not all(0 <= p <= 1 for p in (self.low, self.peak, self.mean)):
            raise ValueError('Mask thresholds must lie in [0, 1]')
        if self.min_area < 0 or self.closing not in (0, 3, 5) or not 0 < self.duplicate_iou <= 1:
            raise ValueError('Invalid area, elliptical closing kernel or duplicate IoU')


def roi_for_box(box, shape, config):
    height, width = shape
    values = np.asarray(box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError('Expected finite xyxy coordinates')
    x0, y0, x1, y1 = values.clip([0, 0, 0, 0], [width, height, width, height])
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Box has no positive area inside the image')
    return crop_box((x0, y0, x1-x0, y1-y0), width, height, config.context, config.min_side)


def pack_roi(rgb, side, target=None):
    """RGB only. An optional semantic union target uses the same letterbox."""
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert('RGB')
    width, height = image.size
    scale = side / max(width, height)
    rw, rh = max(1, round(width*scale)), max(1, round(height*scale))
    left, top = (side-rw)//2, (side-rh)//2
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    window = np.s_[top:top+rh, left:left+rw]
    canvas[window] = np.asarray(image.resize((rw, rh), Image.Resampling.BILINEAR))
    tensor = (canvas.transpose(2, 0, 1).astype(np.float32)/255 - MEAN)/STD
    label = None
    if target is not None:
        if target.shape != (height, width):
            raise ValueError('ROI and target geometry mismatch')
        label = np.zeros((1, side, side), dtype=np.float32)
        resized = Image.fromarray((target > 0).astype(np.uint8)).resize((rw, rh), Image.Resampling.NEAREST)
        label[0][window] = np.asarray(resized)
    return tensor, label, (width, height, rw, rh, left, top, side)


def native_probability(probability, geometry):
    width, height, rw, rh, left, top, side = map(int, geometry)
    p = np.asarray(probability, dtype=np.float32)
    if (p.shape != (side, side) or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1
            or min(width, height, rw, rh) < 1 or min(left, top) < 0
            or left+rw > side or top+rh > side):
        raise ValueError('Invalid probability or letterbox geometry')
    return np.asarray(Image.fromarray(p[top:top+rh, left:left+rw]).resize(
        (width, height), Image.Resampling.BILINEAR))


def crop_components(probability, roi, shape, rules, crop_id, detector_score):
    """One ROI can yield zero, one, or multiple instances. Score on original p."""
    import cv2
    from pycocotools import mask as mu

    p = np.asarray(probability, dtype=np.float32)
    x0, y0, x1, y1 = map(int, roi)
    h, w = shape
    if (not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h) or p.shape != (y1-y0, x1-x0)
            or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1 or not 0 <= detector_score <= 1):
        raise ValueError('Invalid native ROI or probability values')
    foreground = (p > rules.low).astype(np.uint8)
    if rules.closing:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rules.closing, rules.closing))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    total = np.bincount(labels.ravel(), weights=p.ravel(), minlength=n)
    peak = np.zeros(n, dtype=np.float32)
    np.maximum.at(peak, labels.ravel(), p.ravel())
    output = []
    for component in range(1, n):
        x, y, w0, h0, area = map(int, stats[component])
        mean = float(total[component] / area)
        if area < rules.min_area or peak[component] < rules.peak or mean < rules.mean:
            continue
        native = np.zeros(shape, dtype=np.uint8, order='F')
        native[y0+y:y0+y+h0, x0+x:x0+x+w0] = labels[y:y+h0, x:x+w0] == component
        rle = mu.encode(native)
        output.append({'segmentation_rle': rle['counts'].decode('ascii'), 'area': area,
                       'confidence': mean, 'peak_probability': float(peak[component]),
                       'detector_confidence': float(detector_score), 'crop_id': crop_id})
    return output


def finalize_instances(file_name, shape, candidates, duplicate_iou):
    """Mask-IoU dedup across crops, then confidence-priority unique ownership."""
    from pycocotools import mask as mu

    order = sorted(range(len(candidates)), key=lambda i: (-candidates[i]['confidence'], -candidates[i]['area'], i))
    ordered = [candidates[i] for i in order]
    rles = instance_rles({'size': shape, 'instances': ordered})
    ious = mu.iou(rles, rles, [0]*len(rles)) if rles else np.zeros((0, 0))
    kept = []
    for i, candidate in enumerate(ordered):
        if not any(ordered[k]['crop_id'] != candidate['crop_id'] and ious[i, k] >= duplicate_iou for k in kept):
            kept.append(i)
    stem = Path(file_name).stem
    entry = {'file_name': file_name, 'size': list(shape), 'instances': [
        {**ordered[i], 'filament_id': f'{stem}_{j}'} for j, i in enumerate(kept, 1)]}
    entry, ownership = disjoint_instances(entry)
    for j, instance in enumerate(entry['instances'], 1):
        instance['filament_id'] = f'{stem}_{j}'
    return entry, {'roi_components': len(candidates), 'duplicates_removed': len(ordered)-len(kept),
                   'empty_removed': len(ownership['removed_empty_ids']),
                   'overlapping_pairs_after': ownership['overlapping_pairs_after']}


def rectangle_rles(rois, shape):
    from pycocotools import mask as mu

    boxes = np.asarray([(x0, y0, x1-x0, y1-y0) for x0,y0,x1,y1 in rois], dtype=np.float64).reshape(-1, 4)
    return mu.frPyObjects(boxes, *shape) if len(boxes) else []


def coverage(gt, rois, shape):
    """For every GT, max fraction of its area inside a single ROI (not IoU)."""
    from pycocotools import mask as mu

    boxes = rectangle_rles(rois, shape)
    if not gt or not boxes:
        return np.zeros(len(gt), dtype=np.float64)
    # COCO crowd IoU uses intersection / detection area; here GT is the
    # detection operand, and the ROI rectangle is the crowd operand.
    ratios = np.asarray(mu.iou(gt, boxes, [1]*len(boxes)), dtype=np.float64)
    return ratios.max(axis=1)


def best_box_iou(gt_boxes, predictions):
    """Best detector-box IoU for each GT box, before context expansion."""
    gt = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    pred = np.asarray(predictions, dtype=np.float64).reshape(-1, 4)
    if not len(gt) or not len(pred):
        return np.zeros(len(gt), dtype=np.float64)
    top_left = np.maximum(gt[:, None, :2], pred[None, :, :2])
    bottom_right = np.minimum(gt[:, None, 2:], pred[None, :, 2:])
    overlap = np.maximum(bottom_right - top_left, 0).prod(axis=2)
    union = (gt[:, 2:] - gt[:, :2]).prod(axis=1)[:, None] + (pred[:, 2:] - pred[:, :2]).prod(axis=1) - overlap
    return np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0).max(axis=1)


def score_version(gt, predictions, roi_boxes, shape):
    matrix = pairwise_rle_iou(gt, predictions)
    counts = overlap_statistics(matrix)
    best_coverage = coverage(gt, roi_boxes, shape)
    covered = best_coverage >= .9
    missed = ~(matrix > np.float32(.5)).any(axis=1)
    counts.update(GT=len(gt), GT_covered_90=int(covered.sum()),
                  FN_no_crop_overlap=int((missed & (best_coverage == 0)).sum()),
                  FN_partial_crop=int((missed & (best_coverage > 0) & ~covered).sum()),
                  FN_covered_90=int((missed & covered).sum()))
    return counts
