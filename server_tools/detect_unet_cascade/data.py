"""Derive detection labels and image-only U-Net crops from the locked split."""
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mu

from dataset_tools.prepare import materialize, stable_seed
from server_tools.evaluate_pq import atomic_json, polygon_rle, read_json, sha256, validate_manifest
from server_tools.prepare_submissions import test_manifest
from .core import pack_roi, rectangle_rles, roi_for_box

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / 'datasets/magfilo_max_3fold'
MANIFEST = DATASET / 'manifest.json'
SOURCE = ROOT / 'MAGFiLO_1.0_Kaggle_2026'
ORIGINAL = SOURCE / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
FOLD_CONTENT_SHA256 = '2ecddcb42bd37753415b764947560b1f47c25016df84605b30d3faeaacf8b5b7'


def fold_data(fold):
    if not MANIFEST.is_file():
        raise FileNotFoundError('Prepare the fixed folds with python -m dataset_tools.prepare build; see README.md')
    manifest = read_json(MANIFEST)
    # The copy/hardlink choice is not part of the fixed split identity.
    fixed = {k: v for k, v in manifest.items() if k != 'image_mode'}
    encoded = json.dumps(fixed, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != FOLD_CONTENT_SHA256:
        raise ValueError('Prepared dataset differs from the released MAGFiLO split')
    validation, training = validate_manifest(manifest, fold)
    if sha256(ORIGINAL) != manifest['source_annotation_sha256']:
        raise ValueError('Original annotations differ from the fixed MAGFiLO release')
    for row in manifest['images']:
        if not (DATASET / 'images/train' / row['file_name']).is_file():
            raise FileNotFoundError(row['file_name'])
    annotations = defaultdict(list)
    for annotation in read_json(ORIGINAL)['annotations']:
        annotations[str(annotation['image_id'])].append(annotation)
    return DATASET, manifest, validation, training, annotations, ORIGINAL


def images_for_split(fold, split):
    dataset, manifest, validation, training, _, _ = fold_data(fold)
    if split == 'test':
        rows = test_manifest(dataset / 'images/test', manifest['n_test_images'])['images']
        return dataset / 'images/test', rows
    return dataset / 'images/train', training if split == 'train' else validation


def selected_masks(row, annotations):
    return [polygon_rle(a, row['height'], row['width']) for a in annotations[row['selected_image_id']]]


def mask_boxes(rles):
    for rle in rles:
        x, y, w, h = map(float, mu.toBbox(rle))
        if w > 0 and h > 0:
            yield (x, y, x+w, y+h)


def detection_labels(row, annotations):
    w, h = row['width'], row['height']
    for x0, y0, x1, y1 in mask_boxes(selected_masks(row, annotations)):
        coords = ((x0+x1)/(2*w), (y0+y1)/(2*h), (x1-x0)/w, (y1-y0)/h)
        yield '0 ' + ' '.join(f'{v:.9f}' for v in coords)


def prepare_detection(fold, out):
    import json

    dataset, _, validation, training, annotations, _ = fold_data(fold)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    counts = {}
    for split, rows in [('train', training), ('val', validation)]:
        labels = out / 'labels' / split
        labels.mkdir(parents=True)
        count = 0
        for row in rows:
            name = row['file_name']
            # Copy rather than link: an upstream image repair must not modify the source.
            materialize(dataset / 'images/train' / name, out / 'images' / split / name, 'copy')
            lines = list(detection_labels(row, annotations))
            count += len(lines)
            (labels / (Path(name).stem+'.txt')).write_text('\n'.join(lines)+('\n' if lines else ''), encoding='utf-8')
        counts[split] = {'images': len(rows), 'boxes': count}
    yaml = out / 'dataset.yaml'
    yaml.write_text(f'path: {json.dumps(out.as_posix(), ensure_ascii=False)}\n'
                    'train: images/train\nval: images/val\nnames:\n  0: filament\n', encoding='utf-8')
    record = {'fold': fold, 'dataset_manifest_sha256': sha256(MANIFEST), 'splits': counts,
              'dataset_yaml_sha256': sha256(yaml)}
    atomic_json(out / 'dataset.json', record)
    return record


@lru_cache(maxsize=4)
def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))


def negative_roi(union, shape, rng):
    h, w = shape
    side = min(h, w, int(rng.integers(96, 385)))
    starts = [(int(rng.integers(w-side+1)), int(rng.integers(h-side+1))) for _ in range(20)]
    boxes = [(x, y, x+side, y+side) for x,y in starts]
    if union is None:
        return boxes[0]
    hits = mu.iou([union], rectangle_rles(boxes, shape), [0]*len(boxes))[0]
    return next((box for box, hit in zip(boxes, hits) if hit == 0), None)


class TrainingCrops:
    """Crop targets are the union of one chosen annotation version, never mixed GT."""

    def __init__(self, rows, annotations, image_dir, config, seed, augment, real_boxes=None):
        self.samples = []
        self.image_dir, self.config, self.seed, self.augment = image_dir, config, seed, augment
        self.epoch = 0
        self.unions = {}
        for row in rows:
            name, shape = row['file_name'], (row['height'], row['width'])
            masks = selected_masks(row, annotations)
            union = mu.merge(masks) if masks else None
            self.unions[name] = (shape, union)
            self.samples.extend((name, box, 'gt') for box in mask_boxes(masks))
            if augment:
                neg = negative_roi(union, shape, np.random.default_rng(stable_seed(seed, name, 'negative')))
                if neg is not None:
                    self.samples.append((name, neg, 'background'))
                if real_boxes:
                    boxes = sorted(real_boxes[name], key=lambda b: -b['score'])[:12]
                    self.samples.extend((name, b['xyxy'], 'detector') for b in boxes)
        if not self.samples:
            raise ValueError('No U-Net crop samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, box, source = self.samples[idx]
        shape, union = self.unions[name]
        rng = np.random.default_rng(stable_seed(self.seed, self.epoch, idx))
        if source == 'gt' and self.augment and rng.random() < .75:
            x0, y0, x1, y1 = box
            w, h = x1-x0, y1-y0
            cx, cy = (x0+x1)/2, (y0+y1)/2
            cx += rng.uniform(-.12, .12)*max(w,h)
            cy += rng.uniform(-.12, .12)*max(w,h)
            factor = rng.uniform(.8, 1.2)
            box = (cx-w*factor/2, cy-h*factor/2, cx+w*factor/2, cy+h*factor/2)
            try:
                roi = roi_for_box(box, shape, self.config)
            except ValueError:
                roi = roi_for_box(self.samples[idx][1], shape, self.config)
        else:
            roi = box if source == 'background' else roi_for_box(box, shape, self.config)
        x0,y0,x1,y1 = roi
        image = read_rgb(self.image_dir / name)[y0:y1,x0:x1]
        target = mu.decode(union)[y0:y1,x0:x1] if union is not None else np.zeros(image.shape[:2], np.uint8)
        if self.augment and rng.random() < .5:
            image, target = image[:, ::-1], target[:, ::-1]
        if self.augment and rng.random() < .5:
            image, target = image[::-1], target[::-1]
        x, y, _ = pack_roi(image, self.config.side, target)
        return x, y
