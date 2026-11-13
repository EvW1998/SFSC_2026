"""Prepare, train, cache and evaluate the local YOLO26m → U-Net cascade."""
import argparse
from collections import Counter
import csv
from dataclasses import asdict
import math
from pathlib import Path

import numpy as np

from dataset_tools.prepare import write_jsonl
from server_tools.evaluate_pq import atomic_json, polygon_rle, pool_statistics, read_json, sha256
from server_tools.mask_postprocess import instance_rles
from server_tools.prepare_submissions import FIELDS, test_manifest, validate_csv
from .core import (
    CropConfig, MaskRules, best_box_iou, coverage, crop_components,
    finalize_instances, native_probability, roi_for_box, score_version,
)
from .data import MANIFEST, fold_data, images_for_split, mask_boxes, prepare_detection, read_rgb, selected_masks
from .models import find_boxes, load_detector, load_unet, segment_image, train_detector, train_unet


def save_summary(path, summary):
    atomic_json(path, summary)
    print(f'Wrote {path.resolve()}', flush=True)


def metadata(fold, split):
    return {'fold': fold, 'split': split, 'dataset_manifest_sha256': sha256(MANIFEST)}


def validate_boxes(boxes, shape):
    for box in boxes:
        if not 0 <= box['score'] <= 1:
            raise ValueError('Nonfinite or out-of-range detector score')
        roi_for_box(box['xyxy'], shape, CropConfig())


def load_boxes(path, fold, split, rows):
    payload = read_json(path)
    if payload.get('schema') != 'magfilo-detector-boxes-v1' or any(
        payload.get(key) != value for key, value in metadata(fold, split).items()
    ):
        raise ValueError('Box cache belongs to a different split, fold or manifest')
    boxes = payload['boxes']
    if set(boxes) != {row['file_name'] for row in rows}:
        raise ValueError('Box cache must cover exactly the requested images')
    if not 0 <= payload['inference']['conf'] <= 1:
        raise ValueError('Invalid box cache confidence floor')
    for row in rows:
        items = boxes[row['file_name']]
        validate_boxes(items, (row['height'], row['width']))
        if any(b['score'] < payload['inference']['conf'] for b in items):
            raise ValueError('Cached candidate is below its inference confidence floor')
    return payload


def filter_boxes(payload, threshold):
    if not payload['inference']['conf'] <= threshold <= 1:
        raise ValueError('Threshold is below cached detector floor or above 1; rebuild boxes if needed')
    return {name: [box for box in boxes if box['score'] >= threshold] for name, boxes in payload['boxes'].items()}


def detect(args):
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    checkpoint = args.weights.resolve()
    receipt = read_json(checkpoint.parents[1] / 'cascade_detector.json')
    if (receipt['status'] != 'PASS' or receipt['fold'] != args.fold
            or receipt['checkpoint_sha256'] != sha256(checkpoint)
            or receipt['dataset_manifest_sha256'] != sha256(MANIFEST)):
        raise ValueError('Detector checkpoint is not verified for this fold')
    model = load_detector(checkpoint)
    image_dir, rows = images_for_split(args.fold, args.split)
    boxes = {}
    for i, row in enumerate(rows, 1):
        name, shape = row['file_name'], (row['height'], row['width'])
        found = find_boxes(model, image_dir / name, shape, args)
        validate_boxes(found, shape)
        boxes[name] = found
        if i % 25 == 0 or i == len(rows):
            print(f'Detector: {i}/{len(rows)} images', flush=True)
    out.mkdir(parents=True, exist_ok=False)
    save_summary(out / 'boxes.json', {'schema': 'magfilo-detector-boxes-v1', **metadata(args.fold,args.split),
                'checkpoint_sha256': receipt['checkpoint_sha256'],
                'inference': {'imgsz': args.imgsz, 'conf': args.conf, 'iou': args.iou, 'max_det': args.max_det,
                              'end2end': bool(getattr(model.predictor.model, 'end2end', False))}, 'boxes': boxes})


def measure_coverage(args, config):
    _, _, rows, _, annotations, _ = fold_data(args.fold)
    cache = load_boxes(args.boxes, args.fold, 'val', rows)
    variants = [(conf, filter_boxes(cache, conf)) for conf in args.thresholds]
    totals = [Counter() for _ in variants]
    for row in rows:
        shape = row['height'], row['width']
        gt = [[polygon_rle(a, *shape) for a in annotations[v['image_id']]] for v in row['versions']]
        gt_boxes = [list(mask_boxes(masks)) for masks in gt]
        for (_, boxes), total in zip(variants, totals):
            candidates = boxes[row['file_name']]
            native_boxes = [b['xyxy'] for b in candidates]
            rois = [roi_for_box(box, shape, config) for box in native_boxes]
            total['rois'] += len(rois)
            total['empty_images'] += not rois
            for masks, targets in zip(gt, gt_boxes):
                ratios = coverage(masks, rois, shape)
                total['box_gt'] += len(targets)
                total['box_matched50'] += int((best_box_iou(targets, native_boxes) >= .5).sum())
                total['gt'] += len(masks)
                total['covered90'] += int((ratios >= .9).sum())
                total['uncovered'] += int((ratios == 0).sum())
    results = [{'conf': conf, **dict(total), 'coverage90': total['covered90']/total['gt'],
                'box_recall50': total['box_matched50']/total['box_gt'] if total['box_gt'] else None,
                'rois_per_image': total['rois']/len(rows)} for (conf,_),total in zip(variants,totals)]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    save_summary(out / 'coverage.json', {**metadata(args.fold,'val'), 'images': len(rows),
                'box_cache_sha256': sha256(args.boxes), 'crop_config': asdict(config), 'results': results})
    for result in results:
        print(result)


def segment(args):
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    image_dir, rows = images_for_split(args.fold, args.split)
    if args.oracle:
        if args.split != 'val':
            raise ValueError('Oracle crop diagnostic is validation-only')
        _, _, _, _, annotations, _ = fold_data(args.fold)
        boxes = {row['file_name']: [{'xyxy': list(box), 'score': 1.} for box in
                 mask_boxes(selected_masks(row, annotations))] for row in rows}
    else:
        boxes = filter_boxes(load_boxes(args.boxes, args.fold, args.split, rows), args.box_conf)
    model, config = load_unet(args.weights, args.fold, args.device)
    out.mkdir(parents=True, exist_ok=False)
    records = {}
    for i, row in enumerate(rows, 1):
        name, shape = row['file_name'], (row['height'], row['width'])
        image = read_rgb(image_dir / name)
        if image.shape[:2] != shape:
            raise ValueError('Image and manifest geometry differ')
        crops, probabilities = segment_image(model, image, boxes[name], config, args.device, args.batch)
        filename = f'{i:04d}.npz'
        np.savez_compressed(out / filename, probabilities=probabilities)
        records[name] = {'file': filename, 'sha256': sha256(out / filename), 'crops': crops, 'size': list(shape)}
        if i % 25 == 0 or i == len(rows):
            print(f'U-Net: {i}/{len(rows)} images', flush=True)
    save_summary(out / 'index.json', {'schema': 'magfilo-unet-probabilities-v1', **metadata(args.fold,args.split),
                'candidate_source': 'oracle_selected_gt_boxes' if args.oracle else 'detector',
                'box_conf': None if args.oracle else args.box_conf,
                'box_cache_sha256': None if args.oracle else sha256(args.boxes),
                'unet_checkpoint_sha256': sha256(args.weights), 'crop_config': asdict(config), 'images': records})


def postprocess(args, rules):
    cache = args.probabilities.resolve()
    index = read_json(cache / 'index.json')
    fold, split = index['fold'], index['split']
    if (index.get('schema') != 'magfilo-unet-probabilities-v1' or split not in ('val','test')
            or index['dataset_manifest_sha256'] != sha256(MANIFEST)
            or index['candidate_source'] not in ('detector','oracle_selected_gt_boxes')
            or (split == 'test' and index['candidate_source'] != 'detector')):
        raise ValueError('Invalid probability cache provenance')
    config = CropConfig(**index['crop_config'])
    image_dir, rows = images_for_split(fold, split)
    if set(index['images']) != {row['file_name'] for row in rows}:
        raise ValueError('Probability cache must cover exactly the requested images')
    annotations = fold_data(fold)[4] if split == 'val' else None
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    predictions, image_scores, scores = [], [], []
    audit = Counter()
    for row in rows:
        name, shape = row['file_name'], (row['height'],row['width'])
        record = index['images'][name]
        filename = record['file']
        if Path(filename).name != filename or not filename.endswith('.npz') or record['size'] != list(shape):
            raise ValueError('Invalid probability file or native shape')
        probability_file = cache / filename
        if sha256(probability_file) != record['sha256']:
            raise ValueError('Probability cache has changed')
        candidates, rois = [], []
        with np.load(probability_file, allow_pickle=False) as array:
            probabilities = array['probabilities']
            if probabilities.shape != (len(record['crops']), config.side, config.side):
                raise ValueError('Probability cache has wrong crop count or shape')
            for i, (p, crop) in enumerate(zip(probabilities, record['crops'])):
                rois.append(crop['roi'])
                native = native_probability(p, crop['geometry'])
                candidates.extend(crop_components(native,crop['roi'],shape,rules,i,crop['score']))
        entry, stats = finalize_instances(name, shape, candidates, rules.duplicate_iou)
        predictions.append(entry)
        audit.update(stats)
        if annotations is not None:
            pred = instance_rles(entry)
            versions = []
            for version in row['versions']:
                gt = [polygon_rle(a, *shape) for a in annotations[version['image_id']]]
                counts = score_version(gt, pred, rois, shape)
                versions.append({'image_id': version['image_id'], **counts})
                scores.append(counts)
            image_scores.append({'file_name': name, 'crops': len(rois), 'versions': versions})
    write_jsonl(out / 'predictions.jsonl', predictions)
    summary = {'status': 'PASS', **metadata(fold,split), 'images': len(rows),
               'candidate_source': index['candidate_source'], 'box_conf': index['box_conf'],
               'crop_config': asdict(config), 'mask_rules': asdict(rules), 'postprocess': dict(audit),
               'probability_index_sha256': sha256(cache/'index.json'),
               'predictions_sha256': sha256(out/'predictions.jsonl')}
    if scores:
        summary['evaluation'] = 'oracle crop diagnostic; not end-to-end' if index['candidate_source'].startswith('oracle') else 'end-to-end; all GT versions'
        summary['metrics'] = pool_statistics(scores)
        summary['annotation_versions'] = len(scores)
        summary['coverage_and_misses'] = {key:sum(r[key] for r in scores) for key in
            ('GT','GT_covered_90','FN_no_crop_overlap','FN_partial_crop','FN_covered_90')}
        write_jsonl(out/'per_image.jsonl', image_scores)
        print(summary['metrics'], summary['coverage_and_misses'])
    if split == 'test':
        csv_path = out/'submission.csv'
        with csv_path.open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, FIELDS, extrasaction='ignore')
            writer.writeheader()
            for entry in predictions:
                writer.writerows(entry['instances'])
        summary['submission'] = validate_csv(csv_path, test_manifest(image_dir,len(rows)),predictions)
    save_summary(out/'summary.json', summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    def command(name, fold=True):
        p = sub.add_parser(name)
        p.add_argument('--out', type=Path, required=True, help='New local output directory')
        if fold:
            p.add_argument('--fold', type=int, choices=(0,1,2), default=0)
        return p

    def crop_args(p):
        p.add_argument('--context', type=float, default=1.5)
        p.add_argument('--min-side', type=int, default=96)
        p.add_argument('--side', type=int, default=384)

    def train_args(p, epochs, batch, device):
        p.add_argument('--epochs', type=int, default=epochs)
        p.add_argument('--batch', type=int, default=batch)
        p.add_argument('--seed', type=int, default=2026)
        p.add_argument('--device', default=device)
        p.add_argument('--run-training', action='store_true', help='Explicitly enable training')

    command('prepare-det')
    p = command('train-det', False)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--imgsz', type=int, default=2048)
    train_args(p,100,2,'0')
    p = command('detect')
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--split', choices=('train','val','test'), default='val')
    p.add_argument('--imgsz', type=int, default=2048)
    p.add_argument('--device', default='0')
    p.add_argument('--conf', type=float, default=.01)
    p.add_argument('--iou', type=float, default=.7)
    p.add_argument('--max-det', type=int, default=300)
    p = command('coverage')
    p.add_argument('--boxes', type=Path, required=True)
    p.add_argument('--thresholds', type=float, nargs='+', default=[.01,.03,.05,.1,.2,.35])
    crop_args(p)
    p = command('train-unet')
    p.add_argument('--train-boxes', type=Path)
    p.add_argument('--lr', type=float, default=1e-4)
    train_args(p,50,8,'cuda:0')
    crop_args(p)
    p = command('segment')
    p.add_argument('--split', choices=('val','test'), default='val')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--boxes', type=Path)
    source.add_argument('--oracle', action='store_true')
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--box-conf', type=float, default=.05)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--device', default='cuda:0')
    p = command('postprocess', False)
    p.add_argument('--probabilities', type=Path, required=True)
    p.add_argument('--low', type=float, default=.5)
    p.add_argument('--peak', type=float, default=0.)
    p.add_argument('--mean', type=float, default=0.)
    p.add_argument('--min-area', type=int, default=0)
    p.add_argument('--closing', type=int, choices=(0,3,5), default=0)
    p.add_argument('--duplicate-iou', type=float, default=.7)
    args = parser.parse_args(argv)

    if hasattr(args,'batch') and args.batch < 1:
        parser.error('batch must be positive')
    if hasattr(args,'epochs') and (not args.run_training or args.epochs < 1):
        parser.error('training needs --run-training and a positive epoch count')
    if args.command == 'prepare-det':
        print(prepare_detection(args.fold,args.out))
    elif args.command == 'train-det':
        if args.imgsz < 32 or args.imgsz % 32:
            parser.error('imgsz must be a positive multiple of 32')
        train_detector(args)
    elif args.command == 'detect':
        if not (0 <= args.conf <= 1 and 0 <= args.iou <= 1 and args.max_det >= 1 and args.imgsz >= 32):
            parser.error('invalid detector thresholds, size or proposal limit')
        detect(args)
    elif args.command in ('coverage','train-unet'):
        config = CropConfig(args.context,args.min_side,args.side)
        if args.command == 'coverage':
            measure_coverage(args,config)
        else:
            if not math.isfinite(args.lr) or args.lr <= 0:
                parser.error('U-Net learning rate must be positive and finite')
            real_boxes = None
            if args.train_boxes:
                _, rows = images_for_split(args.fold,'train')
                real_boxes = load_boxes(args.train_boxes,args.fold,'train',rows)['boxes']
            train_unet(args,config,real_boxes)
    elif args.command == 'segment':
        segment(args)
    else:
        postprocess(args,MaskRules(args.low,args.peak,args.mean,args.min_area,args.closing,args.duplicate_iou))


if __name__ == '__main__':
    main()
