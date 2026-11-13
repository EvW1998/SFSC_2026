import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from pycocotools import mask as mu

from server_tools.detect_unet_cascade import __main__ as cli
from server_tools.detect_unet_cascade.core import (
    CropConfig, MaskRules, best_box_iou, coverage, crop_components, finalize_instances,
    native_probability, pack_roi, roi_for_box, score_version,
)
from server_tools.detect_unet_cascade.data import MANIFEST, ORIGINAL, ROOT, TrainingCrops, detection_labels, fold_data
from server_tools.evaluate_pq import read_json, sha256
from server_tools.mask_postprocess import assert_disjoint, instance_rles


def rle(mask):
    return mu.encode(np.asfortranarray(mask, dtype=np.uint8))


class GeometryAndMasks(unittest.TestCase):
    def test_letterbox_restoration_keeps_native_crop_coordinates(self):
        x, y, geometry = pack_roi(np.zeros((12, 24, 3), np.uint8), 32, np.ones((12, 24), np.uint8))
        self.assertEqual(x.shape, (3, 32, 32))
        self.assertEqual(y.shape, (1, 32, 32))
        self.assertTrue(np.all(native_probability(y[0], geometry) == 1))
        self.assertEqual(roi_for_box([-4, 2, 10, 7], (50, 80), CropConfig(1.5, 16, 32)), (0, 0, 16, 16))
        with self.assertRaises(ValueError):
            roi_for_box([0, 0, float('nan'), 3], (50, 80), CropConfig())

    def test_multiple_components_and_three_filters(self):
        p = np.zeros((20, 20), np.float32)
        p[2:5, 2:5] = .9
        p[8:10, 8:10] = .6
        p[15:17, 15:17] = .6
        p[15, 15] = .98
        roi = (5, 4, 25, 24)
        args = (p, roi, (40, 40))
        self.assertEqual(len(crop_components(*args, MaskRules(), 0, .02)), 3)
        self.assertEqual(len(crop_components(*args, MaskRules(min_area=5), 0, .02)), 1)
        self.assertEqual(len(crop_components(*args, MaskRules(peak=.95), 0, .02)), 1)
        self.assertEqual(len(crop_components(*args, MaskRules(mean=.8), 0, .02)), 1)
        got, = crop_components(*args, MaskRules(min_area=5), 0, .02)
        mask = mu.decode({'size': [40, 40], 'counts': got['segmentation_rle'].encode('ascii')})
        self.assertEqual(mu.toBbox(rle(mask)).tolist(), [7, 6, 3, 3])

    def test_closing_is_optional_and_scores_original_probability(self):
        p = np.zeros((25, 25), np.float32)
        p[7:17, 5:10] = .9
        p[7:17, 11:16] = .9
        args = (p, (0, 0, 25, 25), (25, 25))
        self.assertEqual(len(crop_components(*args, MaskRules(), 0, .1)), 2)
        joined, = crop_components(*args, MaskRules(closing=5), 0, .1)
        self.assertGreater(joined['area'], 100)
        self.assertLess(joined['confidence'], .9)
        self.assertFalse(crop_components(*args, MaskRules(closing=5, mean=.89), 0, .1))

    def test_cross_crop_dedup_and_zero_overlap(self):
        a = np.zeros((20, 20), np.uint8)
        b = a.copy()
        a[1:8, 1:8] = 1
        b[5:12, 5:12] = 1
        candidates = [{'segmentation_rle': rle(m)['counts'].decode('ascii'), 'area': int(m.sum()),
                       'confidence': score, 'crop_id': i}
                      for i, (m, score) in enumerate([(a, .9), (a, .85), (b, .8)])]
        entry, audit = finalize_instances('test.jpeg', a.shape, candidates, .7)
        self.assertEqual(audit['duplicates_removed'], 1)
        self.assertEqual(audit['overlapping_pairs_after'], 0)
        self.assertEqual([i['area'] for i in entry['instances']], [49, 40])
        assert_disjoint(instance_rles(entry))

    def test_coverage_and_missed_instance_categories(self):
        left = np.zeros((20, 30), np.uint8)
        middle, right = left.copy(), left.copy()
        left[1:5, 1:5] = 1
        middle[1:5, 11:15] = 1
        right[1:5, 21:25] = 1
        gt = [rle(m) for m in (left, middle, right)]
        rois = [(10, 0, 13, 8), (20, 0, 30, 8)]
        np.testing.assert_allclose(coverage(gt, rois, left.shape), [0., .5, 1.])
        counts = score_version(gt, [], rois, left.shape)
        self.assertEqual([counts[k] for k in ('FN', 'FN_no_crop_overlap', 'FN_partial_crop', 'FN_covered_90')], [3, 1, 1, 1])

    def test_box_recall_is_distinct_from_crop_coverage(self):
        gt_boxes = [(0, 0, 4, 4), (10, 10, 12, 12)]
        np.testing.assert_allclose(best_box_iou(gt_boxes, [(0, 0, 4, 2)]), [.5, 0.])
        np.testing.assert_array_equal(best_box_iou(gt_boxes, []), [0., 0.])


class LocalIntegration(unittest.TestCase):
    @unittest.skipUnless(MANIFEST.is_file() and ORIGINAL.is_file(), 'Requires the locally prepared MAGFiLO dataset')
    def test_real_fold_and_image_only_training_targets(self):
        dataset, _, val, train, annotations, _ = fold_data(0)
        self.assertEqual((len(train), len(val), sum(len(r['versions']) for r in val)), (472, 235, 396))
        self.assertEqual(sum(len(list(detection_labels(row, annotations))) for row in train), 3728)
        dataset = TrainingCrops(train[:2], annotations, dataset/'images/train', CropConfig(), 2026, True)
        self.assertTrue(any(source == 'background' for _,_,source in dataset.samples))
        x, y = dataset[0]
        self.assertEqual(x.shape, (3, 384, 384))
        self.assertEqual(y.shape, (1, 384, 384))
        self.assertTrue(y.any())
        self.assertTrue(np.isfinite(x).all())

    def test_validation_cache_to_pq_and_test_cache_to_csv(self):
        # Tiny synthetic smoke data. This is a functional check, not a benchmark.
        with tempfile.TemporaryDirectory(prefix='cascade_test_', dir=ROOT) as temporary:
            root = Path(temporary)
            shape = (2048, 2048)
            image = '20200101Ch.jpeg'
            row = {'file_name': image, 'height': 2048, 'width': 2048, 'versions': [{'image_id': 'v0'}]}
            Image.new('L', shape).save(root/image)
            manifest = root/'manifest.json'
            manifest.write_text('{}', encoding='utf-8')
            prob = np.zeros((1, 32, 32), dtype=np.float32)
            prob[0, 8:16, 8:16] = .95
            gt = np.zeros(shape, np.uint8)
            gt[48:56, 28:36] = 1
            encoded = rle(gt)
            annotations = {'v0': [{'segmentation': {'size': list(shape), 'counts': encoded['counts'].decode('ascii')}}]}
            for split in ('val', 'test'):
                cache = root/f'cache_{split}'
                cache.mkdir()
                np.savez_compressed(cache/'0001.npz', probabilities=prob)
                index = {'schema': 'magfilo-unet-probabilities-v1', 'fold':0, 'split':split,
                         'dataset_manifest_sha256':sha256(manifest),
                         'candidate_source': 'detector', 'box_conf': .05,
                         'crop_config': {'context':1.5,'min_side':96,'side':32}, 'images': {image: {
                             'size':list(shape), 'file':'0001.npz', 'sha256':sha256(cache/'0001.npz'),
                             'crops':[{'roi':[20,40,52,72], 'geometry':[32,32,32,32,0,0,32], 'score':.1}],
                         }}}
                (cache/'index.json').write_text(json.dumps(index), encoding='utf-8')
                out = root/f'out_{split}'
                with patch.object(cli, 'images_for_split', return_value=(root, [row])), patch.object(
                    cli, 'fold_data', return_value=(None, None, [row], [], annotations, None)
                ), patch.object(cli, 'MANIFEST', manifest):
                    cli.main(['postprocess', '--probabilities', str(cache), '--out', str(out)])
                summary = read_json(out/'summary.json')
                self.assertEqual(summary['postprocess']['overlapping_pairs_after'], 0)
                if split == 'val':
                    self.assertEqual(summary['metrics']['PQ'], 1.)
                    self.assertEqual(summary['coverage_and_misses']['GT_covered_90'], 1)
                else:
                    self.assertEqual(summary['submission']['csv_rows'], 1)
                    self.assertEqual(summary['submission']['rle_decoded_and_reencoded'], 1)

    def test_cannot_recover_boxes_below_cache_floor(self):
        payload = {'inference': {'conf': .01}, 'boxes': {'x': [{'score': .03}, {'score': .1}]}}
        self.assertEqual(cli.filter_boxes(payload, .05), {'x': [{'score': .1}]})
        with self.assertRaises(ValueError):
            cli.filter_boxes(payload, .001)


if __name__ == '__main__':
    unittest.main()
