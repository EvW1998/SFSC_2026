"""Explicit local training/inference entry points; no download or upload."""
from dataclasses import asdict
import os
from pathlib import Path
import random

import numpy as np

from server_tools.evaluate_pq import atomic_json, read_json, sha256
from .core import CropConfig, pack_roi, roi_for_box
from .data import MANIFEST, ROOT, TrainingCrops, fold_data, read_rgb

DEFAULT_ASSETS = ROOT / 'model_assets'
ENCODER_WEIGHTS = DEFAULT_ASSETS / 'weights/resnet18_imagenet/model.safetensors'
ENCODER_SHA256 = 'b1d1d36f298c29c07afa9d5be76d888b01d174824177026613bbd64a26612d19'


def create_unet(*, initialize_encoder=False):
    import segmentation_models_pytorch as smp

    model = smp.Unet(encoder_name='resnet18', encoder_weights=None,
                     in_channels=3, classes=1, activation=None)
    if initialize_encoder:
        from safetensors.torch import load_file

        if not ENCODER_WEIGHTS.is_file() or sha256(ENCODER_WEIGHTS) != ENCODER_SHA256:
            raise ValueError(f'Expected the local ResNet18 ImageNet encoder file documented in README: {ENCODER_WEIGHTS}')
        model.encoder.load_state_dict(load_file(str(ENCODER_WEIGHTS), device='cpu'), strict=True)
    return model


def require_weight(path):
    path = Path(path).resolve()
    if not path.is_file() or path.suffix != '.pt':
        raise FileNotFoundError(f'Explicit local .pt checkpoint required: {path}')
    return path


def load_detector(path):
    path = require_weight(path)
    os.environ.update(YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true')
    os.environ.setdefault('YOLO_CONFIG_DIR', str(DEFAULT_ASSETS / 'runtime/detect_unet_cascade'))
    from ultralytics import YOLO

    model = YOLO(str(path))
    if model.task != 'detect' or model.model.yaml.get('scale') != 'm':
        raise ValueError('Expected a YOLO26m detection checkpoint (not a segmentation checkpoint)')
    return model


def train_detector(args):
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    yaml = args.dataset.resolve()
    record = read_json(yaml.with_name('dataset.json'))
    if record['dataset_manifest_sha256'] != sha256(MANIFEST) or record['dataset_yaml_sha256'] != sha256(yaml):
        raise ValueError('Detection dataset was not built from the current locked fold')
    model = load_detector(args.weights)
    model.train(data=str(yaml), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
                device=args.device, workers=0, seed=args.seed, amp=False, cache=False,
                patience=0, project=str(out.parent), name=out.name, exist_ok=False)
    checkpoint = Path(model.trainer.last).resolve()
    if not checkpoint.is_file() or int(model.trainer.epoch)+1 != args.epochs:
        raise RuntimeError('Detector did not finish the requested epoch budget')
    atomic_json(checkpoint.parents[1] / 'cascade_detector.json', {
        'status': 'PASS', 'fold': record['fold'], 'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256(checkpoint), 'epochs': args.epochs,
        'dataset_manifest_sha256': sha256(MANIFEST), 'selection': 'last; select inference threshold by crop coverage',
    })
    print(f'Detector ready: {checkpoint}')


def find_boxes(model, image_path, shape, args):
    result, = model.predict(source=str(image_path), imgsz=args.imgsz, device=args.device,
                            conf=args.conf, iou=args.iou, max_det=args.max_det, classes=[0],
                            augment=False, save=False, verbose=False, quantize=None)
    if tuple(result.orig_shape) != tuple(shape):
        raise ValueError('Detector coordinates are not in the original image')
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    return [{'xyxy': [float(v) for v in box], 'score': float(score)} for box,score in zip(xyxy,scores)]


def objective(logits, target):
    import torch
    import torch.nn.functional as F

    p = torch.sigmoid(logits.float())
    inter = (p*target).sum(dim=(2,3))
    dice = (2*inter+1e-6)/(p.sum(dim=(2,3))+target.sum(dim=(2,3))+1e-6)
    return .5*F.binary_cross_entropy_with_logits(logits.float(), target) + .5*(1-dice.mean())


def train_unet(args, config, real_boxes=None):
    import torch
    from torch.utils.data import DataLoader

    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    dataset, _, val, train, annotations, _ = fold_data(args.fold)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    model = create_unet(initialize_encoder=True).to(device).float()
    images = dataset / 'images/train'
    training = TrainingCrops(train, annotations, images, config, args.seed, True, real_boxes)
    validation = TrainingCrops(val, annotations, images, config, args.seed, False)
    loader = DataLoader(training, batch_size=args.batch, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(validation, batch_size=args.batch, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    out.mkdir(parents=True, exist_ok=False)
    history = []
    for epoch in range(args.epochs):
        model.train()
        training.epoch = epoch
        loss_sum = 0.
        for x,y in loader:
            x,y = x.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite U-Net training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10., error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item()*len(x)
        scheduler.step()
        model.eval()
        val_sum, intersection, foreground = 0.,0.,0.
        with torch.inference_mode():
            for x,y in val_loader:
                x,y = x.to(device),y.to(device)
                logits = model(x)
                loss = objective(logits, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite U-Net validation loss')
                mask = logits > 0
                intersection += (mask*y).sum().item()
                foreground += mask.sum().item()+y.sum().item()
                val_sum += loss.item()*len(x)
        record = {'epoch': epoch+1, 'train_loss': loss_sum/len(training), 'oracle_crop_loss': val_sum/len(validation),
                  'oracle_crop_dice': 2*intersection/foreground if foreground else 0.}
        history.append(record)
        print(record, flush=True)
        atomic_json(out / 'history.json', history)
    checkpoint = out / 'last.pt'
    torch.save({'schema': 'magfilo-image-only-unet-v1', 'fold': args.fold, 'crop_config': asdict(config),
                'dataset_manifest_sha256': sha256(MANIFEST), 'epochs': args.epochs,
                'state_dict': {k:v.detach().cpu() for k,v in model.state_dict().items()}}, checkpoint)
    atomic_json(out / 'summary.json', {'status': 'PASS', 'checkpoint_sha256': sha256(checkpoint),
                'fold': args.fold, 'train_crops': len(training), 'oracle_val_crops': len(validation),
                'train_boxes_sha256': sha256(args.train_boxes) if args.train_boxes else None,
                'checkpoint_selection': 'fixed final epoch; oracle crop Dice is not PQ', 'history': history})
    print(f'U-Net ready: {checkpoint}')


def load_unet(path, fold, device):
    import torch

    path = require_weight(path)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload.get('schema') != 'magfilo-image-only-unet-v1' or payload['fold'] != fold
            or payload['dataset_manifest_sha256'] != sha256(MANIFEST)):
        raise ValueError('Expected image-only U-Net weights for the same fold and manifest')
    config = CropConfig(**payload['crop_config'])
    model = create_unet()
    model.load_state_dict(payload['state_dict'], strict=True)
    return model.to(device).float().eval(), config


def segment_image(model, image, boxes, config, device, batch):
    import torch

    crops, probabilities = [], []
    shape = image.shape[:2]
    for i in range(0, len(boxes), batch):
        packed = []
        for box in boxes[i:i+batch]:
            roi = roi_for_box(box['xyxy'], shape, config)
            x0,y0,x1,y1 = roi
            x, _, geometry = pack_roi(image[y0:y1,x0:x1], config.side)
            crops.append({'roi': list(roi), 'geometry': list(geometry), 'score': box['score']})
            packed.append(x)
        with torch.inference_mode():
            logits = model(torch.from_numpy(np.stack(packed)).to(device))
            if logits.shape != (len(packed),1,config.side,config.side) or not torch.isfinite(logits).all():
                raise ValueError('Invalid U-Net output')
            probabilities.extend(logits.float().sigmoid()[:,0].cpu().numpy())
    return crops, np.asarray(probabilities, dtype=np.float32).reshape(-1, config.side, config.side)
