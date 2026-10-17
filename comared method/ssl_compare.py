import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

from ssl_algorithms import (
    AdaptiveThresholdController, Classifier2, FlexState, ModelEMA, SoftState,
    train_one_epoch_fixmatch, train_one_epoch_infomatch, train_one_epoch_maxmatch,
)
from ssl_randaugment import RandAugment


class SSLClassifier(Classifier2):
    def __init__(self, projection=False):
        super().__init__(num_classes=3)
        self.projector = nn.Sequential(nn.Linear(1024, 1024), nn.BatchNorm1d(1024),
                                       nn.ReLU(), nn.Linear(1024, 128)) if projection else None

    def forward(self, images, return_projection=False):
        logits, features = super().forward(images)
        if return_projection:
            return logits, F.normalize(self.projector(features), dim=1)
        return logits


def image_tensor(image):
    array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div_(127.5).sub_(1.0)


def open_image(path):
    with Image.open(path) as source:
        image = source.convert('RGB')
    width, height = image.size
    scale = 128.0 / max(width, height)
    image = image.resize((int(width * scale), int(height * scale)))
    canvas = Image.new('RGB', (128, 128))
    canvas.paste(image, ((128 - image.width) // 2, (128 - image.height) // 2))
    return canvas


def spatial_view(image):
    if torch.rand(()).item() < 0.5:
        image = ImageOps.mirror(image)
    padded = np.pad(np.asarray(image), ((16, 16), (16, 16), (0, 0)), mode='reflect')
    top, left = torch.randint(0, 33, (2,)).tolist()
    return Image.fromarray(padded[top:top + 128, left:left + 128])


class Images(Dataset):
    def __init__(self, paths, labels=None, views=False, augment=False):
        self.paths = tuple(paths)
        self.labels = None if labels is None else tuple(labels)
        self.views = views
        self.augment = augment
        self.strong = RandAugment(2, 10)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = open_image(self.paths[index])
        if self.views:
            return image_tensor(spatial_view(image)), image_tensor(self.strong(spatial_view(image))), index
        image = spatial_view(image) if self.augment else image
        label = 0 if self.labels is None else self.labels[index]
        return image_tensor(image), label, index, str(self.paths[index])


def class_files(root):
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    if not root.is_dir():
        raise FileNotFoundError(root)
    result = {}
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        result[folder.name] = sorted(path for path in folder.rglob('*') if path.suffix.lower() in extensions)
    return result


def read_split(root, count, seed):
    test = class_files(root / 'test')
    if len(test) != 3 or any(not files for files in test.values()):
        raise ValueError('OpenSARShip test must contain three nonempty class folders')
    if (root / 'train').is_dir():
        pool = class_files(root / 'train')
        existing = {}
    else:
        pseudo = root / 'spseudolabel1'
        if pseudo.exists() and any(path.is_file() for path in pseudo.rglob('*')):
            raise ValueError('Use a complete source pool; spseudolabel1 still contains allocated samples')
        pool = class_files(root / 'unlabel1')
        existing = class_files(root / 'labeldata1') if (root / 'labeldata1').is_dir() else {}
        for name, files in existing.items():
            pool.setdefault(name, []).extend(files)
    if set(pool) != set(test):
        raise ValueError('Train and test class folders must match exactly')
    rng = random.Random(seed)
    labeled_paths, labels, unlabeled_paths, test_paths, targets = [], [], [], [], []
    preserve = all(len(existing.get(name, [])) == count for name in test)
    for label, name in enumerate(test):
        paths = sorted(pool[name])
        if len(paths) <= count:
            raise ValueError(f'{name} needs more than {count} training samples')
        selected = existing[name] if preserve else rng.sample(paths, count)
        selected_set = set(selected)
        labeled_paths.extend(selected)
        labels.extend([label] * len(selected))
        unlabeled_paths.extend(path for path in paths if path not in selected_set)
        test_paths.extend(test[name])
        targets.extend([label] * len(test[name]))
    return labeled_paths, labels, unlabeled_paths, test_paths, targets


class CompleteBatches:
    def __init__(self, size, batch_size):
        if size < 2 or batch_size < 2:
            raise ValueError('Unlabeled pool and batch size must be at least two')
        order = torch.randperm(size).tolist()
        self.batches = [order[start:start + batch_size] for start in range(0, size, batch_size)]
        if len(self.batches[-1]) == 1:
            self.batches[-2].extend(self.batches.pop())

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def train_flex_soft(model, ema, optimizer, labeled, unlabeled, device, algorithm, lambda_u):
    model.train()
    totals = np.zeros(3)
    steps = 0
    for (images, labels, _, _), (weak, strong, indices) in zip(labeled, unlabeled):
        images, labels, weak, strong = images.to(device), labels.to(device), weak.to(device), strong.to(device)
        logits = model(torch.cat((images, weak, strong)))
        logits_l, logits_w, logits_s = torch.split(logits, (len(labels), len(weak), len(strong)))
        supervised = F.cross_entropy(logits_l, labels)
        unsupervised, diagnostics = algorithm.compute(logits_w, logits_s, indices, logits_lb=logits_l)
        loss = supervised + lambda_u * unsupervised
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if ema is not None:
            with torch.no_grad():
                for target, source in zip(ema.model.parameters(), model.parameters()):
                    target.mul_(ema.decay).add_(source.detach(), alpha=1.0 - ema.decay)
                for target, source in zip(ema.model.buffers(), model.buffers()):
                    target.copy_(source)
        totals += (supervised.item(), unsupervised.item(), diagnostics['mask_mean'])
        steps += 1
    return tuple(totals / steps)


@torch.no_grad()
def accuracy(model, loader, device, seed):
    model.eval()
    correct, size = 0, 0
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for images, labels, _, _ in loader:
            predictions = model(images.to(device)).argmax(1).cpu()
            correct += predictions.eq(labels).sum().item()
            size += len(labels)
    return correct / size


def parser_for(method):
    parser = argparse.ArgumentParser(description=f'{method}: existing small-CNN adaptation for three-class OpenSARShip')
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    parser.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parent / 'results')
    parser.add_argument('--datanum', type=int, nargs='+', default=[10])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--tau', type=float, default=0.0 if method == 'maxmatch' else 0.95)
    parser.add_argument('--lambda-u', type=float, default=1.0)
    parser.add_argument('--lambda-c', type=float, default=1.0)
    parser.add_argument('--uratio', type=int, default=1 if method in ('flexmatch', 'softmatch') else 3)
    parser.add_argument('--weight-decay', type=float, default=0.0 if method in ('flexmatch', 'softmatch') else 5e-4)
    parser.add_argument('--ema-decay', type=float, default=0.999)
    return parser


def main(method):
    args = parser_for(method).parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.uratio < 1 or min(args.datanum) < 1:
        raise ValueError('Invalid epoch, batch, ratio, or labeled count')
    device = torch.device(args.device)
    flexible = method in ('flexmatch', 'softmatch')
    for count in args.datanum:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        paths, labels, unlabel, test_paths, targets = read_split(args.data_root, count, args.seed)
        labeled = Images(paths, labels, augment=flexible)
        unlabeled = Images(unlabel, views=flexible)
        test = DataLoader(Images(test_paths, targets), batch_size=args.batch_size)
        output = args.output_root / method / 'opensarship' / str(count) / f'seed_{args.seed}'
        output.mkdir(parents=True, exist_ok=True)
        metrics = output / 'metrics.csv'
        if metrics.exists():
            raise FileExistsError(f'Existing results: {metrics}; choose another --output-root')
        model = SSLClassifier(projection=method == 'infomatch').to(device)
        ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=not flexible)
        scheduler = None if flexible else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        state = FlexState(3, len(unlabel), p_cutoff=args.tau) if method == 'flexmatch' else SoftState(3) if method == 'softmatch' else None
        controller = AdaptiveThresholdController(3, args.tau) if method == 'infomatch' else None
        with metrics.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(('epoch', 'supervised_loss', 'unsupervised_loss', 'mask_mean', 'student_accuracy', 'ema_accuracy'))
            for epoch in range(1, args.epochs + 1):
                batches = CompleteBatches(len(unlabel), args.batch_size * args.uratio)
                uloader = DataLoader(unlabeled, batch_sampler=batches)
                sampler = RandomSampler(labeled, replacement=True, num_samples=len(batches) * args.batch_size)
                lloader = DataLoader(labeled, batch_size=args.batch_size, sampler=sampler)
                if method == 'fixmatch':
                    losses = train_one_epoch_fixmatch(model, ema, optimizer, lloader, uloader, device, args.tau, args.lambda_u, 2, 0.5)
                elif method == 'infomatch':
                    sup, weak, mixed, contrast, mask = train_one_epoch_infomatch(model, ema, optimizer, lloader, uloader, device, controller, args.tau, args.lambda_u, args.lambda_c, 0.5, 1.0, 2, 0.5, 3)
                    losses = (sup, args.lambda_u * (weak + mixed) + args.lambda_c * contrast, mask)
                elif method == 'maxmatch':
                    losses = train_one_epoch_maxmatch(model, ema, optimizer, lloader, uloader, device, args.lambda_u, args.tau, 1.0, 4, 2, 0.5)
                else:
                    losses = train_flex_soft(model, ema, optimizer, lloader, uloader, device, state, args.lambda_u)
                if scheduler is not None:
                    scheduler.step()
                student_acc = accuracy(model, test, device, args.seed + 900000 + epoch)
                ema_acc = accuracy(ema.model, test, device, args.seed + 900000 + epoch) if ema else student_acc
                row = (epoch, *losses, student_acc, ema_acc)
                if not all(math.isfinite(float(value)) for value in row):
                    raise FloatingPointError('Non-finite metric')
                writer.writerow(row)
                handle.flush()
                print(f'{method} epoch={epoch} student={student_acc:.6f} ema={ema_acc:.6f}', flush=True)
