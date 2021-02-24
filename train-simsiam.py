import os
import sys
import argparse
from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms
from sklearn import metrics

from dataset import *
from resnet import *
from simsiam_classifier import *

# args
parser = argparse.ArgumentParser(description='Chromo Scorer Training')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
# parser.add_argument('--device', default='cuda', help='device (cuda / cpu)')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--max_iter', default=-1, type=int, help='max iter for quick testing')
parser.add_argument('--root_path', default='./data/fire', help='dataset root path')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--scheduler_step', default=15, type=int, help='epoch to reduce learning rate')
parser.add_argument('--scheduler_gamma', default=0.1, type=float, help='step scheduler gamma')
parser.add_argument('--train_name', default='simsiam', help='train name')
parser.add_argument('--train_id', default='01', help='train id')
parser.add_argument('--epoch', default=20, type=int, help='epoch to train')
parser.add_argument('--warmup_epoch', default=10, type=int, help='epoch to warmup')
parser.add_argument('--warmup_gamma', default=0.1, type=float, help='warmup learning rate ratio')
flags = parser.parse_args()

if flags.local_rank == 0:
    print('\nAll Flags:\n')
    for k, v in sorted(vars(flags).items()):
        print('{}: {}'.format(k, v))
    print('\n')
    
# consts
TRAIN_NAME = flags.train_name
TRAIN_ID = flags.train_id

# data consts
ROOT_PATH = flags.root_path
NUM_CLASSES = 2 # fg + 1(bg)
INPUT_SIZE = 512
# BATCH_SIZE = 16 * 2
BATCH_SIZE = flags.batch_size
MAX_ITER = flags.max_iter
NUM_WORKERS = 8

# trainer consts
# DEVICE = flags.device
LR = flags.lr * BATCH_SIZE / 256.
EPOCH = flags.epoch
WARMUP_EPOCH = flags.warmup_epoch
WARMUP_GAMMA= flags.warmup_gamma
STEP = flags.scheduler_step - flags.warmup_epoch
GAMMA = flags.scheduler_gamma

# distribution
dist.init_process_group(backend='nccl')
torch.cuda.set_device(flags.local_rank)

# data
train_trans = transforms.Compose([
    transforms.ToPILImage(),
    PadSquare(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = SimSiamDataset(
    ROOT_PATH,
    transform=train_trans,
    image_ext='.png'
)

train_sampler = DistributedSampler(
    train_dataset
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE//4,
    sampler=train_sampler,
    drop_last=True,
    collate_fn=simsiam_collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# model
model = resnet50(pretrained=True, num_classes=NUM_CLASSES)
feature = SiamFeature(model)
feature = nn.SyncBatchNorm.convert_sync_batchnorm(feature).cuda()
feature = nn.parallel.DistributedDataParallel(feature, device_ids=[flags.local_rank], find_unused_parameters=True)

predictor = SiamPredictor()
predictor = nn.SyncBatchNorm.convert_sync_batchnorm(predictor).cuda()
predictor = nn.parallel.DistributedDataParallel(predictor, device_ids=[flags.local_rank], find_unused_parameters=True)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LR if WARMUP_EPOCH <= 0 else LR*WARMUP_GAMMA,
    momentum=0.9, weight_decay=0.0001
)
scheduler = lr_scheduler.StepLR(optimizer, STEP, gamma=GAMMA, last_epoch=-1)

if __name__ == '__main__':
    train_simsiam(
        TRAIN_NAME, TRAIN_ID,
        feature, predictor,
        train_sampler, train_loader,
        negative_cosine_similarity,
        optimizer, scheduler, 
        epochs=EPOCH, 
        warmup_epoch=WARMUP_EPOCH, warmup_gamma=WARMUP_GAMMA,
        metric=metrics.accuracy_score, 
        max_iter=MAX_ITER,
        local_rank=flags.local_rank
    )
    