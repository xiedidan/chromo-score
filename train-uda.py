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
from uda_classifier import *
from sampler import *

# args
parser = argparse.ArgumentParser(description='Chromo Scorer Training')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
# parser.add_argument('--device', default='cuda', help='device (cuda / cpu)')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--root_path', default='./data/fire', help='dataset root path')
parser.add_argument('--lr', default=1e-6, type=float, help='learning rate')
parser.add_argument('--scheduler_step', default=15, type=int, help='epoch to reduce learning rate')
parser.add_argument('--scheduler_gamma', default=0.1, type=float, help='step scheduler gamma')
parser.add_argument('--train_name', default='score', help='train name')
parser.add_argument('--train_id', default='01', help='train id')
parser.add_argument('--epochs', default=20, type=int, help='epoch to train')
parser.add_argument('--unsup_weight', default=1., type=float, help='unsupervision loss weight, defaults to 1.0')
parser.add_argument('--unsup_ratio', default=1, type=int, help='unsupervision batch ratio, set 0 to disable unsupervision')
parser.add_argument('--unsup_copy', default=1, type=int, help='unsupervision augment copy for each unsupervision sample')
parser.add_argument('--unsup_confidence_thres', default=0.8, type=float, help='unsupervision confidence threshold, 0.8 for CIFAR and 0.5 for ImageNet')
parser.add_argument('--unsup_softmax_temp', default=0.4, type=float, help='unsupervision softmax temperature, 0.4 for cv tasks')
flags = parser.parse_args()

# consts
TRAIN_NAME = flags.train_name
TRAIN_ID = flags.train_id

# data consts
ROOT_PATH = flags.root_path
NUM_CLASSES = 2 # fg + 1(bg)
INPUT_SIZE = 512
# BATCH_SIZE = 16 * 2
BATCH_SIZE = flags.batch_size
NUM_WORKERS = 32

# trainer consts
# DEVICE = flags.device
LR = flags.lr
EPOCH = flags.epochs
STEP = flags.scheduler_step
GAMMA = flags.scheduler_gamma
UNSUP_WEIGHT = flags.unsup_weight
UNSUP_RATIO = flags.unsup_ratio
UNSUP_COPY = flags.unsup_copy
UNSUP_SOFTMAX_TEMP = flags.unsup_softmax_temp
UNSUP_CONFIDENCE_THRES = flags.unsup_confidence_thres

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
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
val_trans = transforms.Compose([
    transforms.ToPILImage(),
    PadSquare(),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = UdaDataset(
    ROOT_PATH,
    training=True,
    image_ext='.png',
    unsup_ratio=UNSUP_RATIO,
    unsup_copy=UNSUP_COPY,
    sup_transform=train_trans,
    unsup_transform=train_trans
)
val_dataset = UdaDataset(
    ROOT_PATH,
    training=False,
    image_ext='.png',
    sup_transform=val_trans
)

train_labels = [train_dataset.labels[i] for i in train_dataset.indices]
train_sampler = DistributedBalancedStatisticSampler(
    train_labels,
    NUM_CLASSES,
    BATCH_SIZE
)
# val_sampler = DistributedSampler(val_dataset.indices, shuffle=False)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    collate_fn=partial(uda_collate_fn, unsup_ratio=UNSUP_RATIO, unsup_copy=UNSUP_COPY),
    num_workers=NUM_WORKERS,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    # batch_sampler=val_sampler,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# model
# device = torch.device(DEVICE)

model = resnet50(pretrained=True, num_classes=NUM_CLASSES)
model = model.cuda()
model = nn.parallel.DistributedDataParallel(model, device_ids=[flags.local_rank])
    
# model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.StepLR(optimizer, STEP, gamma=GAMMA, last_epoch=-1)
sup_criterion = nn.CrossEntropyLoss()
unsup_criterion = KL_Divergence_with_Logits()

if __name__ == '__main__':
    train_uda(
        TRAIN_NAME, TRAIN_ID,
        model,
        train_sampler, train_loader, val_loader,
        sup_criterion,
        optimizer, scheduler, 
        epochs=EPOCH, 
        metric=metrics.accuracy_score, 
        unsup_criterion=unsup_criterion,
        unsup_weight=UNSUP_WEIGHT,
        unsup_softmax_temp=UNSUP_SOFTMAX_TEMP,
        unsup_confidence_thres=UNSUP_CONFIDENCE_THRES,
        unsup_ratio=UNSUP_RATIO, unsup_copy=UNSUP_COPY,
        local_rank=flags.local_rank
    )