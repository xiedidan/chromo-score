import os
import sys
import argparse

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn import metrics

from dataset import *
from resnet import *
from classifier import *
from sampler import BalancedStatisticSampler

# args
parser = argparse.ArgumentParser(description='Chromo Scorer Training')
parser.add_argument('--device', default='cuda', help='device (cuda / cpu)')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--root_path', default='./data', help='dataset root path')
parser.add_argument('--lr', default=1e-6, type=float, help='learning rate')
parser.add_argument('--scheduler_step', default=15, type=int, help='epoch to reduce learning rate')
parser.add_argument('--scheduler_gamma', default=0.1, type=float, help='step scheduler gamma')
parser.add_argument('--train_name', default='score', help='train name')
parser.add_argument('--train_id', default='01', help='train id')
parser.add_argument('--epochs', default=20, type=int, help='epoch to train')
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
NUM_WORKERS = 16

# trainer consts
DEVICE = flags.device
LR = flags.lr
EPOCH = flags.epochs
STEP = flags.scheduler_step
GAMMA = flags.scheduler_gamma

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

train_dataset = ChromoDataset(
    ROOT_PATH,
    training=True,
    image_ext='.png',
    transform=train_trans
)
val_dataset = ChromoDataset(
    ROOT_PATH,
    training=False,
    image_ext='.png',
    transform=val_trans
)

train_labels = [train_dataset.labels[i] for i in train_dataset.indices]
train_sampler = BalancedStatisticSampler(
    train_labels,
    NUM_CLASSES,
    BATCH_SIZE
)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# model
device = torch.device(DEVICE)

model = resnet50(pretrained=True, num_classes=NUM_CLASSES)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.StepLR(optimizer, STEP, gamma=GAMMA, last_epoch=-1)
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    train(TRAIN_NAME, TRAIN_ID, model, device, train_loader, val_loader, criterion, optimizer, scheduler, epochs=EPOCH, metric=metrics.accuracy_score)