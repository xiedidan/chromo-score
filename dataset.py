import os
import sys
import random
import multiprocessing
from multiprocessing import Pool
from functools import partial

import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

class PadSquare():
    def __init__(self, fill=255):
        self.fill = fill
        
    def __call__(self, img):
        # img is np.array
        
        w, h = img.size
        
        if h > w:
            return transforms.functional.pad(img, (h-w, 0), fill=self.fill)
        else:
            return transforms.functional.pad(img, (0, w-h), fill=self.fill)

# dataset
class ChromoDataset(Dataset):
    def __init__(
        self,
        root_path,
        training=True,
        image_ext='.png',
        test_ratio=0.1,
        transform=None
    ):
        self.root_path = root_path
        self.training = training
        self.transform = transform
        
        # train / val split per image
        filenames = os.listdir(os.path.join(root_path, 'a'))
        img_filenames = []
        
        for filename in filenames:
            if image_ext in filename:
                img_filenames.append(filename)
                
        self.img_filenames = img_filenames
        
        k_filenames = os.listdir(os.path.join(root_path, 'k'))
        k_names = []
        
        for k_filename in k_filenames:
            if image_ext in k_filename:
                k_name = k_filename.split('.K{}'.format(image_ext))[0]
                k_names.append(k_name)
                
        self.labels = []
        
        for img_filename in self.img_filenames:
            img_name = img_filename.split('.A{}'.format(image_ext))[0]
            
            if img_name in k_names:
                self.labels.append(1)
            else:
                self.labels.append(0)
        
        train_indices, val_indices, _, _ = train_test_split(
            list(range(len(img_filenames))),
            self.labels,
            test_size=test_ratio,
            random_state=0,
            stratify=self.labels
        )
        
        if self.training:
            self.indices = train_indices
        else:
            self.indices = val_indices
            
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, index):
        img_file = os.path.join(
            self.root_path,
            'a',
            self.img_filenames[self.indices[index]]
        )
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, self.labels[self.indices[index]]

def uda_collate_fn(batches, unsup_ratio=1, unsup_copy=1):
    imgs = []
    labels = []
    
    sup_imgs = []
    unsup_raws = []
    unsup_augs = []
    
    for batch in batches:
        batch_imgs, batch_label = batch
        
        sup_imgs.append(batch_imgs[0].unsqueeze(0))
        unsup_raws.append(batch_imgs[1:(1+unsup_ratio)])
        unsup_augs.append(batch_imgs[(1+unsup_ratio):])
        
        labels.append(batch_label)
    
    return torch.cat(sup_imgs, dim=0), torch.cat(unsup_raws, dim=0), torch.cat(unsup_augs, dim=0), torch.tensor(labels)
    
class UdaDataset(Dataset):
    def __init__(
        self,
        root_path,
        training=True,
        image_ext='.png',
        test_ratio=0.1,
        unsup_ratio=0,
        unsup_copy=1,
        sup_transform=None,
        unsup_transform=None
    ):
        self.root_path = root_path
        self.training = training
        self.sup_transform = sup_transform
        self.unsup_transform = unsup_transform
        self.unsup_ratio = unsup_ratio
        self.unsup_copy = unsup_copy
        
        # supervision
        # train / val split per image
        self.sup_path = os.path.join(root_path, 'labeled')
        
        pos_filenames = os.listdir(os.path.join(self.sup_path, '1'))
        neg_filenames = os.listdir(os.path.join(self.sup_path, '0'))
        img_filenames = []
        self.labels = []
        
        for pos_filename in pos_filenames:
            if image_ext in pos_filename:
                img_filenames.append(pos_filename)
                self.labels.append(1)
                
        for neg_filename in neg_filenames:
            if image_ext in neg_filename:
                img_filenames.append(neg_filename)
                self.labels.append(0)
                
        self.img_filenames = img_filenames
        
        train_indices, val_indices, _, _ = train_test_split(
            list(range(len(self.img_filenames))),
            self.labels,
            test_size=test_ratio,
            random_state=0,
            stratify=self.labels
        )
        
        if self.training:
            self.indices = train_indices
        else:
            self.indices = val_indices
            
        print('Phase: {}, Supervision samples: {}'.format('Train' if self.training else 'Val', len(self.indices)))
        
        if self.training and (self.unsup_ratio > 0):
            self.unsup_list = []
            
            self.unsup_path = os.path.join(self.root_path, 'unlabeled')
            unsup_filenames = os.listdir(os.path.join(self.unsup_path, 'original'))
            
            for unsup_filename in unsup_filenames:
                if image_ext in unsup_filename:
                    self.unsup_list.append(os.path.join(self.unsup_path, 'original', unsup_filename))
                    
            print('Phase: {}, Unsupervision samples: {}'.format('Train' if self.training else 'Val', len(self.unsup_list)))
            
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, index):
        # supervision
        label = self.labels[self.indices[index]]
        
        img_file = os.path.join(
            self.sup_path,
            '{}'.format(label),
            self.img_filenames[self.indices[index]]
        )
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.sup_transform is not None:
            img = self.sup_transform(img)
        
        if self.training:
            # unsupervision
            unsup_raws = []
            unsup_augs = []

            if self.unsup_ratio > 0:
                unsup_files = random.sample(self.unsup_list, self.unsup_ratio)

                for unsup_file in unsup_files:
                    unsup_img = cv2.imread(unsup_file)
                    unsup_img = cv2.cvtColor(unsup_img, cv2.COLOR_BGR2RGB)

                    unsup_img = self.unsup_transform(unsup_img)
                    unsup_raws.append(unsup_img.clone())

                    for i in range(self.unsup_copy):
                        unsup_aug = self.unsup_transform(unsup_img)
                        unsup_augs.append(unsup_aug.clone())

            imgs = torch.cat([
                img.unsqueeze(0),
                torch.stack(unsup_augs, dim=0),
                torch.stack(unsup_raws, dim=0)
            ], dim=0)

            return imgs, label
        else:
            return img, label
    
def simsiam_collate_fn(batches, unsup_ratio=1, unsup_copy=1):
    x1s = []
    x2s = []
    
    for batch in batches:
        x1, x2 = batch
        
        x1s.append(x1)
        x2s.append(x2)

    return torch.stack(x1s), torch.stack(x2s)

class SimSiamDataset(Dataset):
    def __init__(
        self,
        root_path,
        transform,
        image_ext='.png'
    ):
        self.root_path = root_path
        self.transform = transform
        self.image_ext = image_ext

        filenames = os.listdir(self.root_path)
        img_filenames = []

        for filename in filenames:
            if self.image_ext in filename:
                img_filenames.append(filename)

        self.img_filenames = img_filenames

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, index):
        img_file = os.path.join(
            self.root_path,
            self.img_filenames[index]
        )

        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x1 = self.transform(img.copy())
        x2 = self.transform(img)

        return x1, x2

class SegmentationDataset(Dataset):
    def __init__(
        self,
        root_path,
        training=True,
        image_ext='.png',
        test_ratio=0.1,
        transform=None
    ):
        self.root_path = root_path
        self.training = training
        self.transform = transform
        
        # train / val split per image
        filenames = os.listdir(os.path.join(root_path, 'images'))
        img_filenames = []
        
        for filename in filenames:
            if image_ext in filename:
                img_filenames.append(filename)
                
        self.img_filenames = img_filenames
        
        random.seed(0)
        train_indices = random.sample(
            list(range(len(img_filenames))),
            int(len(img_filenames)*(1.-test_ratio))
        )
        
        if self.training:
            self.indices = train_indices
        else:
            self.indices = []
        
            for i in range(len(img_filenames)):
                if i not in train_indices:
                    self.indices.append(i)
                    
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, index):
        img_file = os.path.join(
            self.root_path,
            'images',
            self.img_filenames[self.indices[index]]
        )

        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask_file = os.path.join(
            self.root_path,
            'masks',
            self.img_filenames[self.indices[index]]
        )
        mask = cv2.imread(mask_file)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        if self.transform is not None:
            img, mask = self.transform(img, mask)
            
        return img, mask

# put png images under root_path/images, and masks under root_path/masks
class SegSliceDataset(Dataset):
    def __init__(
        self,
        root_path,
        slice_size=40,
        read_size=(640,480),
        training=True,
        transform=None,
        image_ext='.png',
        test_ratio=0.1,
        pos_thres=0.1,
        neg_thres=0.01
    ):
        self.root_path = root_path
        self.training = training
        self.transform = transform
        self.slice_size = slice_size
        self.read_size = read_size
        self.pos_thres = pos_thres
        self.neg_thres = neg_thres
        self.num_classes = 2
        
        # train / val split per image
        filenames = os.listdir(os.path.join(root_path, 'images'))
        img_filenames = []
        
        for filename in filenames:
            if image_ext in filename:
                img_filenames.append(filename)
                
        self.img_filenames = img_filenames
        
        random.seed(0)
        train_indices = random.sample(list(range(len(img_filenames))), int(len(img_filenames)*(1.-test_ratio)))
        
        if self.training:
            self.indices = train_indices
        else:
            self.indices = []
        
            for i in range(len(img_filenames)):
                if i not in train_indices:
                    self.indices.append(i)
                    
        # create slices
        self.xs = []
        self.ys = []
        
        with tqdm(total=len(self.indices), file=sys.stdout) as pbar:
            for i, file_index in enumerate(self.indices):
                xs, ys = self._split_img(
                    os.path.join(root_path, 'images', img_filenames[file_index]),
                    os.path.join(root_path, 'masks', img_filenames[file_index])
                )
                self.xs.extend(xs)
                self.ys.extend(ys)
                
                pbar.update(1)
        
    def _split_img(self, img_path, mask_path):        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.read_size)
        
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, self.read_size)

        h, w = mask.shape
        
        imgs = []
        masks = []
        
        for i in range(h//self.slice_size):
            for j in range(w//self.slice_size):
                imgs.append(img[
                    i*self.slice_size:(i+1)*self.slice_size-1,
                    j*self.slice_size:(j+1)*self.slice_size-1,
                    :
                ])
                masks.append(mask[
                    i*self.slice_size:(i+1)*self.slice_size-1,
                    j*self.slice_size:(j+1)*self.slice_size-1
                ])
                # print(i, j, masks[-1].shape)
        
        xs = []
        ys = []
        
        for i in range(len(masks)):
            total_pixels = masks[i].shape[0] * masks[i].shape[1]
            pixel_count = cv2.countNonZero(masks[i])
            ratio = pixel_count / total_pixels
            
            if ratio < self.neg_thres:
                xs.append(imgs[i])
                ys.append(0)
            elif ratio > self.pos_thres:
                xs.append(imgs[i])
                ys.append(1)
                
        return xs, ys
        
    def __len__(self):
        return len(self.ys)
        
    def __getitem__(self, index):
        if self.transform is not None:
            img = self.transform(self.xs[index])
            
        return img, self.ys[index]

class ConfiCaliDataset(Dataset):
    def __init__(
        self,
        root_path,
        image_ext='.png',
        num_classes=2,
        transform=None
    ):
        self.root_path = root_path
        self.image_ext = image_ext
        self.num_classes = num_classes
        self.transform = transform
        
        self.filenames = []
        self.labels = []
        
        for i in range(num_classes):
            cls_filenames = os.listdir(os.path.join(self.root_path, '{}'.format(i)))
            cls_filenames = [f for f in cls_filenames if self.image_ext in f]
            
            for f in cls_filenames:
                self.filenames.append(os.path.join(self.root_path, '{}/{}'.format(i, f)))
                self.labels.append(i)
            
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, index):
        img_file = os.path.join(
            self.root_path,
            self.filenames[index]
        )

        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, self.labels[index]
    