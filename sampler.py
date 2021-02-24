import math

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.sampler import BatchSampler

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
    
# first sample class, then item with replacement
# num_classes = fg_num_classes + 1
class BalancedStatisticSampler(BatchSampler):
    def __init__(self, labels, num_classes, batch_size):
        self.labels = np.array(labels)
        self.n_classes = num_classes
        self.batch_size = batch_size
        
        self.n_dataset = len(self.labels)
        self.labels_set = list(set(self.labels))
        self.indices_table = {
            label: np.where(self.labels == label)[0] for label in self.labels_set
        }
        
    def __iter__(self):
        self.count = 0
        
        while self.count + self.batch_size < self.n_dataset:
            indices = []
            
            for i in range(self.batch_size):
                curr_cls = np.random.randint(0, self.n_classes)
                index = np.random.randint(0, len(self.indices_table[curr_cls]))
                indices.append(self.indices_table[curr_cls][index])
            
            yield indices
            
            self.count += self.batch_size
        
    def __len__(self):
        return self.n_dataset // self.batch_size
    
class DistributedBalancedStatisticSampler(BatchSampler):
    def __init__(self, labels, num_classes, batch_size, num_replicas=None, rank=None, seed=0):
        self.labels = np.array(labels)
        self.n_classes = num_classes
        self.batch_size = batch_size
        
        self.labels_set = list(set(self.labels))
        self.indices_table = {
            label: np.where(self.labels == label)[0] for label in self.labels_set
        }
        
        # distributed related
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
            
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
            
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.nums_samples = [int(math.ceil(len(self.indices_table[label]) * 1.0 / self.num_replicas)) for label in self.labels_set]
        self.total_sizes = [self.nums_samples[i] * self.num_replicas for i in range(len(self.nums_samples))]
        self.seed = seed
        self.n_dataset = np.sum(self.nums_samples)
    
    def __iter__(self):
        # create random epoch indices table
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        self.epoch_indice_table = {label: [] for label in self.labels_set}

        for label in self.labels_set:
            indices_of_indices = torch.randperm(len(self.indices_table[label]), generator=g).tolist()
            self.epoch_indice_table[label] = [self.indices_table[label][i] for i in indices_of_indices]

        # add extra indices
        for i, label in enumerate(self.labels_set):
            indice_len = len(self.epoch_indice_table[label])
            self.epoch_indice_table[label] += self.epoch_indice_table[label][:self.total_sizes[i]-indice_len]

            assert len(self.epoch_indice_table[label]) == self.total_sizes[i]

        # create local indice table
        self.my_epoch_indice_table = {label: [] for label in self.labels_set}

        for i, label in enumerate(self.labels_set):
            self.my_epoch_indice_table[label] = self.epoch_indice_table[label][self.rank:self.total_sizes[i]:self.num_replicas]

            assert len(self.my_epoch_indice_table[label]) == self.nums_samples[i]

        # create iter
        self.count = 0

        while self.count + self.batch_size < self.n_dataset:
            indices = []

            for i in range(self.batch_size):
                curr_cls = torch.randint(0, self.n_classes, (1,), generator=g).item()
                index = torch.randint(0, len(self.my_epoch_indice_table[curr_cls]), (1,), generator=g).item()
                indices.append(self.indices_table[curr_cls][index])

            yield indices

            self.count += self.batch_size
    
    def __len__(self):
        return self.n_dataset // self.batch_size
    
    def set_epoch(self, epoch):
        self.epoch = epoch
