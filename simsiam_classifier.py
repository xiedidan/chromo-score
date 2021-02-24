import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from tqdm import tqdm
from tensorboardX import SummaryWriter
    
def negative_cosine_similarity(p, z):
    z = z.detach()

    p = F.normalize(p)
    z = F.normalize(z)

    return -(p * z).sum(dim=1).mean()

class SiamFeature(nn.Module):
    def __init__(self, resnet, projector_input=2048, projector_hidden=2048):
        super(SiamFeature, self).__init__()
        
        self.resnet = resnet
        self.projector = SiamProjector(projector_input, projector_hidden)
        
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.projector(x)
        
        return x

class SiamProjector(nn.Module):
    def __init__(self, input_size=2048, hidden_size=2048):
        super(SiamProjector, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.projector = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
        )
        
    def forward(self, x):
        x = self.projector(x)
        
        return x

class SiamPredictor(nn.Module):
    def __init__(self, inout_size=2048, hidden_size=512):
        super(SiamPredictor, self).__init__()
        
        self.inout_size = inout_size
        self.hidden_size = hidden_size # hidden_size should be inout_size * 1/4
        
        self.predictor = nn.Sequential(
            nn.Linear(self.inout_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, self.inout_size)
        )
        
    def forward(self, x):
        x = self.predictor(x)
        
        return x
    
def train_epoch(epoch, model, predictor, sampler, loader, criterion, optimizer, scheduler, warmup_epoch, warmup_gamma, writer, max_iter=-1, local_rank=0):
    model.train()
    sampler.set_epoch(epoch)

    running_loss = 0.
    outputs = []
    
    if (warmup_epoch > 0) and (epoch == warmup_epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] /= warmup_gamma
    
    with tqdm(total=len(loader) if max_iter==-1 else max_iter, file=sys.stdout) as pbar:
        for iter_no, samples in enumerate(loader):
            x1, x2 = samples
            x1 = x1.cuda()
            x2 = x2.cuda()

            optimizer.zero_grad()

            z1 = model(x1)
            z2 = model(x2)

            p1 = predictor(z1)
            p2 = predictor(z2)

            loss = criterion(p1, z2) / 2. + criterion(p2, z1) / 2.

            loss.backward()
            optimizer.step()
            
            if local_rank == 0:
                writer.add_scalar(
                    'train/loss',
                    loss.item(),
                    epoch*len(loader)+iter_no
                )
            
            # collect losses
            world_size = float(dist.get_world_size())

            dist.reduce(loss, 0, op=dist.ReduceOp.SUM)
            running_loss += loss.item() / world_size
            
            with torch.no_grad():
                outputs.append(p1.detach())

            pbar.update(1)
            
            if iter_no == max_iter-1:
                break
    
    # collect std
    with torch.no_grad():
        outputs = torch.cat(outputs)
        all_outputs = [outputs.detach().clone() for i in range(dist.get_world_size())]
        
        dist.all_gather(all_outputs, outputs)
        
        all_outputs = torch.cat(all_outputs)
        output_std = torch.mean(torch.std(all_outputs, dim=1))
    
    if (warmup_epoch > 0) and (epoch < warmup_epoch):
        pass
    else:
        scheduler.step()
    
    return running_loss/len(loader), scheduler.get_last_lr(), output_std

def train_simsiam(name, train_id, model, predictor, train_sampler, train_loader, criterion, optimizer, scheduler, epochs=100, warmup_epoch=10, warmup_gamma=0.1, log_path='./logs', metric=None, max_iter=-1, local_rank=0):
    if local_rank == 0:
        writer = SummaryWriter(os.path.join(log_path, '{}_{}'.format(name, train_id)))
    else:
        writer = None
        
    for epoch in range(epochs):
        train_loss, last_lr, output_std = train_epoch(
            epoch,
            model, predictor,
            train_sampler, train_loader,
            criterion,
            optimizer, scheduler,
            warmup_epoch, warmup_gamma,
            writer,
            max_iter,
            local_rank=local_rank
        )
        
        if local_rank == 0:
            writer.add_scalar(
                'epoch/loss',
                train_loss,
                epoch
            )
            
            writer.add_scalar(
                'epoch/std',
                output_std,
                epoch
            )
                
            writer.add_scalar(
                'epoch/lr',
                last_lr,
                epoch
            )

            model_path = os.path.join('./models', '{}_{}'.format(name, train_id))

            if not os.path.exists(model_path):
                os.makedirs(model_path)

            print('epoch: {}, lr: {}, train_loss: {:.6f}, std: {:.6f}'.format(epoch, last_lr, train_loss, output_std))
            
            torch.save(model.state_dict(), os.path.join(model_path, '{:0>3d}.pth'.format(epoch)))
    
    if local_rank == 0:
        writer.close()