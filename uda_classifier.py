import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from tensorboardX import SummaryWriter

class KL_Divergence_with_Logits(nn.Module):
    def __init__(self):
        super(KL_Divergence_with_Logits, self).__init__()
        
    def forward(self, p_logits, q_logits):
        p = F.softmax(p_logits, -1)
        log_p = F.log_softmax(p_logits, -1)
        log_q = F.log_softmax(q_logits, -1)
        
        kl = torch.sum(p * (log_p - log_q), -1)
        
        return kl
    
def train_epoch(epoch, model, loader, device, sup_criterion, optimizer, scheduler, writer, unsup_criterion=None, unsup_weight=1.0, unsup_softmax_temp=-1, unsup_confidence_thres=-1, unsup_ratio=0, unsup_copy=1):
    model.train()
    
    running_loss = 0
    running_sup_loss = 0
    running_unsup_loss = 0
    
    with tqdm(total=len(loader), file=sys.stdout) as pbar:
        for iter_no, samples in enumerate(loader):
            if unsup_criterion is None:
                imgs, gts = samples
                
                imgs = imgs.to(device)
                gts = gts.to(device)

                optimizer.zero_grad()

                results = model(imgs)
                losses = sup_criterion(results, gts)

                losses.backward()
                optimizer.step()

                running_loss += losses.item()
                writer.add_scalar(
                    'train/loss',
                    losses.item(),
                    epoch*len(loader)+iter_no
                )
            else:
                imgs, gts = samples
                
                imgs = imgs.to(device)
                gts = gts.to(device)
                
                batch_size = int(imgs.shape[0] / (1 + unsup_ratio + unsup_ratio * unsup_copy))
                grad_index = int(batch_size * (1 + unsup_ratio * unsup_copy))
                
                grad_imgs = imgs[:grad_index]
                no_grad_imgs = imgs[grad_index:]
                
                optimizer.zero_grad()
                
                grad_logits = model(grad_imgs)

                sup_logits = grad_logits[:batch_size]
                sup_losses = sup_criterion(sup_logits, gts)
                
                unsup_aug_logits = grad_logits[batch_size]
                
                with torch.no_grad():
                    unsup_raw_logits = model(no_grad_imgs)
                    
                    # sharpening predictions
                    if unsup_softmax_temp != -1:
                        unsup_raw_logits_gt = unsup_raw_logits / unsup_softmax_temp
                    else:
                        unsup_raw_logits_gt = unsup_raw_logits
                
                unsup_aug_losses = unsup_criterion(unsup_raw_logits_gt, unsup_aug_logits)
                
                # confidence-based masking
                if unsup_confidence_thres != -1:
                    with torch.no_grad():
                        # compute loss mask
                        raw_probs = F.softmax(unsup_raw_logits, -1)
                        larger_probs, _ = torch.max(raw_probs, -1)

                        aug_loss_mask = torch.gt(larger_probs, unsup_confidence_thres)

                    unsup_aug_losses = unsup_aug_losses * aug_loss_mask
                    
                unsup_aug_losses = unsup_aug_losses.mean()
                
                losses = sup_losses + unsup_weight * unsup_aug_losses

                losses.backward()
                optimizer.step()

                running_loss += losses.item()
                running_sup_loss += sup_losses.item()
                running_unsup_loss == unsup_aug_losses.item()
                
                writer.add_scalar(
                    'train/loss',
                    losses.item(),
                    epoch*len(loader)+iter_no
                )
                
                writer.add_scalars(
                    'train/losses',
                    { 
                        'sup': sup_losses.item(),
                        'unsup': unsup_aug_losses.item()
                    },
                    epoch*len(loader)+iter_no
                )

                pbar.update(1)
            
    scheduler.step()
    
    return running_loss/len(loader), running_sup_loss/len(loader), running_unsup_loss/len(loader)

def eval(epoch, model, loader, criterion, device, writer, metric=None):
    model.eval()
    
    running_loss = 0.
    running_metric = 0.
    
    with torch.no_grad():
        with tqdm(total=len(loader), file=sys.stdout) as pbar:
            for iter_no, (imgs, gts) in enumerate(loader):
                imgs = imgs.to(device)
                gts = gts.to(device)
                
                results = model(imgs)
                losses = criterion(results, gts)
                
                running_loss += losses.item()
                
                # be ware torch.max is overloaded
                preds = torch.max(nn.functional.softmax(results, dim=1), 1)[1]
                
                preds = preds.cpu().view(-1).numpy()
                gts = gts.cpu().squeeze().view(-1).numpy()
                
                if metric is not None:
                    m = metric(gts, preds)
                    running_metric += m

                pbar.update(1)
                
        if metric is not None:
            writer.add_scalar(
                'val/metric',
                running_metric/len(loader),
                epoch
            )
                
    return running_loss/len(loader)

def train_uda(name, train_id, model, device, train_loader, val_loader, sup_criterion, optimizer, scheduler, epochs=100, log_path='./logs', metric=None, unsup_criterion=None, unsup_weight=1.0, unsup_softmax_temp=-1, unsup_confidence_thres=-1, unsup_ratio=0, unsup_copy=1):
    writer = SummaryWriter(os.path.join(log_path, '{}_{}'.format(name, train_id)))
    
    for epoch in range(epochs):
        train_loss, sup_loss, unsup_loss = train_epoch(
            epoch,
            model, train_loader, device,
            sup_criterion,
            optimizer, scheduler,
            writer,
            unsup_criterion=unsup_criterion,
            unsup_weight=unsup_weight,
            unsup_softmax_temp=unsup_softmax_temp,
            unsup_confidence_thres=unsup_confidence_thres,
            unsup_ratio=unsup_ratio,
            unsup_copy=unsup_copy
        )
        eval_loss = eval(epoch, model, val_loader, sup_criterion, device, writer, metric)
        
        writer.add_scalars(
            'avg/loss',
            {
                'train': train_loss,
                'val': eval_loss
            },
            epoch
        )
        writer.add_scalars(
            'avg/losses',
            {
                'sup': sup_loss,
                'unsup': unsup_loss
            },
            epoch
        )
        
        model_path = os.path.join('./models', '{}_{}'.format(name, train_id))
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        print('epoch: {}, train_loss: {}={}+{}, eval_loss: {}'.format(epoch, train_loss, sup_loss, unsup_loss, eval_loss))
        torch.save(model.state_dict(), os.path.join(model_path, '{:0>3d}.pth'.format(epoch)))
        
    writer.close()