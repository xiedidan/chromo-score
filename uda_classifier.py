import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

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
    
def train_epoch(epoch, model, sampler, loader, sup_criterion, optimizer, scheduler, writer, unsup_criterion=None, unsup_weight=1.0, unsup_softmax_temp=-1, unsup_confidence_thres=-1, unsup_ratio=0, unsup_copy=1, local_rank=0):
    model.train()
    
    running_loss = 0.
    running_sup_loss = 0.
    running_unsup_loss = 0.
    
    sampler.set_epoch(epoch)
    
    with tqdm(total=len(loader), file=sys.stdout) as pbar:
        for iter_no, samples in enumerate(loader):
            if unsup_criterion is None:
                imgs, gts = samples
                
                imgs = imgs.cuda(non_blocking=True)
                gts = gts.cuda(non_blocking=True)

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
                sup_imgs, unsup_raws, unsup_augs, gts = samples

                sup_imgs = sup_imgs.cuda(non_blocking=True)
                unsup_raws = unsup_raws.cuda(non_blocking=True)
                unsup_augs = unsup_augs.cuda(non_blocking=True)
                gts = gts.cuda(non_blocking=True)
                
                batch_size = sup_imgs.shape[0]
                
                grad_imgs = torch.cat([sup_imgs, unsup_augs], 0)
                no_grad_imgs = unsup_raws
                
                with torch.no_grad():
                    unsup_raw_logits = model(no_grad_imgs)

                    # sharpening predictions
                    if unsup_softmax_temp != -1:
                        unsup_raw_logits_gt = unsup_raw_logits / unsup_softmax_temp
                    else:
                        unsup_raw_logits_gt = unsup_raw_logits

                optimizer.zero_grad()

                grad_logits = model(grad_imgs)

                sup_logits = grad_logits[:batch_size]
                sup_losses = sup_criterion(sup_logits, gts)

                unsup_aug_logits = grad_logits[batch_size:]

                unsup_aug_losses = unsup_criterion(unsup_raw_logits_gt, unsup_aug_logits)

                # confidence-based masking
                if unsup_confidence_thres != -1:
                    with torch.no_grad():
                        # compute loss mask
                        raw_probs = F.softmax(unsup_raw_logits, -1)
                        larger_probs, _ = torch.max(raw_probs, -1)

                        aug_loss_mask = torch.gt(larger_probs, unsup_confidence_thres)
                        
                    unsup_aug_losses *= aug_loss_mask

                avg_unsup_aug_losses = unsup_aug_losses.mean()

                losses = sup_losses + unsup_weight * avg_unsup_aug_losses

                losses.backward()
                optimizer.step()
                
                # collect losses
                world_size = float(dist.get_world_size())

                dist.reduce(losses, 0, op=dist.ReduceOp.SUM)
                running_loss += losses.item() / world_size

                dist.reduce(sup_losses, 0, op=dist.ReduceOp.SUM)
                running_sup_loss += sup_losses.item() / world_size

                dist.reduce(avg_unsup_aug_losses, 0, op=dist.ReduceOp.SUM)
                running_unsup_loss += avg_unsup_aug_losses.item() / world_size

                pbar.update(1)
            
    scheduler.step()
    
    return running_loss/len(loader), running_sup_loss/len(loader), running_unsup_loss/len(loader)

def eval(epoch, model, loader, criterion, writer, metric=None):
    model.eval()
    
    running_loss = 0.
    running_metric = 0.
    
    with torch.no_grad():
        with tqdm(total=len(loader), file=sys.stdout) as pbar:
            for iter_no, (imgs, gts) in enumerate(loader):
                imgs = imgs.cuda(non_blocking=True)
                gts = gts.cuda(non_blocking=True)
                
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
                
    return running_loss/len(loader), running_metric/len(loader)

def train_uda(name, train_id, model, train_sampler, train_loader, val_loader, sup_criterion, optimizer, scheduler, epochs=100, log_path='./logs', metric=None, unsup_criterion=None, unsup_weight=1.0, unsup_softmax_temp=-1, unsup_confidence_thres=-1, unsup_ratio=0, unsup_copy=1, local_rank=0):
    if local_rank == 0:
        writer = SummaryWriter(os.path.join(log_path, '{}_{}'.format(name, train_id)))
    else:
        writer = None
        
    for epoch in range(epochs):
        train_loss, sup_loss, unsup_loss = train_epoch(
            epoch,
            model,
            train_sampler, train_loader,
            sup_criterion,
            optimizer, scheduler,
            writer,
            unsup_criterion=unsup_criterion,
            unsup_weight=unsup_weight,
            unsup_softmax_temp=unsup_softmax_temp,
            unsup_confidence_thres=unsup_confidence_thres,
            unsup_ratio=unsup_ratio,
            unsup_copy=unsup_copy,
            local_rank=local_rank
        )
        
        if local_rank == 0:
            eval_loss, eval_metric = eval(epoch, model, val_loader, sup_criterion, writer, metric)

            writer.add_scalars(
                'epoch/loss',
                {
                    'train': train_loss,
                    'val': eval_loss
                },
                epoch
            )
            writer.add_scalars(
                'train/loss',
                {
                    'sup': sup_loss,
                    'unsup': unsup_loss
                },
                epoch
            )
            
            if metric is not None:
                writer.add_scalar(
                    'val/metric',
                    eval_metric,
                    epoch
                )

            model_path = os.path.join('./models', '{}_{}'.format(name, train_id))

            if not os.path.exists(model_path):
                os.makedirs(model_path)

            print('epoch: {}, train_loss: {}={}+{}, eval_loss: {}'.format(epoch, train_loss, sup_loss, unsup_loss, eval_loss))
            torch.save(model.state_dict(), os.path.join(model_path, '{:0>3d}.pth'.format(epoch)))
    
    if local_rank == 0:
        writer.close()