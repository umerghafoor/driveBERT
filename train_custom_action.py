"""
Custom Action Recognition Training Script
Extended from train_action.py to support custom datasets
"""

import os
import numpy as np
import time
import sys
import argparse
import errno
from collections import OrderedDict
import tensorboardX
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.model.loss import *
from lib.data.dataset_action import NTURGBD
from lib.data.dataset_custom import CustomActionDataset, VideoActionDataset
from lib.model.model_action import ActionNet

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/action/MB_train_CUSTOM.yaml", 
                       help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', 
                       help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', 
                       help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', 
                       help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', 
                       help='checkpoint to evaluate (file name)')
    parser.add_argument('-freq', '--print_freq', default=100, type=int)
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', 
                       help='checkpoint to finetune (file name)')
    
    # Custom dataset specific arguments
    parser.add_argument('--dataset_type', default='pickle', choices=['pickle', 'video'],
                       help='Dataset type: pickle (preprocessed) or video (extract on-the-fly)')
    parser.add_argument('--video_dir', default=None, type=str,
                       help='Directory containing video files (for video dataset type)')
    parser.add_argument('--annotation_file', default=None, type=str,
                       help='Annotation file path (for video dataset type)')
    parser.add_argument('--keypoint_extractor', default='mediapipe', 
                       choices=['mediapipe', 'openpose', 'alphapose'],
                       help='Keypoint extraction method (for video dataset type)')
    
    opts = parser.parse_args()
    return opts

def create_dataset(args, opts, data_split, is_train=True):
    """
    Create dataset based on dataset type
    """
    if opts.dataset_type == 'pickle':
        # Use preprocessed pickle dataset
        data_path = f'data/action/{args.dataset}.pkl'
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        dataset = CustomActionDataset(
            data_path=data_path,
            data_split=data_split,
            n_frames=args.clip_len,
            random_move=args.random_move and is_train,
            scale_range=args.scale_range_train if is_train else args.scale_range_test
        )
        
    elif opts.dataset_type == 'video':
        # Use video dataset with on-the-fly keypoint extraction
        if not opts.video_dir or not opts.annotation_file:
            raise ValueError("video_dir and annotation_file must be provided for video dataset type")
        
        dataset = VideoActionDataset(
            video_dir=opts.video_dir,
            annotation_file=opts.annotation_file,
            keypoint_extractor=opts.keypoint_extractor,
            data_split=data_split,
            n_frames=args.clip_len,
            random_move=args.random_move and is_train,
            scale_range=args.scale_range_train if is_train else args.scale_range_test
        )
    
    else:
        raise ValueError(f"Unknown dataset type: {opts.dataset_type}")
    
    return dataset

def validate(test_loader, model, criterion, opts):
    """Validation function"""
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        end = time.time()
        for idx, (batch_input, batch_gt) in tqdm(enumerate(test_loader)):
            batch_size = len(batch_input)    
            if torch.cuda.is_available():
                batch_gt = batch_gt.cuda()
                batch_input = batch_input.cuda()
            
            output = model(batch_input)    # (N, num_classes)
            loss = criterion(output, batch_gt)

            # update metric
            losses.update(loss.item(), batch_size)
            
            # Calculate accuracy (top5 only if we have more than 5 classes)
            topk = (1, min(5, output.size(1)))
            acc1, acc5 = accuracy(output, batch_gt, topk=topk)
            top1.update(acc1[0], batch_size)
            if output.size(1) >= 5:
                top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx+1) % opts.print_freq == 0:
                if output.size(1) >= 5:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                           idx, len(test_loader), batch_time=batch_time,
                           loss=losses, top1=top1, top5=top5))
                else:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                           idx, len(test_loader), batch_time=batch_time,
                           loss=losses, top1=top1))
    
    return losses.avg, top1.avg, top5.avg

def train_with_config(args, opts):
    print(args)
    
    # Create checkpoint directory
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    
    # Setup tensorboard logging
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))
    
    # Load backbone model
    model_backbone = load_backbone(args)
    
    # Load pretrained weights if finetuning
    if args.finetune:
        if opts.resume or opts.evaluate:
            pass
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            print('Loading backbone', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model_pos']
            model_backbone = load_pretrained_weights(model_backbone, checkpoint)
    
    if args.partial_train:
        model_backbone = partial_train_layers(model_backbone, args.partial_train)
    
    # Create action recognition model
    model = ActionNet(
        backbone=model_backbone, 
        dim_rep=args.dim_rep, 
        num_classes=args.action_classes, 
        dropout_ratio=args.dropout_ratio, 
        version=args.model_version, 
        hidden_dim=args.hidden_dim, 
        num_joints=args.num_joints
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
    
    best_acc = 0
    model_params = sum(p.numel() for p in model.parameters())
    print('INFO: Trainable parameter count:', model_params)
    
    # Setup data loaders
    print('Loading dataset...')
    trainloader_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }
    testloader_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }
    
    # Create datasets
    train_dataset = create_dataset(args, opts, data_split=args.data_split+'_train', is_train=True)
    val_dataset = create_dataset(args, opts, data_split=args.data_split+'_val', is_train=False)
    
    train_loader = DataLoader(train_dataset, **trainloader_params)
    test_loader = DataLoader(val_dataset, **testloader_params)
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    
    # Load checkpoint if resuming or evaluating
    chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
    if os.path.exists(chk_filename):
        opts.resume = chk_filename
    
    if opts.resume or opts.evaluate:
        chk_filename = opts.evaluate if opts.evaluate else opts.resume
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'], strict=True)
    
    if not opts.evaluate:
        # Setup optimizer
        optimizer = optim.AdamW([
            {"params": filter(lambda p: p.requires_grad, model.module.backbone.parameters()), 
             "lr": args.lr_backbone},
            {"params": filter(lambda p: p.requires_grad, model.module.head.parameters()), 
             "lr": args.lr_head},
        ], lr=args.lr_backbone, weight_decay=args.weight_decay)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)
        st = 0
        
        print('INFO: Training on {} batches'.format(len(train_loader)))
        
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            if 'best_acc' in checkpoint and checkpoint['best_acc'] is not None:
                best_acc = checkpoint['best_acc']
        
        # Training loop
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            losses_train = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            
            model.train()
            end = time.time()
            
            for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):
                data_time.update(time.time() - end)
                batch_size = len(batch_input)
                
                if torch.cuda.is_available():
                    batch_gt = batch_gt.cuda()
                    batch_input = batch_input.cuda()
                
                output = model(batch_input)  # (N, num_classes)
                
                optimizer.zero_grad()
                loss_train = criterion(output, batch_gt)
                losses_train.update(loss_train.item(), batch_size)
                
                # Calculate accuracy
                topk = (1, min(5, output.size(1)))
                acc1, acc5 = accuracy(output, batch_gt, topk=topk)
                top1.update(acc1[0], batch_size)
                if output.size(1) >= 5:
                    top5.update(acc5[0], batch_size)
                
                loss_train.backward()
                optimizer.step()
                
                batch_time.update(time.time() - end)
                end = time.time()
                
                if (idx + 1) % opts.print_freq == 0:
                    if output.size(1) >= 5:
                        print('Train: [{0}][{1}/{2}]\t'
                              'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                              'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                              'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                               epoch, idx + 1, len(train_loader), batch_time=batch_time,
                               data_time=data_time, loss=losses_train, top1=top1, top5=top5))
                    else:
                        print('Train: [{0}][{1}/{2}]\t'
                              'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                              'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                              'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                               epoch, idx + 1, len(train_loader), batch_time=batch_time,
                               data_time=data_time, loss=losses_train, top1=top1))
                    sys.stdout.flush()
            
            # Validation
            test_loss, test_top1, test_top5 = validate(test_loader, model, criterion, opts)
            
            # Log metrics
            train_writer.add_scalar('train_loss', losses_train.avg, epoch + 1)
            train_writer.add_scalar('train_top1', top1.avg, epoch + 1)
            if args.action_classes >= 5:
                train_writer.add_scalar('train_top5', top5.avg, epoch + 1)
                train_writer.add_scalar('test_top5', test_top5, epoch + 1)
            train_writer.add_scalar('test_loss', test_loss, epoch + 1)
            train_writer.add_scalar('test_top1', test_top1, epoch + 1)
            
            scheduler.step()
            
            # Save latest checkpoint
            chk_path = os.path.join(opts.checkpoint, 'latest_epoch.bin')
            print('Saving checkpoint to', chk_path)
            torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_acc': best_acc
            }, chk_path)
            
            # Save best checkpoint
            if test_top1 > best_acc:
                best_acc = test_top1
                best_chk_path = os.path.join(opts.checkpoint, 'best_epoch.bin')
                print(f"New best accuracy: {best_acc:.3f}, saving best checkpoint")
                torch.save({
                    'epoch': epoch+1,
                    'lr': scheduler.get_last_lr(),
                    'optimizer': optimizer.state_dict(),
                    'model': model.state_dict(),
                    'best_acc': best_acc
                }, best_chk_path)

    if opts.evaluate:
        test_loss, test_top1, test_top5 = validate(test_loader, model, criterion, opts)
        if args.action_classes >= 5:
            print('Loss {loss:.4f} \t'
                  'Acc@1 {top1:.3f} \t'
                  'Acc@5 {top5:.3f} \t'.format(loss=test_loss, top1=test_top1, top5=test_top5))
        else:
            print('Loss {loss:.4f} \t'
                  'Acc@1 {top1:.3f} \t'.format(loss=test_loss, top1=test_top1))

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    train_with_config(args, opts)
