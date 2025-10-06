"""
Vision Transformer (ViT) Action Recognition Training Script
Based on train_custom_action.py structure for DrivenAct dataset
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
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Import ViT from transformers
from transformers import ViTForImageClassification, ViTImageProcessor

from lib.utils.tools import *
from lib.utils.learning import *

# Comet ML integration
from comet_ml import start
import subprocess

# Start Comet ML experiment
experiment = start(
    api_key="OFmOeurqHyyi2aSzabZhxJz9Q",
    project_name="drivebert-vit",
    workspace="umerghafoor"
)

# Log current git commit hash
try:
    commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    experiment.log_other("git_commit", commit_hash)
except Exception as e:
    print(f"Could not log git commit: {e}")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class DrivenActVisionDataset(Dataset):
    """Dataset class for DrivenAct RGB frames with ViT preprocessing"""
    
    def __init__(self, dataset_root, csv_file, video_dir_name='a_column_co_driver', 
                 image_size=224, n_frames=16, is_train=True):
        """
        Args:
            dataset_root: Root directory of DrivenAct dataset
            csv_file: CSV file with annotations (train/val/test split)
            video_dir_name: Directory name containing videos (default: a_column_co_driver)
            image_size: Size to resize images (default: 224 for ViT)
            n_frames: Number of frames to sample from each action clip
            is_train: Whether this is training dataset (for augmentation)
        """
        self.dataset_root = dataset_root
        self.video_dir_name = video_dir_name
        self.image_size = image_size
        self.n_frames = n_frames
        self.is_train = is_train
        
        # Load annotations
        self.annotations = pd.read_csv(csv_file)
        print(f"Loaded {len(self.annotations)} annotations from {csv_file}")
        
        # Create class to index mapping
        self.class_names = sorted(self.annotations['activity'].unique())
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        
        print(f"Number of classes: {self.num_classes}")
        print("Classes:", self.class_names)
        
        # Data transforms
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """Get a sample with sampled frames and label"""
        row = self.annotations.iloc[idx]
        
        # Get video path
        video_path = os.path.join(
            self.dataset_root, self.video_dir_name,
            f"{row['file_id']}.mp4"
        )
        
        # Sample frames from the action segment
        frames = self._extract_frames(video_path, row['frame_start'], row['frame_end'])
        
        # Get label
        label = self.class_to_idx[row['activity']]
        
        return frames, label
    
    def _extract_frames(self, video_path, start_frame, end_frame):
        """Extract and sample frames from video segment"""
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            # Return dummy frames if video not found
            dummy_frame = torch.zeros(3, self.image_size, self.image_size)
            return torch.stack([dummy_frame] * self.n_frames)
        
        cap = cv2.VideoCapture(video_path)
        
        # Calculate frame indices to sample
        total_frames_in_segment = end_frame - start_frame + 1
        if total_frames_in_segment <= self.n_frames:
            # If segment is shorter than n_frames, repeat frames
            frame_indices = np.linspace(start_frame, end_frame, self.n_frames, dtype=int)
        else:
            # Sample n_frames uniformly from the segment
            frame_indices = np.linspace(start_frame, end_frame, self.n_frames, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = self.transform(frame)
                frames.append(frame)
            else:
                # If frame read fails, use last valid frame or zeros
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(torch.zeros(3, self.image_size, self.image_size))
        
        cap.release()
        
        # Stack frames: (n_frames, 3, H, W)
        return torch.stack(frames)

class ViTActionClassifier(nn.Module):
    """Vision Transformer for Action Recognition"""
    
    def __init__(self, num_classes, n_frames=16, pretrained_model="google/vit-base-patch16-224"):
        super(ViTActionClassifier, self).__init__()
        
        self.n_frames = n_frames
        self.num_classes = num_classes
        
        # Load pretrained ViT
        self.vit = ViTForImageClassification.from_pretrained(
            pretrained_model,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # For handling multiple frames, we'll average features from individual frames
        # Alternative: could use a temporal aggregation method
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, n_frames, 3, H, W)
        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        batch_size, n_frames, C, H, W = x.shape
        
        # Reshape to process all frames at once
        x = x.view(batch_size * n_frames, C, H, W)
        
        # Get ViT features for all frames
        outputs = self.vit(x)
        logits = outputs.logits  # (batch_size * n_frames, num_classes)
        
        # Reshape back and average across frames
        logits = logits.view(batch_size, n_frames, self.num_classes)
        logits = torch.mean(logits, dim=1)  # Average pooling across time
        
        return logits

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='/mnt/1C00FF7F00FF5DE8/Users/datasets/Drivenact', 
                       type=str, help='Root directory of DrivenAct dataset')
    parser.add_argument('--train_csv', 
                       default='/mnt/1C00FF7F00FF5DE8/Users/datasets/Drivenact/activities_3s/kinect_color/midlevel.chunks_90.split_0.train.csv',
                       type=str, help='Training CSV file')
    parser.add_argument('--val_csv',
                       default='/mnt/1C00FF7F00FF5DE8/Users/datasets/Drivenact/activities_3s/kinect_color/midlevel.chunks_90.split_0.val.csv',
                       type=str, help='Validation CSV file')
    parser.add_argument('--video_dir_name', default='a_column_co_driver', type=str,
                       help='Video directory name in dataset')
    parser.add_argument('--checkpoint_dir', default='checkpoint/vit_action', type=str,
                       help='Checkpoint directory')
    parser.add_argument('--pretrained_model', default='google/vit-base-patch16-224', type=str,
                       help='Pretrained ViT model name')
    
    # Training parameters
    parser.add_argument('--epochs', default=50, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
    parser.add_argument('--lr_decay', default=0.9, type=float, help='Learning rate decay factor')
    
    # Model parameters
    parser.add_argument('--image_size', default=224, type=int, help='Input image size')
    parser.add_argument('--n_frames', default=16, type=int, help='Number of frames to sample')
    
    # Other parameters
    parser.add_argument('--print_freq', default=50, type=int, help='Print frequency')
    parser.add_argument('--resume', default='', type=str, help='Resume from checkpoint')
    parser.add_argument('--evaluate', default='', type=str, help='Evaluate model')
    
    return parser.parse_args()

def validate(test_loader, model, criterion, args):
    """Validation function"""
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        end = time.time()
        for idx, (batch_input, batch_gt) in tqdm(enumerate(test_loader), desc="Validating"):
            batch_size = batch_input.size(0)
            
            if torch.cuda.is_available():
                batch_gt = batch_gt.cuda()
                batch_input = batch_input.cuda()
            
            output = model(batch_input)
            loss = criterion(output, batch_gt)
            
            # Update metrics
            losses.update(loss.item(), batch_size)
            acc1 = accuracy(output, batch_gt, topk=(1,))[0]
            top1.update(acc1[0], batch_size)
            
            # Collect predictions for F1 score
            preds = torch.argmax(output, dim=1).cpu().numpy()
            labels = batch_gt.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if (idx + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       idx + 1, len(test_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
    
    # Calculate macro F1 score
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f'Validation Results: Loss {losses.avg:.4f}, Acc@1 {top1.avg:.3f}, Macro-F1 {macro_f1:.3f}')
    
    return losses.avg, top1.avg, macro_f1

def main():
    args = parse_args()
    
    # Create checkpoint directory
    try:
        os.makedirs(args.checkpoint_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint_dir)
    
    # Setup tensorboard logging
    train_writer = tensorboardX.SummaryWriter(os.path.join(args.checkpoint_dir, "logs"))
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = DrivenActVisionDataset(
        dataset_root=args.dataset_root,
        csv_file=args.train_csv,
        video_dir_name=args.video_dir_name,
        image_size=args.image_size,
        n_frames=args.n_frames,
        is_train=True
    )
    
    val_dataset = DrivenActVisionDataset(
        dataset_root=args.dataset_root,
        csv_file=args.val_csv,
        video_dir_name=args.video_dir_name,
        image_size=args.image_size,
        n_frames=args.n_frames,
        is_train=False
    )
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print("Creating ViT model...")
    model = ViTActionClassifier(
        num_classes=train_dataset.num_classes,
        n_frames=args.n_frames,
        pretrained_model=args.pretrained_model
    )
    
    criterion = nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=10, gamma=args.lr_decay)
    
    best_acc = 0
    best_f1 = 0
    start_epoch = 0
    
    # Resume from checkpoint
    if args.resume:
        print(f'Loading checkpoint {args.resume}')
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        best_f1 = checkpoint.get('best_f1', 0)
    
    # Evaluation only
    if args.evaluate:
        print("Evaluating model...")
        test_loss, test_acc, test_f1 = validate(val_loader, model, criterion, args)
        return
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f'Training epoch {epoch}')
        
        model.train()
        losses_train = AverageMeter()
        top1 = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        end = time.time()
        
        for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}"):
            data_time.update(time.time() - end)
            batch_size = batch_input.size(0)
            
            if torch.cuda.is_available():
                batch_gt = batch_gt.cuda()
                batch_input = batch_input.cuda()
            
            output = model(batch_input)
            
            optimizer.zero_grad()
            loss_train = criterion(output, batch_gt)
            losses_train.update(loss_train.item(), batch_size)
            
            # Calculate accuracy
            acc1 = accuracy(output, batch_gt, topk=(1,))[0]
            top1.update(acc1[0], batch_size)
            
            loss_train.backward()
            optimizer.step()
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if (idx + 1) % args.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       epoch, idx + 1, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses_train, top1=top1))
                sys.stdout.flush()
        
        # Validation
        test_loss, test_acc, test_f1 = validate(val_loader, model, criterion, args)
        
        # Log metrics
        train_writer.add_scalar('train_loss', losses_train.avg, epoch + 1)
        train_writer.add_scalar('train_acc', top1.avg, epoch + 1)
        train_writer.add_scalar('val_loss', test_loss, epoch + 1)
        train_writer.add_scalar('val_acc', test_acc, epoch + 1)
        train_writer.add_scalar('val_macro_f1', test_f1, epoch + 1)
        
        # Log to Comet ML
        experiment.log_metric('train_loss', losses_train.avg, step=epoch + 1)
        experiment.log_metric('train_acc', top1.avg, step=epoch + 1)
        experiment.log_metric('val_loss', test_loss, step=epoch + 1)
        experiment.log_metric('val_acc', test_acc, step=epoch + 1)
        experiment.log_metric('val_macro_f1', test_f1, step=epoch + 1)
        
        scheduler.step()
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, 'latest_epoch.bin')
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_acc': best_acc,
            'best_f1': best_f1
        }, checkpoint_path)
        
        # Save best checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            best_f1 = test_f1
            best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_epoch.bin')
            print(f"New best accuracy: {best_acc:.3f}, F1: {best_f1:.3f}")
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_acc,
                'best_f1': best_f1
            }, best_checkpoint_path)
    
    print(f"Training completed. Best accuracy: {best_acc:.3f}, Best F1: {best_f1:.3f}")

if __name__ == "__main__":
    main()