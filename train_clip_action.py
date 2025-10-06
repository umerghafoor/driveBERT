"""
CLIP Zero-Shot Action Recognition Script
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Import CLIP
import clip

from lib.utils.tools import *
from lib.utils.learning import *

# Comet ML integration
from comet_ml import start
import subprocess

# Start Comet ML experiment
experiment = start(
    api_key="OFmOeurqHyyi2aSzabZhxJz9Q",
    project_name="drivebert-clip",
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

class DrivenActCLIPDataset(Dataset):
    """Dataset class for DrivenAct RGB frames with CLIP preprocessing"""
    
    def __init__(self, dataset_root, csv_file, video_dir_name='a_column_co_driver', 
                 n_frames=8, clip_model_name="ViT-B/32"):
        """
        Args:
            dataset_root: Root directory of DrivenAct dataset
            csv_file: CSV file with annotations (train/val/test split)
            video_dir_name: Directory name containing videos
            n_frames: Number of frames to sample from each action clip
            clip_model_name: CLIP model variant to use
        """
        self.dataset_root = dataset_root
        self.video_dir_name = video_dir_name
        self.n_frames = n_frames
        
        # Load CLIP preprocessing
        _, self.preprocess = clip.load(clip_model_name, device="cpu")
        
        # Load annotations
        self.annotations = pd.read_csv(csv_file)
        print(f"Loaded {len(self.annotations)} annotations from {csv_file}")
        
        # Create class to index mapping and text prompts
        self.class_names = sorted(self.annotations['activity'].unique())
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        
        # Create text prompts for each action class
        self.text_prompts = self._create_text_prompts()
        
        print(f"Number of classes: {self.num_classes}")
        print("Sample classes:", self.class_names[:10])
    
    def _create_text_prompts(self):
        """Create descriptive text prompts for each action class"""
        prompts = []
        
        # Enhanced prompt templates for driver actions
        prompt_templates = [
            "A person {}",
            "A driver {}",
            "A person in a car {}",
            "Someone {}",
            "A driver in a vehicle {}"
        ]
        
        for class_name in self.class_names:
            # Clean up class name for better text description
            action_description = class_name.replace('_', ' ')
            
            # Create multiple prompts for each class and take the most relevant one
            class_prompts = []
            for template in prompt_templates:
                class_prompts.append(template.format(action_description))
            
            # For simplicity, use the first template. In practice, you might want to test different templates
            prompts.append(class_prompts[0])
        
        return prompts
    
    def get_text_prompts(self):
        """Return text prompts for all classes"""
        return self.text_prompts
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """Get a sample with sampled frames and label"""
        row = self.annotations.iloc[idx]
        
        # Get video path
        video_path = os.path.join(
            self.dataset_root, self.video_dir_name,
            f"vp{row['participant_id']}", 
            f"{row['file_id']}.mp4"
        )
        
        # Sample frames from the action segment
        frames = self._extract_frames(video_path, row['frame_start'], row['frame_end'])
        
        # Get label
        label = self.class_to_idx[row['activity']]
        
        return frames, label, row['activity']
    
    def _extract_frames(self, video_path, start_frame, end_frame):
        """Extract and sample frames from video segment"""
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            # Return dummy frames if video not found
            dummy_frame = torch.zeros(3, 224, 224)
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
                # Apply CLIP preprocessing
                frame = self.preprocess(frame)
                frames.append(frame)
            else:
                # If frame read fails, use last valid frame or zeros
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(torch.zeros(3, 224, 224))
        
        cap.release()
        
        # Stack frames: (n_frames, 3, H, W)
        return torch.stack(frames)

class CLIPActionClassifier:
    """CLIP-based zero-shot action classifier"""
    
    def __init__(self, clip_model_name="ViT-B/32", device="cuda"):
        self.device = device
        self.model, self.preprocess = clip.load(clip_model_name, device=device)
        print(f"Loaded CLIP model: {clip_model_name}")
    
    def encode_text_prompts(self, text_prompts):
        """Encode text prompts using CLIP text encoder"""
        text_tokens = clip.tokenize(text_prompts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def predict_batch(self, image_batch, text_features):
        """
        Predict action classes for a batch of images
        Args:
            image_batch: Tensor of shape (batch_size, n_frames, 3, H, W)
            text_features: Encoded text features for all classes
        Returns:
            logits: Similarity scores between images and text prompts
        """
        batch_size, n_frames, C, H, W = image_batch.shape
        
        # Reshape to process all frames at once
        images = image_batch.view(batch_size * n_frames, C, H, W)
        
        with torch.no_grad():
            # Encode images
            image_features = self.model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Reshape back to separate frames
            image_features = image_features.view(batch_size, n_frames, -1)
            
            # Average features across frames for each sample
            image_features = torch.mean(image_features, dim=1)  # (batch_size, feature_dim)
            
            # Calculate similarity with text prompts
            logits = (image_features @ text_features.T) * self.model.logit_scale.exp()
        
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
    parser.add_argument('--test_csv',
                       default='/mnt/1C00FF7F00FF5DE8/Users/datasets/Drivenact/activities_3s/kinect_color/midlevel.chunks_90.split_0.test.csv',
                       type=str, help='Test CSV file')
    parser.add_argument('--video_dir_name', default='a_column_co_driver', type=str,
                       help='Video directory name in dataset')
    parser.add_argument('--clip_model', default='ViT-B/32', type=str,
                       help='CLIP model variant (ViT-B/32, ViT-B/16, ViT-L/14, etc.)')
    parser.add_argument('--output_dir', default='results/clip_action', type=str,
                       help='Output directory for results')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--n_frames', default=8, type=int, help='Number of frames to sample')
    parser.add_argument('--print_freq', default=50, type=int, help='Print frequency')
    
    # Split selection
    parser.add_argument('--split', default=0, type=int, help='Data split to use (0, 1, 2)')
    
    return parser.parse_args()

def evaluate_split(classifier, data_loader, text_features, args, split_name="val"):
    """Evaluate CLIP model on a data split"""
    print(f"Evaluating on {split_name} split...")
    
    all_preds = []
    all_labels = []
    all_similarities = []
    
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    end = time.time()
    
    with torch.no_grad():
        for idx, (batch_input, batch_gt, activity_names) in tqdm(enumerate(data_loader), desc=f"Evaluating {split_name}"):
            batch_size = batch_input.size(0)
            
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
                batch_gt = batch_gt.cuda()
            
            # Get similarity scores (logits)
            logits = classifier.predict_batch(batch_input, text_features)
            
            # Calculate accuracy
            topk = (1, min(5, logits.size(1)))
            acc1, acc5 = accuracy(logits, batch_gt, topk=topk)
            top1.update(acc1[0], batch_size)
            if logits.size(1) >= 5:
                top5.update(acc5[0], batch_size)
            
            # Collect predictions for detailed metrics
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch_gt.cpu().numpy()
            similarities = logits.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_similarities.extend(similarities)
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if (idx + 1) % args.print_freq == 0:
                if logits.size(1) >= 5:
                    print(f'{split_name}: [{idx+1}/{len(data_loader)}]\t'
                          f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')
                else:
                    print(f'{split_name}: [{idx+1}/{len(data_loader)}]\t'
                          f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})')
    
    # Calculate macro F1 score
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # Calculate per-class metrics
    class_report = classification_report(
        all_labels, all_preds, 
        target_names=data_loader.dataset.class_names,
        output_dict=True
    )
    
    print(f'{split_name} Results:')
    print(f'Accuracy: {top1.avg:.3f}')
    if logits.size(1) >= 5:
        print(f'Top-5 Accuracy: {top5.avg:.3f}')
    print(f'Macro F1: {macro_f1:.3f}')
    
    return {
        'accuracy': top1.avg,
        'top5_accuracy': top5.avg if logits.size(1) >= 5 else 0,
        'macro_f1': macro_f1,
        'predictions': all_preds,
        'labels': all_labels,
        'similarities': all_similarities,
        'class_report': class_report
    }

def save_results(results, text_prompts, class_names, output_dir, split_name):
    """Save evaluation results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary metrics
    summary_file = os.path.join(output_dir, f'{split_name}_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"CLIP Zero-Shot Action Recognition Results ({split_name})\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy: {results['accuracy']:.3f}\n")
        if results['top5_accuracy'] > 0:
            f.write(f"Top-5 Accuracy: {results['top5_accuracy']:.3f}\n")
        f.write(f"Macro F1: {results['macro_f1']:.3f}\n\n")
        
        f.write("Text Prompts Used:\n")
        f.write("-" * 20 + "\n")
        for i, (class_name, prompt) in enumerate(zip(class_names, text_prompts)):
            f.write(f"{i}: {class_name} -> '{prompt}'\n")
        
        f.write("\nPer-class Results:\n")
        f.write("-" * 20 + "\n")
        for class_name in class_names:
            if class_name in results['class_report']:
                metrics = results['class_report'][class_name]
                f.write(f"{class_name}: P={metrics['precision']:.3f}, "
                       f"R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}\n")
    
    # Save detailed predictions
    predictions_file = os.path.join(output_dir, f'{split_name}_predictions.npz')
    np.savez(predictions_file,
             predictions=results['predictions'],
             labels=results['labels'],
             similarities=results['similarities'],
             class_names=class_names,
             text_prompts=text_prompts)
    
    print(f"Results saved to {output_dir}")

def main():
    args = parse_args()
    
    # Update CSV files based on split
    base_path = '/mnt/1C00FF7F00FF5DE8/Users/datasets/Drivenact/activities_3s/kinect_color'
    args.train_csv = f'{base_path}/midlevel.chunks_90.split_{args.split}.train.csv'
    args.val_csv = f'{base_path}/midlevel.chunks_90.split_{args.split}.val.csv'
    args.test_csv = f'{base_path}/midlevel.chunks_90.split_{args.split}.test.csv'
    
    print(f"Using data split {args.split}")
    print(f"Train CSV: {args.train_csv}")
    print(f"Val CSV: {args.val_csv}")
    print(f"Test CSV: {args.test_csv}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize CLIP classifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = CLIPActionClassifier(clip_model_name=args.clip_model, device=device)
    
    # Create datasets
    print("Creating datasets...")
    
    # We'll use train dataset just to get class information and text prompts
    train_dataset = DrivenActCLIPDataset(
        dataset_root=args.dataset_root,
        csv_file=args.train_csv,
        video_dir_name=args.video_dir_name,
        n_frames=args.n_frames,
        clip_model_name=args.clip_model
    )
    
    val_dataset = DrivenActCLIPDataset(
        dataset_root=args.dataset_root,
        csv_file=args.val_csv,
        video_dir_name=args.video_dir_name,
        n_frames=args.n_frames,
        clip_model_name=args.clip_model
    )
    
    test_dataset = DrivenActCLIPDataset(
        dataset_root=args.dataset_root,
        csv_file=args.test_csv,
        video_dir_name=args.video_dir_name,
        n_frames=args.n_frames,
        clip_model_name=args.clip_model
    )
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    print(f'Test samples: {len(test_dataset)}')
    
    # Get text prompts and encode them
    text_prompts = train_dataset.get_text_prompts()
    class_names = train_dataset.class_names
    
    print("\nText prompts:")
    for i, (class_name, prompt) in enumerate(zip(class_names[:10], text_prompts[:10])):
        print(f"{i}: {class_name} -> '{prompt}'")
    print("...")
    
    print("Encoding text prompts...")
    text_features = classifier.encode_text_prompts(text_prompts)
    
    # Create data loaders
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate on validation set
    val_results = evaluate_split(classifier, val_loader, text_features, args, "validation")
    
    # Evaluate on test set
    test_results = evaluate_split(classifier, test_loader, text_features, args, "test")
    
    # Log to Comet ML
    experiment.log_metric('val_accuracy', val_results['accuracy'])
    experiment.log_metric('val_macro_f1', val_results['macro_f1'])
    experiment.log_metric('test_accuracy', test_results['accuracy'])
    experiment.log_metric('test_macro_f1', test_results['macro_f1'])
    
    if val_results['top5_accuracy'] > 0:
        experiment.log_metric('val_top5_accuracy', val_results['top5_accuracy'])
        experiment.log_metric('test_top5_accuracy', test_results['top5_accuracy'])
    
    # Save results
    save_results(val_results, text_prompts, class_names, args.output_dir, f"split_{args.split}_validation")
    save_results(test_results, text_prompts, class_names, args.output_dir, f"split_{args.split}_test")
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Validation Accuracy: {val_results['accuracy']:.3f}")
    print(f"Validation Macro F1: {val_results['macro_f1']:.3f}")
    print(f"Test Accuracy: {test_results['accuracy']:.3f}")
    print(f"Test Macro F1: {test_results['macro_f1']:.3f}")
    if test_results['top5_accuracy'] > 0:
        print(f"Test Top-5 Accuracy: {test_results['top5_accuracy']:.3f}")
    print("="*50)

if __name__ == "__main__":
    main()