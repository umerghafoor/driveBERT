# Training Action Module on Custom Datasets

This comprehensive guide will help you train the DriveBERT action recognition module on your custom dataset containing videos and action classes.

## Overview

The DriveBERT action module expects skeleton keypoint data in a specific format. Since you have videos and action classes, you'll need to:

1. Extract keypoint data from videos
2. Convert data to the expected format
3. Create dataset configurations
4. Train the model

## Current System Understanding

The current system expects:
- **Input Format**: Skeleton keypoints (M × T × J × C format)
  - M: Number of people (max 2)
  - T: Number of frames (default 243)
  - J: Number of joints (17 joints in H36M format)
  - C: Coordinates + confidence (x, y, confidence)
- **Data Storage**: Pickle files containing annotations and splits
- **Coordinate System**: Camera coordinates normalized to [-1, 1] range

## Step-by-Step Implementation Plan

### Phase 1: Data Preprocessing (Estimated Time: 2-3 days)

#### 1.1 Video Keypoint Extraction
You'll need to extract 2D keypoints from your videos. Recommended approaches:

**Option A: Using OpenPose**
```bash
# Install OpenPose
# Extract keypoints for all videos
python extract_keypoints_openpose.py --video_dir /path/to/videos --output_dir keypoints/
```

**Option B: Using AlphaPose (Recommended)**
```bash
# Install AlphaPose
# Extract keypoints with higher accuracy
python -m alphapose.demo --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
    --checkpoint pretrained_models/fast_res50_256x192.pth \
    --video /path/to/video.mp4 --outdir output/
```

**Option C: Using MediaPipe**
```python
import mediapipe as mp
import cv2
import numpy as np

def extract_keypoints_mediapipe(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Extract keypoints and convert to required format
            # Add keypoint processing logic here
            pass
    
    cap.release()
    return keypoints
```

#### 1.2 Data Format Conversion
Create a conversion script to transform your data into the expected format:

```python
# tools/convert_custom_dataset.py

import os
import pickle
import numpy as np
from collections import defaultdict

def convert_keypoints_to_h36m(coco_keypoints):
    """
    Convert COCO keypoints to H36M format
    COCO: 17 keypoints
    H36M: 17 keypoints (different order/mapping)
    """
    # Implementation based on coco2h36m function in dataset_action.py
    # You'll need to map your keypoint format to H36M
    pass

def create_dataset_pickle(video_dir, annotation_file, output_path):
    """
    Create dataset pickle file compatible with DriveBERT
    
    Expected structure:
    {
        'split': {
            'train': [list of sample names],
            'val': [list of sample names],
            'test': [list of sample names]
        },
        'annotations': [
            {
                'frame_dir': 'sample_name',
                'total_frames': int,
                'img_shape': (height, width),
                'keypoint': np.array,  # (M, T, V, 2) - M persons, T frames, V joints, 2 coords
                'keypoint_score': np.array,  # (M, T, V) - confidence scores
                'label': int  # action class index
            },
            ...
        ]
    }
    """
    
    # Load your annotation data
    # Process videos and extract keypoints
    # Create the required data structure
    
    dataset = {
        'split': create_data_splits(),
        'annotations': process_annotations()
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

def create_data_splits():
    """Create train/val/test splits"""
    # Implement your splitting logic
    # Common ratios: 70% train, 15% val, 15% test
    pass

def process_annotations():
    """Process your video annotations"""
    annotations = []
    
    for video_path, action_label in video_annotations:
        # Extract keypoints from video
        keypoints, scores = extract_keypoints(video_path)
        
        # Convert to required format
        keypoints_h36m = convert_keypoints_to_h36m(keypoints)
        
        # Create annotation entry
        annotation = {
            'frame_dir': os.path.basename(video_path).replace('.mp4', ''),
            'total_frames': len(keypoints),
            'img_shape': (video_height, video_width),
            'keypoint': keypoints_h36m,  # (M, T, 17, 2)
            'keypoint_score': scores,     # (M, T, 17)
            'label': action_label
        }
        annotations.append(annotation)
    
    return annotations
```

### Phase 2: Dataset Integration (Estimated Time: 1 day)

#### 2.1 Create Custom Dataset Class
Extend the existing dataset classes for your data:

```python
# lib/data/dataset_custom.py

from lib.data.dataset_action import ActionDataset
import numpy as np

class CustomActionDataset(ActionDataset):
    def __init__(self, data_path, data_split, n_frames=243, random_move=True, scale_range=[1,1]):
        """
        Custom dataset class for your action recognition data
        """
        super(CustomActionDataset, self).__init__(data_path, data_split, n_frames, random_move, scale_range)
        
        # Add any custom preprocessing specific to your dataset
        self.preprocess_custom_data()
    
    def preprocess_custom_data(self):
        """Add any custom preprocessing steps"""
        # Handle different keypoint formats
        # Apply dataset-specific normalization
        # Handle missing keypoints
        pass
    
    def __getitem__(self, idx):
        """Generate one sample of data"""
        motion, label = self.motions[idx], self.labels[idx]  # (M,T,J,C)
        
        if self.random_move:
            motion = random_move(motion)
        if self.scale_range:
            result = crop_scale(motion, scale_range=self.scale_range)
        else:
            result = motion
            
        return result.astype(np.float32), label
```

#### 2.2 Update Training Script
Modify the training script to use your custom dataset:

```python
# train_action_custom.py

# Add import for your custom dataset
from lib.data.dataset_custom import CustomActionDataset

# In train_with_config function, replace:
# ntu60_xsub_train = NTURGBD(data_path=data_path, ...)
# with:
custom_train = CustomActionDataset(data_path=data_path, data_split=args.data_split+'_train', ...)
custom_val = CustomActionDataset(data_path=data_path, data_split=args.data_split+'_val', ...)
```

### Phase 3: Configuration Setup (Estimated Time: 0.5 day)

#### 3.1 Create Custom Configuration File
Create a config file for your dataset:

```yaml
# configs/action/MB_train_CUSTOM.yaml

# General  
finetune: False
partial_train: null

# Training 
epochs: 300
batch_size: 32 
lr_backbone: 0.0001
lr_head: 0.0001
weight_decay: 0.01
lr_decay: 0.99

# Model
model_version: class
maxlen: 243
dim_feat: 512
mlp_ratio: 2
depth: 5
dim_rep: 512
num_heads: 8
att_fuse: True
num_joints: 17
hidden_dim: 2048
dropout_ratio: 0.5

# Data
dataset: custom_dataset  # This should match your pickle file name
data_split: train_val_split  # Your split naming convention
clip_len: 243
action_classes: YOUR_NUM_CLASSES  # Replace with actual number

# Augmentation
random_move: True
scale_range_train: [1, 3]
scale_range_test: [2, 2]
```

#### 3.2 Create Action Class Names File
Create a file listing your action classes:

```python
# data/action/custom_actions.txt
0. action_class_1
1. action_class_2
2. action_class_3
...
```

### Phase 4: Training and Evaluation (Estimated Time: Variable based on dataset size)

#### 4.1 Training Commands

**Train from scratch:**
```bash
python train_action.py \
--config configs/action/MB_train_CUSTOM.yaml \
--checkpoint checkpoint/action/MB_train_CUSTOM
```

**Fine-tune from pretrained MotionBERT:**
```bash
python train_action.py \
--config configs/action/MB_ft_CUSTOM.yaml \
--pretrained checkpoint/pretrain/MB_release \
--checkpoint checkpoint/action/FT_MB_release_CUSTOM
```

**Evaluate:**
```bash
python train_action.py \
--config configs/action/MB_train_CUSTOM.yaml \
--evaluate checkpoint/action/MB_train_CUSTOM/best_epoch.bin
```

### Phase 5: Data Quality Validation (Estimated Time: 1 day)

#### 5.1 Data Validation Scripts
Create validation scripts to ensure data quality:

```python
# tools/validate_dataset.py

import pickle
import numpy as np
import matplotlib.pyplot as plt

def validate_dataset(dataset_path):
    """Validate the created dataset"""
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Dataset contains {len(data['annotations'])} samples")
    print(f"Splits: {list(data['split'].keys())}")
    
    # Check data consistency
    for i, sample in enumerate(data['annotations'][:10]):  # Check first 10 samples
        keypoints = sample['keypoint']
        scores = sample['keypoint_score']
        
        print(f"Sample {i}: Shape {keypoints.shape}, Label {sample['label']}")
        
        # Validate shape: (M, T, J, C)
        assert len(keypoints.shape) == 4, f"Invalid keypoint shape: {keypoints.shape}"
        assert keypoints.shape[2] == 17, f"Expected 17 joints, got {keypoints.shape[2]}"
        assert keypoints.shape[3] == 2, f"Expected 2 coordinates, got {keypoints.shape[3]}"
        
        # Check for NaN values
        if np.any(np.isnan(keypoints)):
            print(f"Warning: NaN values found in sample {i}")
        
        # Check coordinate ranges (should be in camera coordinates)
        if np.any(np.abs(keypoints) > 10):  # Reasonable range check
            print(f"Warning: Large coordinate values in sample {i}")

def visualize_sample(dataset_path, sample_idx=0):
    """Visualize a sample from the dataset"""
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    sample = data['annotations'][sample_idx]
    keypoints = sample['keypoint'][0]  # First person
    
    # Plot keypoints for a few frames
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, frame_idx in enumerate([0, len(keypoints)//4, len(keypoints)//2, len(keypoints)-1]):
        if i >= 4:
            break
        ax = axes[i]
        joints = keypoints[frame_idx]
        ax.scatter(joints[:, 0], joints[:, 1])
        ax.set_title(f'Frame {frame_idx}')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    
    plt.tight_layout()
    plt.savefig('sample_visualization.png')
    plt.show()

if __name__ == "__main__":
    dataset_path = "data/action/custom_dataset.pkl"
    validate_dataset(dataset_path)
    visualize_sample(dataset_path)
```

## Implementation Details and Best Practices

### 1. Keypoint Quality
- Ensure consistent keypoint detection across all videos
- Handle missing or low-confidence keypoints appropriately
- Consider using multiple keypoint detection methods and ensemble them

### 2. Data Augmentation
- The system supports spatial augmentations (rotation, scaling, translation)
- Temporal augmentations can be added if needed
- Be careful with augmentations that might change action semantics

### 3. Class Balancing
- Ensure balanced distribution of action classes
- Use stratified sampling for train/val/test splits
- Consider class weights if you have imbalanced data

### 4. Validation Strategy
- Use proper cross-validation if dataset is small
- Monitor both training and validation metrics
- Implement early stopping to prevent overfitting

### 5. Performance Optimization
- Use appropriate batch size based on your GPU memory
- Enable mixed precision training if supported
- Consider using multiple GPUs for large datasets

## Troubleshooting Common Issues

### 1. Memory Issues
- Reduce batch size
- Use gradient accumulation
- Process videos in chunks

### 2. Poor Performance
- Check keypoint quality
- Verify data splits
- Tune hyperparameters
- Consider pre-training on larger dataset

### 3. Convergence Issues
- Adjust learning rates
- Use learning rate scheduling
- Check for data leakage between splits

## Expected Results

After successful implementation, you should expect:
- Training loss to decrease consistently
- Validation accuracy to improve over epochs
- Reasonable performance on your test set
- Model checkpoints saved for inference

## Timeline Summary

- **Phase 1 (Preprocessing)**: 2-3 days
- **Phase 2 (Integration)**: 1 day  
- **Phase 3 (Configuration)**: 0.5 day
- **Phase 4 (Training)**: Variable (hours to days depending on dataset size)
- **Phase 5 (Validation)**: 1 day

**Total Estimated Time**: 5-6 days for initial implementation and training

## Next Steps

1. Start with Phase 1 - extract keypoints from a small subset of your videos
2. Implement the data conversion pipeline
3. Test with a small dataset before processing the full dataset
4. Gradually scale up and optimize the pipeline

This plan provides a comprehensive roadmap for adapting the DriveBERT action module to your custom video dataset. Each phase builds upon the previous one, ensuring a systematic approach to the implementation.
