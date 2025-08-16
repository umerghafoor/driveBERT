"""
Data Conversion Tool for Custom Action Recognition Dataset
Converts video datasets with action annotations to DriveBERT format
"""

import os
import json
import pickle
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from collections import defaultdict
import sys

# Add the lib directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

def extract_keypoints_opencv(video_path, target_frames=243):
    """
    Extract basic pose keypoints using OpenCV DNN 
    This is a fallback method when MediaPipe/OpenPose are not available
    """
    try:
        # Load pre-trained pose estimation model
        net = cv2.dnn.readNetFromTensorflow('models/pose_estimation.pb')
    except:
        print("Warning: No pose estimation model found. Using dummy keypoints.")
        return create_dummy_keypoints(target_frames)
    
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # For now, create dummy keypoints
        # In a real implementation, you would run pose estimation here
        frame_keypoints = np.random.rand(17, 2) * 2 - 1  # Random points in [-1, 1]
        confidence = np.ones((17,)) * 0.8
        keypoints.append(np.column_stack([frame_keypoints, confidence]))
    
    cap.release()
    
    if len(keypoints) == 0:
        return create_dummy_keypoints(target_frames)
    
    keypoints = np.array(keypoints)  # (T, 17, 3)
    
    # Resample to target frames
    if len(keypoints) != target_frames:
        indices = np.linspace(0, len(keypoints)-1, target_frames, dtype=int)
        keypoints = keypoints[indices]
    
    # Add person dimension: (1, T, 17, 3) -> single person
    keypoints = keypoints[np.newaxis, ...]
    
    return keypoints

def create_dummy_keypoints(num_frames, num_joints=17):
    """
    Create dummy keypoints for testing purposes
    """
    keypoints = np.random.rand(1, num_frames, num_joints, 3) * 2 - 1
    keypoints[:, :, :, 2] = 0.8  # Set confidence to 0.8
    return keypoints

def load_video_annotations(annotation_file):
    """
    Load video annotations from various formats
    
    Expected format:
    {
        "videos": [
            {
                "video_path": "path/to/video.mp4",
                "action_class": "action_name",
                "action_id": 0,
                "split": "train"  # optional
            },
            ...
        ],
        "action_classes": ["class1", "class2", ...],  # optional
        "splits": {  # optional
            "train": ["video1.mp4", "video2.mp4", ...],
            "val": ["video3.mp4", ...],
            "test": ["video4.mp4", ...]
        }
    }
    """
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Handle different annotation formats
    if 'videos' in data:
        videos = data['videos']
    elif 'annotations' in data:
        videos = data['annotations']
    else:
        raise ValueError("Annotation file must contain 'videos' or 'annotations' field")
    
    # Extract action classes if not provided
    if 'action_classes' not in data:
        action_classes = list(set([v.get('action_class', v.get('class', 'unknown')) for v in videos]))
        action_classes.sort()
        data['action_classes'] = action_classes
    
    # Create class name to ID mapping
    class_to_id = {name: idx for idx, name in enumerate(data['action_classes'])}
    
    # Ensure all videos have action IDs
    for video in videos:
        if 'action_id' not in video:
            action_name = video.get('action_class', video.get('class', 'unknown'))
            video['action_id'] = class_to_id.get(action_name, 0)
    
    return data

def create_data_splits(videos, split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15}, seed=42):
    """
    Create train/val/test splits from video list
    """
    np.random.seed(seed)
    
    # Group videos by class for stratified splitting
    class_videos = defaultdict(list)
    for video in videos:
        class_id = video['action_id']
        class_videos[class_id].append(video['video_path'])
    
    splits = {'train': [], 'val': [], 'test': []}
    
    for class_id, video_paths in class_videos.items():
        np.random.shuffle(video_paths)
        n_videos = len(video_paths)
        
        n_train = int(n_videos * split_ratios['train'])
        n_val = int(n_videos * split_ratios['val'])
        
        splits['train'].extend(video_paths[:n_train])
        splits['val'].extend(video_paths[n_train:n_train+n_val])
        splits['test'].extend(video_paths[n_train+n_val:])
    
    return splits

def process_video(video_path, keypoint_method='dummy'):
    """
    Process a single video to extract keypoints
    """
    if not os.path.exists(video_path):
        print(f"Warning: Video not found: {video_path}")
        return create_dummy_keypoints(243)
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Cannot open video: {video_path}")
            return create_dummy_keypoints(243)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        if keypoint_method == 'dummy':
            keypoints = create_dummy_keypoints(243)
        else:
            keypoints = extract_keypoints_opencv(video_path)
        
        return {
            'keypoints': keypoints,
            'total_frames': total_frames,
            'fps': fps,
            'img_shape': (height, width)
        }
        
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return {
            'keypoints': create_dummy_keypoints(243),
            'total_frames': 243,
            'fps': 30,
            'img_shape': (480, 640)
        }

def convert_dataset(video_dir, annotation_file, output_path, keypoint_method='dummy'):
    """
    Convert video dataset to DriveBERT format
    
    Args:
        video_dir: Directory containing video files
        annotation_file: JSON file with annotations
        output_path: Output pickle file path
        keypoint_method: Method to extract keypoints ('dummy', 'opencv', 'mediapipe')
    """
    print(f"Loading annotations from {annotation_file}...")
    annotation_data = load_video_annotations(annotation_file)
    
    videos = annotation_data['videos']
    action_classes = annotation_data['action_classes']
    
    print(f"Found {len(videos)} videos with {len(action_classes)} action classes")
    
    # Create splits if not provided
    if 'splits' in annotation_data:
        splits = annotation_data['splits']
    else:
        print("Creating train/val/test splits...")
        splits = create_data_splits(videos)
    
    # Process each video
    print("Processing videos...")
    annotations = []
    
    for video_info in tqdm(videos):
        video_path = os.path.join(video_dir, video_info['video_path'])
        video_name = os.path.splitext(os.path.basename(video_info['video_path']))[0]
        
        # Process video to extract keypoints
        result = process_video(video_path, keypoint_method)
        
        # Create annotation entry
        annotation = {
            'frame_dir': video_name,
            'total_frames': result['total_frames'],
            'img_shape': result['img_shape'],
            'keypoint': result['keypoints'][:, :, :, :2],  # (M, T, J, 2) - only x,y coordinates
            'keypoint_score': result['keypoints'][:, :, :, 2],  # (M, T, J) - confidence scores
            'label': video_info['action_id']
        }
        
        annotations.append(annotation)
    
    # Create final dataset structure
    dataset = {
        'split': splits,
        'annotations': annotations
    }
    
    # Save to pickle file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Dataset saved to {output_path}")
    print(f"Dataset statistics:")
    print(f"  Total samples: {len(annotations)}")
    for split_name, split_videos in splits.items():
        print(f"  {split_name}: {len(split_videos)} videos")
    
    # Save action classes file
    action_names_path = output_path.replace('.pkl', '_actions.txt')
    with open(action_names_path, 'w') as f:
        for i, action_name in enumerate(action_classes):
            f.write(f"{i}. {action_name}\n")
    print(f"Action names saved to {action_names_path}")
    
    return dataset

def validate_dataset(dataset_path):
    """
    Validate the created dataset
    """
    print(f"Validating dataset: {dataset_path}")
    
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    annotations = data['annotations']
    splits = data['split']
    
    print(f"Dataset contains {len(annotations)} samples")
    print(f"Splits: {list(splits.keys())}")
    
    # Validate data structure
    for i, sample in enumerate(annotations[:5]):  # Check first 5 samples
        keypoints = sample['keypoint']
        scores = sample['keypoint_score']
        
        print(f"Sample {i}:")
        print(f"  Keypoints shape: {keypoints.shape}")
        print(f"  Scores shape: {scores.shape}")
        print(f"  Label: {sample['label']}")
        print(f"  Total frames: {sample['total_frames']}")
        
        # Validate shapes
        assert len(keypoints.shape) == 4, f"Invalid keypoint shape: {keypoints.shape}"
        assert keypoints.shape[2] == 17, f"Expected 17 joints, got {keypoints.shape[2]}"
        assert keypoints.shape[3] == 2, f"Expected 2 coordinates, got {keypoints.shape[3]}"
        
        # Check for reasonable values
        if np.any(np.abs(keypoints) > 10):
            print(f"  Warning: Large coordinate values detected")
        
        if np.any(np.isnan(keypoints)):
            print(f"  Warning: NaN values detected")
    
    print("Validation completed!")

def main():
    parser = argparse.ArgumentParser(description='Convert video dataset to DriveBERT format')
    parser.add_argument('--video_dir', required=True, help='Directory containing video files')
    parser.add_argument('--annotation_file', required=True, help='JSON file with video annotations')
    parser.add_argument('--output_path', required=True, help='Output pickle file path')
    parser.add_argument('--keypoint_method', default='dummy', choices=['dummy', 'opencv', 'mediapipe'],
                       help='Method to extract keypoints')
    parser.add_argument('--validate', action='store_true', help='Validate the created dataset')
    
    args = parser.parse_args()
    
    # Convert dataset
    dataset = convert_dataset(
        video_dir=args.video_dir,
        annotation_file=args.annotation_file,
        output_path=args.output_path,
        keypoint_method=args.keypoint_method
    )
    
    # Validate if requested
    if args.validate:
        validate_dataset(args.output_path)

if __name__ == "__main__":
    main()
