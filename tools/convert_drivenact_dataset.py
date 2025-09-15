"""
DrivenAct Dataset Converter for DriveBERT Action Recognition
Converts DrivenAct dataset with pre-extracted OpenPose 3D keypoints and activity annotations
"""

import os
import pandas as pd
import numpy as np
import pickle
import json
import argparse
from tqdm import tqdm
from collections import defaultdict
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

# Add the lib directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

# DrivenAct activity mappings will be dynamically created from actual data
# This avoids hardcoding and missing activities
DRIVENACT_ACTIVITIES = {}

def build_activity_mapping(splits, annotation_level='midlevel'):
    """
    Build activity mapping dynamically from actual data using FULL dataset
    """
    all_activities = set()
    
    # Use full dataset if available, otherwise combine all splits
    if '_full_for_activities' in splits:
        full_df = splits['_full_for_activities']
        if 'activity' in full_df.columns:
            all_activities.update(full_df['activity'].unique())
            print(f"Building activity mapping from full dataset with {len(full_df)} samples")
    else:
        # Fallback: collect from all splits
        for split_name, split_df in splits.items():
            if split_name != '_full_for_activities' and 'activity' in split_df.columns:
                activities = split_df['activity'].unique()
                all_activities.update(activities)
        print(f"Building activity mapping from combined splits")
    
    # Create sorted mapping
    activity_mapping = {activity: idx for idx, activity in enumerate(sorted(all_activities))}
    
    print(f"Found {len(activity_mapping)} unique {annotation_level} activities:")
    for activity, idx in sorted(activity_mapping.items(), key=lambda x: x[1]):
        print(f"  {idx:2d}. {activity}")
    
    return activity_mapping

def load_drivenact_annotations(activities_dir, camera_view, annotation_level, split_id=0):
    """
    Load DrivenAct activity annotations
    
    Args:
        activities_dir: Path to activities_3s directory
        camera_view: Camera view (e.g., 'inner_mirror', 'a_column_co_driver') 
        annotation_level: 'midlevel', 'objectlevel', or 'tasklevel'
        split_id: Split ID (0, 1, or 2)
    
    Returns:
        Dictionary with train, val, test splits AND full dataset for activity mapping
    """
    view_dir = os.path.join(activities_dir, camera_view)
    
    splits = {}
    for split_name in ['train', 'val', 'test']:
        csv_file = f"{annotation_level}.chunks_90.split_{split_id}.{split_name}.csv"
        csv_path = os.path.join(view_dir, csv_file)
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            splits[split_name] = df
        else:
            print(f"Warning: {csv_path} not found")
            splits[split_name] = pd.DataFrame()
    
    # IMPORTANT: Load full CSV for complete activity mapping
    full_csv_file = f"{annotation_level}.chunks_90.csv"
    full_csv_path = os.path.join(view_dir, full_csv_file)
    if os.path.exists(full_csv_path):
        full_df = pd.read_csv(full_csv_path)
        splits['_full_for_activities'] = full_df
        print(f"Loaded full dataset for activity mapping: {len(full_df)} samples")
    else:
        print(f"Warning: Full CSV not found: {full_csv_path}")
        # Fallback: combine all splits
        all_data = []
        for split_df in splits.values():
            if not split_df.empty:
                all_data.append(split_df)
        if all_data:
            splits['_full_for_activities'] = pd.concat(all_data, ignore_index=True)
    
    return splits

# Global cache for loaded keypoint files to avoid re-reading
_KEYPOINT_CACHE = {}

def load_openpose_keypoints(keypoints_path):
    """
    Load OpenPose keypoints from CSV file with caching.
    """
    if not os.path.exists(keypoints_path):
        return None
    
    # Check cache first
    if keypoints_path in _KEYPOINT_CACHE:
        return _KEYPOINT_CACHE[keypoints_path]
    
    try:
        # Load keypoints CSV with optimized settings
        keypoints_df = pd.read_csv(keypoints_path, low_memory=False)
        
        # DrivenAct OpenPose CSV format: frame_id, timestamp, then 26 keypoints * 4 (x,y,z,confidence)
        if keypoints_df.shape[1] < 106:
            print(f"Warning: Invalid format {keypoints_path}: {keypoints_df.shape[1]} cols")
            return None
        
        # Extract and reshape keypoints efficiently
        keypoints = keypoints_df.iloc[:, 2:106].values.reshape(-1, 26, 4)
        
        # Keep only x,y,confidence (drop z) - convert to float32 for memory efficiency
        keypoints_2d = keypoints[:, :, [0, 1, 3]].astype(np.float32)
        
        # Cache the result
        _KEYPOINT_CACHE[keypoints_path] = keypoints_2d
        
        return keypoints_2d
        
    except Exception as e:
        print(f"Error loading {keypoints_path}: {e}")
        return None

def map_openpose_to_h36m(openpose_kpts):
    """
    Map DrivenAct OpenPose keypoints (26 points) to H36M format (17 joints)
    
    DrivenAct keypoints (from CSV analysis):
    0: nose, 1: lElbow, 2: lWrist, 3: rHeel, 4: rHip, 5: rSmallToe, 6: neck, 
    7: lSmallToe, 8: rWrist, 9: rAnkle, 10: lHip, 11: lHeel, 12: lKnee, 
    13: lEye, 14: midHip, 15: background, 16: lEar, 17: rElbow, 18: rShoulder, 
    19: rKnee, 20: lShoulder, 21: lBigToe, 22: rEye, 23: rEar, 24: rBigToe, 25: lAnkle
    
    H36M joints (17):
    0: Hip, 1: RHip, 2: RKnee, 3: RAnkle, 4: LHip, 5: LKnee, 6: LAnkle,
    7: Spine, 8: Thorax, 9: Neck/Nose, 10: Head, 11: LShoulder, 12: LElbow,
    13: LWrist, 14: RShoulder, 15: RElbow, 16: RWrist
    """
    if openpose_kpts is None:
        return None, None
    
    T = openpose_kpts.shape[0]
    h36m_keypoints = np.zeros((1, T, 17, 3))  # Single person format (M, T, J, C)
    
    # Mapping from OpenPose indices to H36M
    # Using available keypoints, some may be approximated
    mapping = {
        0: 14,  # Hip -> midHip
        1: 4,   # RHip -> rHip  
        2: 19,  # RKnee -> rKnee
        3: 9,   # RAnkle -> rAnkle
        4: 10,  # LHip -> lHip
        5: 12,  # LKnee -> lKnee
        6: 25,  # LAnkle -> lAnkle
        7: 14,  # Spine -> midHip (approximation)
        8: 6,   # Thorax -> neck
        9: 0,   # Neck/Nose -> nose
        10: 0,  # Head -> nose (approximation)
        11: 20, # LShoulder -> lShoulder
        12: 1,  # LElbow -> lElbow
        13: 2,  # LWrist -> lWrist
        14: 18, # RShoulder -> rShoulder
        15: 17, # RElbow -> rElbow
        16: 8,  # RWrist -> rWrist
    }
    
    # Map coordinates and confidence
    confidence = np.zeros((T, 17))
    
    for h36m_idx, openpose_idx in mapping.items():
        if openpose_idx < openpose_kpts.shape[1]:  # Check bounds
            h36m_keypoints[0, :, h36m_idx, :2] = openpose_kpts[:, openpose_idx, :2]  # x, y
            confidence[:, h36m_idx] = openpose_kpts[:, openpose_idx, 2]  # confidence
    
    return h36m_keypoints, confidence

def convert_drivenact_dataset(dataset_dir, output_path, camera_view='inner_mirror', 
                            annotation_level='midlevel', split_id=0, validate=False):
    """
    Convert DrivenAct dataset to DriveBERT format
    
    Args:
        dataset_dir: Path to DrivenAct dataset directory
        output_path: Output pickle file path
        camera_view: Camera view to use ('inner_mirror', 'a_column_co_driver', etc.)
        annotation_level: 'midlevel', 'objectlevel', or 'tasklevel'
        split_id: Split ID (0, 1, or 2)
    """
    print(f"Converting DrivenAct dataset: {camera_view}, {annotation_level}, split_{split_id}")
    
    # Load activity annotations
    activities_dir = os.path.join(dataset_dir, 'activities_3s')
    splits = load_drivenact_annotations(activities_dir, camera_view, annotation_level, split_id)
    
    # Build dynamic activity mapping from actual data
    activity_mapping = build_activity_mapping(splits, annotation_level)
    global DRIVENACT_ACTIVITIES
    DRIVENACT_ACTIVITIES = activity_mapping
    
    # Setup directory paths
    video_dir = os.path.join(dataset_dir, camera_view)
    keypoints_dir = os.path.join(dataset_dir, 'openpose_3d')
    
    annotations = []
    dataset_splits = {'train': [], 'val': [], 'test': []}
    
    # Process each split (excluding the full dataset used for activity mapping)
    for split_name, split_df in splits.items():
        if split_name == '_full_for_activities':  # Skip the full dataset
            continue
            
        print(f"Processing {split_name} split: {len(split_df)} samples")
        
        # For testing, limit to first 10 samples
        if validate:
            split_df = split_df.head(10)
            print(f"  Limited to {len(split_df)} samples for testing")
        
        for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split_name}"):
            try:
                # Extract information from row (actual CSV structure)
                participant_id = f"vp{row['participant_id']}"
                file_id = row['file_id']
                annotation_id = row['annotation_id']
                frame_start = row['frame_start']
                frame_end = row['frame_end']
                activity = row['activity']
                chunk_id = row['chunk_id']
                
                # Extract run information from file_id
                run_id = file_id.split('/')[-1].replace('.ids_1', '')  # Remove extension

                # print(f"Processing sample: {participant_id}, {run_id}, frames {frame_start}-{frame_end}, activity: {activity}")
                
                # Create unique sample name
                sample_name = f"{participant_id}_{run_id}_{chunk_id}_{activity}"
                
                # Find corresponding keypoints file  
                kpts_vp_dir = os.path.join(keypoints_dir, participant_id)
                keypoints_file = None
                
                if os.path.exists(kpts_vp_dir):
                    keypoints_filename = f"{run_id}.ids_1.openpose.3d.csv"
                    keypoints_file = os.path.join(kpts_vp_dir, keypoints_filename)
                
                if not keypoints_file or not os.path.exists(keypoints_file):
                    print(f"Warning: No keypoints file found for {participant_id}, {run_id}")
                    continue
                
                # Load keypoints for this time chunk
                openpose_kpts = load_openpose_keypoints(keypoints_file)
                if openpose_kpts is None:
                    continue
                
                # Extract specific frame range
                chunk_keypoints = None
                if frame_end <= len(openpose_kpts):
                    chunk_keypoints = openpose_kpts[frame_start:frame_end]
                    
                else:
                    print(f"Warning: Frame range {frame_start}-{frame_end} exceeds keypoints length {len(openpose_kpts)}")
                    continue
                
                if chunk_keypoints is None or len(chunk_keypoints) == 0:
                    continue
                
                # Map to H36M format
                h36m_keypoints, confidence = map_openpose_to_h36m(chunk_keypoints)
                
                if h36m_keypoints is None:
                    continue
                
                # Get activity label
                if annotation_level == 'midlevel':
                    label = DRIVENACT_ACTIVITIES.get(activity, 0)
                else:
                    # For object/task level, you'd need different mappings
                    label = hash(activity) % 100  # Temporary mapping
                
                # Create annotation entry
                annotation = {
                    'frame_dir': sample_name,
                    'img_shape': (480, 640),  # Typical video resolution
                    'keypoint': h36m_keypoints[:, :, :, :2],  # (M, T, J, 2) - x,y coordinates
                    'keypoint_score': confidence[np.newaxis, :, :],  # (M, T, J) - confidence
                    'label': label,
                    'activity_name': activity,
                    'participant': participant_id,
                    'run_id': run_id,
                    'frame_start': frame_start,
                    'frame_end': frame_end,
                    'annotation_id': annotation_id,
                    'chunk_id': chunk_id
                }
                
                annotations.append(annotation)
                dataset_splits[split_name].append(sample_name)
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
    
    # Create final dataset structure
    dataset = {
        'split': dataset_splits,
        'annotations': annotations,
        'metadata': {
            'camera_view': camera_view,
            'annotation_level': annotation_level,
            'split_id': split_id,
            'num_classes': len(DRIVENACT_ACTIVITIES) if annotation_level == 'midlevel' else 100
        }
    }
    
    # Save dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Dataset saved to {output_path}")
    print(f"Total samples: {len(annotations)}")
    for split_name, samples in dataset_splits.items():
        print(f"  {split_name}: {len(samples)} samples")
    
    # Save activity classes
    if annotation_level == 'midlevel':
        activity_names_path = output_path.replace('.pkl', '_actions.txt')
        with open(activity_names_path, 'w') as f:
            for activity, idx in sorted(DRIVENACT_ACTIVITIES.items(), key=lambda x: x[1]):
                f.write(f"{idx}. {activity}\n")
        print(f"Activity names saved to {activity_names_path}")
    
    return dataset

def validate_drivenact_dataset(dataset_path):
    """
    Validate the created DrivenAct dataset
    """
    print(f"Validating dataset: {dataset_path}")
    
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    annotations = data['annotations']
    splits = data['split']
    metadata = data.get('metadata', {})
    
    print(f"Dataset contains {len(annotations)} samples")
    print(f"Splits: {list(splits.keys())}")
    print(f"Metadata: {metadata}")
    
    # Validate a few samples
    for i, sample in enumerate(annotations[:3]):
        keypoints = sample['keypoint']
        scores = sample['keypoint_score']
        
        print(f"\nSample {i} ({sample['frame_dir']}):")
        print(f"  Keypoints shape: {keypoints.shape}")
        print(f"  Scores shape: {scores.shape}")
        print(f"  Activity: {sample['activity_name']} (label: {sample['label']})")
        print(f"  Participant: {sample['participant']}")
        
        # Basic validation
        assert len(keypoints.shape) == 4, f"Invalid keypoint shape: {keypoints.shape}"
        assert keypoints.shape[2] == 17, f"Expected 17 joints, got {keypoints.shape[2]}"
        assert keypoints.shape[3] == 2, f"Expected 2 coordinates, got {keypoints.shape[3]}"
    
    print("\nValidation completed!")

def main():
    parser = argparse.ArgumentParser(description='Convert DrivenAct dataset to DriveBERT format')
    parser.add_argument('--dataset_dir', required=True, help='Path to DrivenAct dataset directory')
    parser.add_argument('--output_path', required=True, help='Output pickle file path')
    parser.add_argument('--camera_view', default='inner_mirror', 
                       choices=['inner_mirror', 'a_column_co_driver', 'a_column_driver', 
                               'ceiling', 'kinect_color', 'kinect_depth', 'kinect_ir', 'steering_wheel'],
                       help='Camera view to use')
    parser.add_argument('--annotation_level', default='midlevel',
                       choices=['midlevel', 'objectlevel', 'tasklevel'],
                       help='Annotation level')
    parser.add_argument('--split_id', type=int, default=0, choices=[0, 1, 2],
                       help='Split ID (0, 1, or 2)')
    parser.add_argument('--validate', action='store_true', help='Validate the created dataset')
    
    args = parser.parse_args()
    
    # Convert dataset
    dataset = convert_drivenact_dataset(
        dataset_dir=args.dataset_dir,
        output_path=args.output_path,
        camera_view=args.camera_view,
        annotation_level=args.annotation_level,
        split_id=args.split_id,
        validate=args.validate
    )
    
    # Validate if requested
    if args.validate:
        validate_drivenact_dataset(args.output_path)

if __name__ == "__main__":
    main()
