#!/usr/bin/env python3
"""
Validate DrivenAct dataset conversion for correctness
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def load_and_analyze_dataset(pkl_path):
    """Load and analyze the converted dataset"""
    print(f"Loading dataset from: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Dataset keys: {dataset.keys()}")
    print(f"Metadata: {dataset.get('metadata', {})}")
    
    # Analyze each split
    for split_name in ['train', 'val', 'test']:
        if split_name in dataset:
            split_data = dataset[split_name]
            print(f"\n{split_name.upper()} SPLIT:")
            print(f"  Number of samples: {len(split_data)}")
            
            if len(split_data) > 0:
                sample = split_data[0]
                print(f"  Sample keys: {sample.keys()}")
                
                # Check keypoints
                if 'keypoints' in sample:
                    kpts = sample['keypoints']
                    print(f"  Keypoints shape: {kpts.shape}")
                    print(f"  Keypoints dtype: {kpts.dtype}")
                    print(f"  Keypoints range: [{np.min(kpts):.3f}, {np.max(kpts):.3f}]")
                    
                # Check scores/confidence
                if 'scores' in sample:
                    scores = sample['scores']
                    print(f"  Scores shape: {scores.shape}")
                    print(f"  Scores range: [{np.min(scores):.3f}, {np.max(scores):.3f}]")
                
                # Check activity labels
                if 'activity' in sample:
                    print(f"  Activity: {sample['activity']}")
                    print(f"  Label: {sample.get('label', 'N/A')}")
                
    return dataset

def validate_keypoint_format(dataset):
    """Validate keypoints are in correct format"""
    print("\n=== KEYPOINT FORMAT VALIDATION ===")
    
    errors = []
    samples_checked = 0
    
    for split_name in ['train', 'val', 'test']:
        if split_name in dataset:
            for i, sample in enumerate(dataset[split_name]):
                samples_checked += 1
                
                # Check keypoints shape
                kpts = sample['keypoints']
                expected_shape = (1, 90, 17, 2)  # (M, T, J, C) - single person, 90 frames, 17 joints, x-y coords
                
                if kpts.shape != expected_shape:
                    errors.append(f"{split_name}[{i}]: Wrong keypoints shape {kpts.shape}, expected {expected_shape}")
                
                # Check scores shape
                if 'scores' in sample:
                    scores = sample['scores']
                    expected_scores_shape = (1, 90, 17)  # (M, T, J)
                    
                    if scores.shape != expected_scores_shape:
                        errors.append(f"{split_name}[{i}]: Wrong scores shape {scores.shape}, expected {expected_scores_shape}")
                
                # Check for NaN/inf values
                if np.any(np.isnan(kpts)) or np.any(np.isinf(kpts)):
                    errors.append(f"{split_name}[{i}]: Contains NaN/inf values in keypoints")
                
                # Check coordinate ranges (should be reasonable for normalized/pixel coordinates)
                if np.max(np.abs(kpts)) > 10000:  # Reasonable upper bound
                    errors.append(f"{split_name}[{i}]: Keypoint coordinates seem unreasonable: max={np.max(np.abs(kpts))}")
    
    print(f"Checked {samples_checked} samples")
    
    if errors:
        print(f"FOUND {len(errors)} ERRORS:")
        for error in errors:
            print(f"  ❌ {error}")
        return False
    else:
        print("✅ All keypoint formats are correct!")
        return True

def validate_activity_labels(dataset):
    """Validate activity labels are correct"""
    print("\n=== ACTIVITY LABEL VALIDATION ===")
    
    all_activities = set()
    all_labels = set()
    activity_to_label = {}
    
    for split_name in ['train', 'val', 'test']:
        if split_name in dataset:
            for sample in dataset[split_name]:
                activity = sample['activity']
                label = sample['label']
                
                all_activities.add(activity)
                all_labels.add(label)
                
                if activity in activity_to_label:
                    if activity_to_label[activity] != label:
                        print(f"❌ Inconsistent labeling: {activity} maps to both {activity_to_label[activity]} and {label}")
                        return False
                else:
                    activity_to_label[activity] = label
    
    print(f"Found {len(all_activities)} unique activities")
    print(f"Found {len(all_labels)} unique labels")
    print(f"Label range: {min(all_labels)} to {max(all_labels)}")
    
    # Check if labels are contiguous
    expected_labels = set(range(len(all_activities)))
    if all_labels != expected_labels:
        print(f"❌ Labels are not contiguous: {sorted(all_labels)} vs expected {sorted(expected_labels)}")
        return False
    
    print("✅ Activity labels are correct!")
    print("\nSample activity mappings:")
    for activity, label in sorted(activity_to_label.items())[:10]:
        print(f"  {label:2d}: {activity}")
    
    return True

def visualize_sample_keypoints(dataset, sample_idx=0):
    """Visualize keypoints from a sample"""
    print(f"\n=== KEYPOINT VISUALIZATION ===")
    
    # Get first sample from train split
    if 'train' not in dataset or len(dataset['train']) == 0:
        print("No training samples to visualize")
        return
    
    sample = dataset['train'][sample_idx]
    keypoints = sample['keypoints'][0]  # Remove batch dimension: (90, 17, 2)
    scores = sample['scores'][0] if 'scores' in sample else None  # (90, 17)
    
    print(f"Visualizing sample: {sample['sample_name']}")
    print(f"Activity: {sample['activity']}")
    print(f"Keypoints shape: {keypoints.shape}")
    
    # Plot first and middle frames
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, (ax, frame_idx) in enumerate(zip(axes, [0, 45])):
        frame_kpts = keypoints[frame_idx]  # (17, 2)
        frame_scores = scores[frame_idx] if scores is not None else np.ones(17)
        
        # Plot keypoints
        valid_mask = frame_scores > 0.1  # Only plot confident keypoints
        x_coords = frame_kpts[valid_mask, 0]
        y_coords = frame_kpts[valid_mask, 1]
        
        ax.scatter(x_coords, y_coords, c=frame_scores[valid_mask], cmap='viridis', s=50)
        ax.set_title(f'Frame {frame_idx}')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add joint numbers
        for j, (x, y, conf) in enumerate(zip(frame_kpts[:, 0], frame_kpts[:, 1], frame_scores)):
            if conf > 0.1:
                ax.annotate(str(j), (x, y), xytext=(3, 3), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('sample_keypoints_visualization.png', dpi=150, bbox_inches='tight')
    print("✅ Keypoints visualization saved to 'sample_keypoints_visualization.png'")
    plt.close()

def check_temporal_consistency(dataset):
    """Check if keypoints have reasonable temporal consistency"""
    print(f"\n=== TEMPORAL CONSISTENCY CHECK ===")
    
    inconsistencies = []
    
    for split_name in ['train', 'val', 'test']:
        if split_name in dataset:
            for i, sample in enumerate(dataset[split_name][:3]):  # Check first 3 samples
                keypoints = sample['keypoints'][0]  # (90, 17, 2)
                
                # Calculate frame-to-frame motion
                motion = np.diff(keypoints, axis=0)  # (89, 17, 2)
                motion_magnitude = np.sqrt(np.sum(motion**2, axis=2))  # (89, 17)
                
                # Check for unreasonable jumps
                max_motion = np.max(motion_magnitude, axis=1)  # (89,) - max motion per frame
                
                # Flag frames with very large motion (potential tracking errors)
                large_motion_frames = np.where(max_motion > 100)[0]  # Threshold depends on coordinate system
                
                if len(large_motion_frames) > 5:  # Allow some tracking errors
                    inconsistencies.append(f"{split_name}[{i}]: {len(large_motion_frames)} frames with large motion")
    
    if inconsistencies:
        print("⚠️  Found some temporal inconsistencies (may be normal):")
        for inc in inconsistencies:
            print(f"  {inc}")
    else:
        print("✅ Temporal consistency looks reasonable!")

def compare_with_original_data(dataset, original_dataset_dir):
    """Compare converted data with original annotations"""
    print(f"\n=== COMPARISON WITH ORIGINAL DATA ===")
    
    # Load original annotations
    annotation_file = Path(original_dataset_dir) / "activities_3s" / "inner_mirror" / "midlevel.chunks_90.split_0.train.csv"
    
    if not annotation_file.exists():
        print(f"⚠️  Original annotation file not found: {annotation_file}")
        return
    
    original_df = pd.read_csv(annotation_file)
    print(f"Original annotations: {len(original_df)} samples")
    
    # Compare activity counts
    converted_activities = {}
    for split_name in ['train', 'val', 'test']:
        if split_name in dataset:
            for sample in dataset[split_name]:
                activity = sample['activity']
                converted_activities[activity] = converted_activities.get(activity, 0) + 1
    
    original_activities = original_df['activity'].value_counts().to_dict()
    
    print("Activity count comparison (first 5):")
    for activity in list(original_activities.keys())[:5]:
        orig_count = original_activities.get(activity, 0)
        conv_count = converted_activities.get(activity, 0)
        print(f"  {activity}: Original={orig_count}, Converted={conv_count}")

def main():
    # Path to test dataset
    test_pkl = Path("data/action/test_drivenact.pkl")
    
    if not test_pkl.exists():
        print(f"❌ Test dataset not found: {test_pkl}")
        print("Please run the conversion script first with --validate flag")
        return
    
    print("🔍 DRIVENACT DATASET VALIDATION")
    print("=" * 50)
    
    # Load and analyze dataset
    dataset = load_and_analyze_dataset(test_pkl)
    
    # Run validation checks
    format_ok = validate_keypoint_format(dataset)
    labels_ok = validate_activity_labels(dataset)
    
    # Visualize sample
    try:
        visualize_sample_keypoints(dataset)
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
    
    # Check temporal consistency
    check_temporal_consistency(dataset)
    
    # Compare with original (if available)
    try:
        compare_with_original_data(dataset, "/mnt/1C00FF7F00FF5DE8/Users/datasets/Drivenact")
    except Exception as e:
        print(f"⚠️  Original comparison failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("🔍 VALIDATION SUMMARY")
    print(f"  Keypoint format: {'✅ PASS' if format_ok else '❌ FAIL'}")
    print(f"  Activity labels: {'✅ PASS' if labels_ok else '❌ FAIL'}")
    
    if format_ok and labels_ok:
        print("\n🎉 VALIDATION PASSED! Your conversion looks correct.")
        print("You can proceed with full dataset conversion.")
    else:
        print("\n⚠️  VALIDATION ISSUES FOUND! Please fix before proceeding.")

if __name__ == "__main__":
    main()
