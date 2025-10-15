#!/usr/bin/env python3
"""
Balance class distribution in skeleton action recognition dataset.
Uses SMOTE for minority classes and undersampling for majority classes.
"""

import argparse
import pickle
import numpy as np
from collections import Counter, defaultdict
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek


def balance_dataset(input_path, output_path, strategy='auto', target_count=None, method='smote'):
    """
    Balance dataset using SMOTE and/or undersampling.
    
    Args:
        input_path: Path to input pickle file
        output_path: Path to output balanced pickle file
        strategy: 'auto' (median), 'minority' (min count), or int (specific target)
        target_count: Explicit target samples per class (overrides strategy)
        method: 'smote', 'undersample', 'combined', or 'smote_tomek'
    """
    
    # Load data
    print(f"Loading: {input_path}")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    annotations = data.get('annotations', [])
    if not annotations:
        raise ValueError("No annotations found in pickle file")
    
    # Extract features and labels
    # Assume each annotation has 'keypoint' (N, T, V, C) and 'label'
    features = []
    labels = []
    
    for ann in annotations:
        label = ann.get('label', -1)
        if label == -1:
            continue
        
        # Flatten keypoint data for SMOTE
        keypoint = ann.get('keypoint', None)
        if keypoint is None:
            print(f"Warning: annotation missing 'keypoint', skipping")
            continue
        
        # Flatten to 1D feature vector
        feat = np.array(keypoint).flatten()
        features.append(feat)
        labels.append(label)
    
    if not features:
        raise ValueError("No valid features found in annotations")
    
    X = np.array(features)
    y = np.array(labels)
    
    print(f"\nOriginal dataset:")
    print(f"  Total samples: {len(y)}")
    counts = Counter(y)
    print(f"  Classes: {len(counts)}")
    print(f"  Min count: {min(counts.values())}")
    print(f"  Max count: {max(counts.values())}")
    print(f"  Median count: {int(np.median(list(counts.values())))}")
    
    # Determine target count
    if target_count is None:
        if strategy == 'auto':
            target_count = int(np.median(list(counts.values())))
        elif strategy == 'minority':
            target_count = min(counts.values())
        else:
            target_count = int(strategy)
    
    print(f"\nTarget samples per class: {target_count}")
    
    # Apply balancing
    if method == 'smote':
        # SMOTE only (oversample minority classes)
        print("Applying SMOTE oversampling...")
        smote = SMOTE(sampling_strategy='auto', k_neighbors=min(5, min(counts.values())-1), random_state=42)
        X_bal, y_bal = smote.fit_resample(X, y)
        
    elif method == 'undersample':
        # Undersample only (reduce majority classes)
        print("Applying random undersampling...")
        rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        X_bal, y_bal = rus.fit_resample(X, y)
        
    elif method == 'combined':
        # SMOTE + Undersampling
        print("Applying SMOTE + undersampling...")
        smote = SMOTE(sampling_strategy='auto', k_neighbors=min(5, min(counts.values())-1), random_state=42)
        X_tmp, y_tmp = smote.fit_resample(X, y)
        
        rus = RandomUnderSampler(sampling_strategy={k: target_count for k in np.unique(y_tmp)}, random_state=42)
        X_bal, y_bal = rus.fit_resample(X_tmp, y_tmp)
        
    elif method == 'smote_tomek':
        # SMOTE + Tomek links cleaning
        print("Applying SMOTE-Tomek...")
        smt = SMOTETomek(sampling_strategy='auto', random_state=42)
        X_bal, y_bal = smt.fit_resample(X, y)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"\nBalanced dataset:")
    print(f"  Total samples: {len(y_bal)}")
    bal_counts = Counter(y_bal)
    print(f"  Classes: {len(bal_counts)}")
    print(f"  Min count: {min(bal_counts.values())}")
    print(f"  Max count: {max(bal_counts.values())}")
    print(f"  Class distribution:")
    for label in sorted(bal_counts.keys()):
        print(f"    Class {label}: {bal_counts[label]}")
    
    # Reconstruct annotations
    # Note: This creates synthetic annotations for SMOTE-generated samples
    # Original shape information is needed to reshape features back
    print("\nReconstructing annotations...")
    
    # Get original keypoint shape from first annotation
    orig_shape = annotations[0]['keypoint'].shape if isinstance(annotations[0]['keypoint'], np.ndarray) else np.array(annotations[0]['keypoint']).shape
    
    balanced_annotations = []
    for feat, label in zip(X_bal, y_bal):
        # Reshape flat feature back to original keypoint shape
        keypoint = feat.reshape(orig_shape)
        
        # Create new annotation (copy structure from original)
        ann = {
            'keypoint': keypoint.tolist() if isinstance(keypoint, np.ndarray) else keypoint,
            'label': int(label),
            'total_frames': orig_shape[1] if len(orig_shape) > 1 else 1,
        }
        balanced_annotations.append(ann)
    
    # Save balanced dataset
    balanced_data = data.copy()
    balanced_data['annotations'] = balanced_annotations
    
    with open(output_path, 'wb') as f:
        pickle.dump(balanced_data, f)
    
    print(f"\n✓ Saved balanced dataset to: {output_path}")
    print(f"  Original: {len(annotations)} samples")
    print(f"  Balanced: {len(balanced_annotations)} samples")


def main():
    parser = argparse.ArgumentParser(description='Balance class distribution in action dataset')
    parser.add_argument('--input', required=True, help='Input pickle file path')
    parser.add_argument('--output', required=True, help='Output balanced pickle file path')
    parser.add_argument('--method', default='combined', choices=['smote', 'undersample', 'combined', 'smote_tomek'],
                        help='Balancing method (default: combined)')
    parser.add_argument('--strategy', default='auto', help='Sampling strategy: auto, minority, or target count')
    parser.add_argument('--target', type=int, help='Explicit target samples per class')
    
    args = parser.parse_args()
    
    balance_dataset(
        input_path=args.input,
        output_path=args.output,
        strategy=args.strategy,
        target_count=args.target,
        method=args.method
    )


if __name__ == '__main__':
    main()
