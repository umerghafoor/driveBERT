#!/usr/bin/env python3
"""
Simple dataset balancing using undersampling only.
No external dependencies beyond pickle, numpy, and random.
"""

import argparse
import pickle
import random
import numpy as np
from collections import Counter, defaultdict


def balance_dataset_simple(input_path, output_path, target_count=None, strategy='median'):
    """
    Balance dataset using simple random undersampling.
    
    Args:
        input_path: Path to input pickle file
        output_path: Path to output balanced pickle file
        target_count: Target samples per class (None = auto calculate)
        strategy: 'median', 'mean', 'min', or specific int
    """
    
    # Load data
    print(f"Loading: {input_path}")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    annotations = data.get('annotations', [])
    if not annotations:
        raise ValueError("No annotations found in pickle file")
    
    # Count labels
    labels = [ann.get('label', -1) for ann in annotations]
    counts = Counter(labels)
    if -1 in counts:
        del counts[-1]
    
    print(f"\nOriginal dataset:")
    print(f"  Total samples: {len(annotations)}")
    print(f"  Classes present: {len(counts)}")
    print(f"  Min count: {min(counts.values())}")
    print(f"  Max count: {max(counts.values())}")
    print(f"  Median count: {int(np.median(list(counts.values())))}")
    print(f"  Mean count: {int(np.mean(list(counts.values())))}")
    
    # Determine target
    if target_count is None:
        if strategy == 'median':
            target_count = int(np.median(list(counts.values())))
        elif strategy == 'mean':
            target_count = int(np.mean(list(counts.values())))
        elif strategy == 'min':
            target_count = min(counts.values())
        else:
            target_count = int(strategy)
    
    print(f"\nTarget samples per class: {target_count}")
    
    # Group by label
    by_label = defaultdict(list)
    for ann in annotations:
        label = ann.get('label', -1)
        if label != -1:
            by_label[label].append(ann)
    
    # Undersample
    balanced = []
    for label, samples in sorted(by_label.items()):
        if len(samples) <= target_count:
            balanced.extend(samples)
            print(f"  Class {label}: kept all {len(samples)} samples")
        else:
            sampled = random.sample(samples, target_count)
            balanced.extend(sampled)
            print(f"  Class {label}: undersampled from {len(samples)} to {target_count}")
    
    # Shuffle
    random.shuffle(balanced)
    
    # Show results
    bal_counts = Counter([a['label'] for a in balanced])
    print(f"\nBalanced dataset:")
    print(f"  Total samples: {len(balanced)}")
    print(f"  Min count: {min(bal_counts.values())}")
    print(f"  Max count: {max(bal_counts.values())}")
    
    # Save
    balanced_data = data.copy()
    balanced_data['annotations'] = balanced
    
    with open(output_path, 'wb') as f:
        pickle.dump(balanced_data, f)
    
    print(f"\n✓ Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Balance dataset via undersampling')
    parser.add_argument('--input', required=True, help='Input pickle path')
    parser.add_argument('--output', required=True, help='Output pickle path')
    parser.add_argument('--strategy', default='median', help='median, mean, min, or int')
    parser.add_argument('--target', type=int, help='Explicit target count')
    
    args = parser.parse_args()
    
    balance_dataset_simple(
        input_path=args.input,
        output_path=args.output,
        target_count=args.target,
        strategy=args.strategy
    )


if __name__ == '__main__':
    main()
