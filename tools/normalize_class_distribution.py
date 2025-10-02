import pickle
import argparse
import os
import random
from collections import defaultdict

parser = argparse.ArgumentParser(description="Normalize class distribution by downsampling to the smallest class size.")
parser.add_argument('--input', type=str, required=True, help='Path to input pickle file')
parser.add_argument('--output', type=str, default=None, help='Path to output pickle file (default: normalized_<input>.pkl)')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
args = parser.parse_args()

random.seed(args.seed)

with open(args.input, 'rb') as f:
    data = pickle.load(f)

if 'annotations' in data:
    annotations = data['annotations']
else:
    raise ValueError('No "annotations" key found in the pickle file.')

# Group samples by class
class_to_samples = defaultdict(list)
for sample in annotations:
    label = sample.get('label', -1)
    if label != -1:
        class_to_samples[label].append(sample)

# Find minimum class size
min_count = min(len(samples) for samples in class_to_samples.values())
print(f"Minimum class size: {min_count}")

# Downsample each class to min_count
normalized_annotations = []
for label, samples in class_to_samples.items():
    if len(samples) > min_count:
        selected = random.sample(samples, min_count)
    else:
        selected = samples
    normalized_annotations.extend(selected)
    print(f"Class {label}: {len(selected)} samples")

random.shuffle(normalized_annotations)
data['annotations'] = normalized_annotations

output_path = args.output or f"normalized_{os.path.basename(args.input)}"
with open(output_path, 'wb') as f:
    pickle.dump(data, f)
print(f"Normalized dataset saved to {output_path}")
