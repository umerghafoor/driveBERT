import pickle
import argparse
import os

parser = argparse.ArgumentParser(description="Filter dataset to only include samples with label < 3.")
parser.add_argument('--input', type=str, required=True, help='Path to input pickle file')
parser.add_argument('--output', type=str, default=None, help='Path to output pickle file (default: filtered_<input>.pkl)')
args = parser.parse_args()

with open(args.input, 'rb') as f:
    data = pickle.load(f)

# Try to find the main annotation list
if 'annotations' in data:
    annotations = data['annotations']
else:
    raise ValueError('No "annotations" key found in the pickle file.')

filtered_annotations = [sample for sample in annotations if sample.get('label', -1) < 3]
print(f"Filtered {len(annotations) - len(filtered_annotations)} samples with label >= 3. Remaining: {len(filtered_annotations)}")

data['annotations'] = filtered_annotations

output_path = args.output or f"filtered_{os.path.basename(args.input)}"
with open(output_path, 'wb') as f:
    pickle.dump(data, f)
print(f"Filtered dataset saved to {output_path}")
