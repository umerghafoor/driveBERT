#!/bin/bash
# Example script to balance the DrivenAct dataset

# Set paths
INPUT="data/action/drivenact_inner_mirror_midlevel_split0.pkl"
OUTPUT="data/action/drivenact_inner_mirror_midlevel_split0.pkl"

echo "Balancing dataset..."
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo ""

# Simple undersampling (no dependencies required)
python tools/balance_dataset_simple.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --strategy median

echo ""
echo "Done! Use the balanced dataset for training to reduce class bias."
