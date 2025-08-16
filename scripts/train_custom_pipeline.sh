#!/bin/bash

# Example Script for Training Action Recognition on Custom Dataset
# This script demonstrates the complete pipeline from data preparation to training

echo "======================================"
echo "Custom Action Recognition Training Pipeline"
echo "======================================"

# Configuration
VIDEO_DIR="/path/to/your/videos"           # CHANGE THIS to your video directory
ANNOTATION_FILE="/path/to/annotations.json" # CHANGE THIS to your annotation file
DATASET_NAME="my_custom_dataset"           # CHANGE THIS to your dataset name
OUTPUT_DIR="data/action"
CONFIG_FILE="configs/action/MB_train_CUSTOM.yaml"
CHECKPOINT_DIR="checkpoint/action/custom_training"

# Step 1: Create data directory
echo "Step 1: Creating data directories..."
mkdir -p $OUTPUT_DIR
mkdir -p $CHECKPOINT_DIR

# Step 2: Convert video dataset to DriveBERT format
echo "Step 2: Converting video dataset..."
python tools/convert_custom_dataset.py \
    --video_dir $VIDEO_DIR \
    --annotation_file $ANNOTATION_FILE \
    --output_path "$OUTPUT_DIR/${DATASET_NAME}.pkl" \
    --keypoint_method dummy \
    --validate

echo "Dataset conversion completed!"

# Step 3: Update configuration file
echo "Step 3: Please update the following in your config file:"
echo "  - action_classes: Set to your number of action classes"
echo "  - dataset: Set to '$DATASET_NAME'"
echo "  - Adjust other hyperparameters as needed"
echo ""
echo "Edit: $CONFIG_FILE"
echo ""

# Step 4: Train from scratch
echo "Step 4: Training from scratch..."
echo "Command to run:"
echo "python train_custom_action.py \\"
echo "    --config $CONFIG_FILE \\"
echo "    --checkpoint $CHECKPOINT_DIR \\"
echo "    --dataset_type pickle"
echo ""

# Step 5: Alternative - Finetune from pretrained
echo "Step 5: Alternative - Finetune from pretrained MotionBERT:"
echo "python train_custom_action.py \\"
echo "    --config configs/action/MB_ft_CUSTOM.yaml \\"
echo "    --pretrained checkpoint/pretrain/MB_release \\"
echo "    --checkpoint checkpoint/action/FT_MB_release_CUSTOM \\"
echo "    --dataset_type pickle"
echo ""

# Step 6: Evaluation
echo "Step 6: Evaluation:"
echo "python train_custom_action.py \\"
echo "    --config $CONFIG_FILE \\"
echo "    --evaluate $CHECKPOINT_DIR/best_epoch.bin \\"
echo "    --dataset_type pickle"
echo ""

echo "======================================"
echo "Pipeline setup complete!"
echo "Please follow the steps above to train your model."
echo "======================================"
