#!/bin/bash

# DrivenAct Training Pipeline for Action Recognition
# This script demonstrates the complete pipeline for training on DrivenAct dataset

echo "======================================"
echo "DrivenAct Action Recognition Training Pipeline"
echo "======================================"

# Configuration - UPDATE THESE PATHS
DRIVENACT_DIR="/mnt/1C00FF7F00FF5DE8/Users/datasets/Drivenact"  # Path to DrivenAct dataset
DATASET_NAME="drivenact_inner_mirror_midlevel_split0"
OUTPUT_DIR="data/action"
CONFIG_FILE="configs/action/DrivenAct_train_midlevel.yaml"
CHECKPOINT_DIR="checkpoint/action/drivenact_training"

# DrivenAct specific parameters
CAMERA_VIEW="inner_mirror"     # Options: inner_mirror, a_column_co_driver, etc.
ANNOTATION_LEVEL="midlevel"    # Options: midlevel, objectlevel, tasklevel
SPLIT_ID=0                     # Options: 0, 1, 2

# Step 1: Create data directory
echo "Step 1: Creating data directories..."
mkdir -p $OUTPUT_DIR
mkdir -p $CHECKPOINT_DIR

# Step 2: Convert DrivenAct dataset to DriveBERT format
echo "Step 2: Converting DrivenAct dataset..."
python tools/convert_drivenact_dataset.py \
    --dataset_dir $DRIVENACT_DIR \
    --output_path "$OUTPUT_DIR/${DATASET_NAME}.pkl" \
    --camera_view $CAMERA_VIEW \
    --annotation_level $ANNOTATION_LEVEL \
    --split_id $SPLIT_ID \
    --target_frames 90 \
    --validate

echo "DrivenAct dataset conversion completed!"

# Step 3: Configuration is ready
echo "Step 3: Using DrivenAct configuration..."
echo "  - 39 midlevel driving activities"
echo "  - 90 frames per clip (3 seconds)"
echo "  - Inner mirror camera view"
echo "  - Pre-extracted OpenPose 3D keypoints"
echo ""
echo "Configuration file: $CONFIG_FILE"
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
echo "    --config configs/action/DrivenAct_ft_midlevel.yaml \\"
echo "    --pretrained checkpoint/pretrain/MB_release \\"
echo "    --checkpoint checkpoint/action/FT_MB_drivenact \\"
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
echo "DrivenAct Training Pipeline setup complete!"
echo ""
echo "Dataset Features:"
echo "  - 15 participants (vp1-vp15)"
echo "  - Multiple camera views"
echo "  - Pre-extracted OpenPose 3D keypoints"
echo "  - 39 midlevel driving activities"
echo "  - 3-second activity segments"
echo "  - Pre-defined train/val/test splits"
echo ""
echo "Next Steps:"
echo "1. Run the conversion command above"
echo "2. Execute the training command"
echo "3. Monitor training progress with TensorBoard"
echo ""
echo "Expected Activities:"
echo "  adjust_air_ventilation, adjust_mirrors, adjust_seat, answer_phone,"
echo "  change_gear, drinking, eating, hand_on_steering_wheel, reading,"
echo "  text_on_phone, working_on_laptop, write_on_paper, and more..."
echo "======================================"
