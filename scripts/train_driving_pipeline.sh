#!/bin/bash

# Driving Behavior Recognition Training Pipeline
# Specialized script for automotive behavior analysis

echo "======================================"
echo "Driving Behavior Recognition Training Pipeline"
echo "======================================"

# Configuration
CSV_PATH="/mnt/1C00FF7F00FF5DE8/Users/Github/driveBERT/sample data/midlevel.chunks_90.csv"
VIDEO_DIR="/path/to/your/video/directory"  # UPDATE THIS PATH
DATASET_NAME="driving_behavior"
OUTPUT_DIR="data/action"
CONFIG_TRAIN="configs/action/MB_train_DRIVING.yaml"
CONFIG_FINETUNE="configs/action/MB_ft_DRIVING.yaml"
CHECKPOINT_DIR="checkpoint/action/driving_behavior"
CHECKPOINT_FT_DIR="checkpoint/action/FT_driving_behavior"

# Step 1: Create data directory
echo "Step 1: Creating data directories..."
mkdir -p $OUTPUT_DIR
mkdir -p $CHECKPOINT_DIR
mkdir -p $CHECKPOINT_FT_DIR

# Step 2: Convert driving behavior dataset
echo "Step 2: Converting driving behavior dataset..."
echo "Using CSV: $CSV_PATH"

python tools/convert_driving_dataset.py \
    --csv_path "$CSV_PATH" \
    --video_dir "$VIDEO_DIR" \
    --output_path "$OUTPUT_DIR/${DATASET_NAME}.pkl" \
    --use_dummy_keypoints \
    --target_frames 243 \
    --validate

echo "Dataset conversion completed!"

# Check if conversion was successful
if [ ! -f "$OUTPUT_DIR/${DATASET_NAME}.pkl" ]; then
    echo "Error: Dataset conversion failed!"
    exit 1
fi

echo "Dataset successfully created with 39 driving behavior classes:"
echo "  - sitting_still, eating, fetching_an_object"
echo "  - reading_magazine, using_multimedia_display" 
echo "  - interacting_with_phone, working_on_laptop"
echo "  - and 32 other automotive behaviors..."
echo ""

# Step 3: Training options
echo "======================================"
echo "Training Options:"
echo "======================================"

echo "Option 1: Train from scratch (Recommended for large datasets)"
echo "Command:"
echo "python train_custom_action.py \\"
echo "    --config $CONFIG_TRAIN \\"
echo "    --checkpoint $CHECKPOINT_DIR \\"
echo "    --dataset_type pickle"
echo ""

echo "Option 2: Finetune from pretrained MotionBERT (Recommended)"
echo "Command:"
echo "python train_custom_action.py \\"
echo "    --config $CONFIG_FINETUNE \\"
echo "    --pretrained checkpoint/pretrain/MB_release \\"
echo "    --checkpoint $CHECKPOINT_FT_DIR \\"
echo "    --dataset_type pickle"
echo ""

echo "Option 3: Evaluate trained model"
echo "Command:"
echo "python train_custom_action.py \\"
echo "    --config $CONFIG_TRAIN \\"
echo "    --evaluate $CHECKPOINT_DIR/best_epoch.bin \\"
echo "    --dataset_type pickle"
echo ""

echo "======================================"
echo "Dataset Statistics:"
echo "======================================"

# Show some statistics about the converted dataset
python -c "
import pickle
try:
    with open('$OUTPUT_DIR/${DATASET_NAME}.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print(f'Total segments: {len(data[\"annotations\"])}')
    print(f'Activity classes: {data[\"num_classes\"]}')
    
    # Show split distribution
    for split_name, split_files in data['split'].items():
        print(f'{split_name.capitalize()}: {len(split_files)} segments')
    
    # Show top 10 activities
    from collections import defaultdict
    activity_counts = defaultdict(int)
    for annotation in data['annotations']:
        activity_counts[annotation['activity_name']] += 1
    
    print('\\nTop 10 Activities:')
    for activity, count in sorted(activity_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f'  {activity}: {count} segments')

except Exception as e:
    print(f'Could not load dataset statistics: {e}')
"

echo ""
echo "======================================"
echo "Special Features for Driving Behavior:"
echo "======================================"

echo "✓ 39 distinct automotive behavior classes"
echo "✓ Temporal segmentation with frame-level precision"
echo "✓ Participant-aware data splits"
echo "✓ Camera calibration data integration ready"
echo "✓ Long-duration video support (20+ minutes)"
echo "✓ Conservative augmentation to preserve behavior semantics"
echo ""

echo "======================================"
echo "Next Steps:"
echo "======================================"

echo "1. If you have actual video files:"
echo "   - Update VIDEO_DIR path in this script"
echo "   - Re-run conversion with --use_dummy_keypoints flag removed"
echo ""

echo "2. For production training:"
echo "   - Start with finetuning (Option 2) for faster convergence"
echo "   - Monitor validation accuracy for overfitting"
echo "   - Consider class balancing if needed"
echo ""

echo "3. For inference on new driving videos:"
echo "   - Use the trained model with infer_action.py"
echo "   - Ensure consistent preprocessing pipeline"
echo ""

echo "Pipeline setup complete!"
echo "Ready to train on your driving behavior dataset!"
echo "======================================"
