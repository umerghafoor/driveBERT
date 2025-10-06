#!/bin/bash

# Run CLIP Zero-Shot Action Recognition Evaluation
# Make sure to install dependencies first: pip install -r requirements.txt

echo "Starting CLIP Zero-Shot Action Recognition Evaluation..."

# Set CUDA device if available
export CUDA_VISIBLE_DEVICES=0

# Run CLIP evaluation for different splits
for split in 0 1 2; do
    echo "Evaluating split $split..."
    python train_clip_action.py \
        --dataset_root /mnt/1C00FF7F00FF5DE8/Users/datasets/Drivenact \
        --video_dir_name a_column_co_driver \
        --clip_model ViT-B/32 \
        --output_dir results/clip_action_ViT-B-32 \
        --batch_size 16 \
        --n_frames 8 \
        --split $split \
        --print_freq 50
    
    echo "Split $split completed!"
done

echo "CLIP evaluation completed for all splits!"

# Also run with larger CLIP model for comparison
echo "Running with ViT-B/16 CLIP model..."
python train_clip_action.py \
    --dataset_root /mnt/1C00FF7F00FF5DE8/Users/datasets/Drivenact \
    --video_dir_name a_column_co_driver \
    --clip_model ViT-B/16 \
    --output_dir results/clip_action_ViT-B-16 \
    --batch_size 8 \
    --n_frames 8 \
    --split 0 \
    --print_freq 50

echo "All CLIP evaluations completed!"