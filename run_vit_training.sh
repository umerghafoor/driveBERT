#!/bin/bash

# Run ViT Action Recognition Training
# Make sure to install dependencies first: pip install -r requirements.txt

echo "Starting ViT Action Recognition Training..."

# Set CUDA device if available
export CUDA_VISIBLE_DEVICES=0

# Run ViT training
python train_vit_action.py \
    --dataset_root /mnt/1C00FF7F00FF5DE8/Users/datasets/Drivenact \
    --train_csv /mnt/1C00FF7F00FF5DE8/Users/datasets/Drivenact/activities_3s/kinect_color/midlevel.chunks_90.split_0.train.csv \
    --val_csv /mnt/1C00FF7F00FF5DE8/Users/datasets/Drivenact/activities_3s/kinect_color/midlevel.chunks_90.split_0.val.csv \
    --video_dir_name a_column_co_driver \
    --checkpoint_dir checkpoint/vit_action_split0 \
    --pretrained_model google/vit-base-patch16-224 \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --image_size 224 \
    --n_frames 16 \
    --print_freq 50

echo "ViT training completed!"