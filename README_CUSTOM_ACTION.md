# Custom Action Recognition Training

This guide provides a complete implementation for training the DriveBERT action recognition module on your custom dataset with videos and action classes.

## Quick Start

### Prerequisites

```bash
# Install required packages
pip install opencv-python mediapipe tqdm tensorboardX

# Optional: for better keypoint extraction
pip install torch torchvision
```

### 1. Prepare Your Data

Create an annotation file in JSON format following this structure:

```json
{
    "action_classes": ["walking", "running", "jumping", ...],
    "videos": [
        {
            "video_path": "video_001.mp4",
            "action_class": "walking", 
            "action_id": 0
        },
        ...
    ]
}
```

See `examples/annotation_format_example.json` for a complete example.

### 2. Convert Your Dataset

```bash
python tools/convert_custom_dataset.py \
    --video_dir /path/to/your/videos \
    --annotation_file /path/to/annotations.json \
    --output_path data/action/custom_dataset.pkl \
    --keypoint_method dummy \
    --validate
```

### 3. Configure Training

Edit `configs/action/MB_train_CUSTOM.yaml`:
- Set `action_classes` to your number of classes
- Set `dataset` to your dataset name (without .pkl extension)
- Adjust other parameters as needed

### 4. Train

**From scratch:**
```bash
python train_custom_action.py \
    --config configs/action/MB_train_CUSTOM.yaml \
    --checkpoint checkpoint/action/custom_training \
    --dataset_type pickle
```

**Finetune from pretrained:**
```bash
python train_custom_action.py \
    --config configs/action/MB_ft_CUSTOM.yaml \
    --pretrained checkpoint/pretrain/MB_release \
    --checkpoint checkpoint/action/FT_custom \
    --dataset_type pickle
```

### 5. Evaluate

```bash
python train_custom_action.py \
    --config configs/action/MB_train_CUSTOM.yaml \
    --evaluate checkpoint/action/custom_training/best_epoch.bin \
    --dataset_type pickle
```

## Alternative: Video-based Training

For smaller datasets, you can train directly from videos without preprocessing:

```bash
python train_custom_action.py \
    --config configs/action/MB_train_CUSTOM.yaml \
    --checkpoint checkpoint/action/video_training \
    --dataset_type video \
    --video_dir /path/to/videos \
    --annotation_file /path/to/annotations.json \
    --keypoint_extractor mediapipe
```

## Files Created

The implementation includes these new files:

### Core Implementation
- `lib/data/dataset_custom.py` - Custom dataset classes
- `train_custom_action.py` - Enhanced training script
- `tools/convert_custom_dataset.py` - Data conversion tool

### Configuration
- `configs/action/MB_train_CUSTOM.yaml` - Training config template
- `configs/action/MB_ft_CUSTOM.yaml` - Finetuning config template

### Examples and Scripts
- `scripts/train_custom_pipeline.sh` - Complete pipeline script
- `examples/annotation_format_example.json` - Annotation format example
- `docs/custom_action_training.md` - Detailed documentation

## Dataset Format

The system expects skeleton keypoints in this format:
- **Input**: (M, T, J, C) where:
  - M: Number of people (max 2, pad with zeros if needed)
  - T: Number of frames (default 243)
  - J: Number of joints (17 in H36M format) 
  - C: Coordinates + confidence (x, y, confidence)
- **Coordinate System**: Normalized to [-1, 1] range

## Keypoint Extraction Methods

### 1. Dummy (for testing)
```bash
--keypoint_method dummy
```
Generates random keypoints for testing the pipeline.

### 2. MediaPipe (recommended for quick setup)
```bash
--keypoint_method mediapipe
pip install mediapipe
```
Fast and easy to use, good accuracy for most use cases.

### 3. OpenPose/AlphaPose (highest accuracy)
For production use, implement OpenPose or AlphaPose extraction in the conversion tool.

## Configuration Options

### Training Parameters
- `epochs`: Number of training epochs (300 for scratch, 100 for finetune)
- `batch_size`: Batch size (adjust based on GPU memory)
- `lr_backbone`: Learning rate for backbone
- `lr_head`: Learning rate for classification head

### Model Parameters
- `action_classes`: Number of action classes in your dataset
- `clip_len`: Number of frames per video clip (243 default)
- `dropout_ratio`: Dropout rate for regularization

### Data Augmentation  
- `random_move`: Apply spatial augmentation (rotation, scaling, translation)
- `scale_range_train`: Scaling range during training
- `scale_range_test`: Scaling range during testing

## Performance Tips

### For Small Datasets (<1000 videos)
- Use finetuning from pretrained MotionBERT
- Lower learning rates (`lr_backbone: 0.00005`)
- Less aggressive augmentation
- Consider data augmentation techniques

### For Large Datasets (>10000 videos)
- Can train from scratch
- Higher learning rates
- More aggressive augmentation
- Use multiple GPUs if available

### Memory Optimization
- Reduce `batch_size` if running out of GPU memory
- Use gradient accumulation for effective larger batch sizes
- Consider preprocessing all keypoints to disk rather than video-based training

## Troubleshooting

### Common Issues

**"Dataset file not found"**
- Ensure the dataset pickle file exists in `data/action/`
- Check that dataset name in config matches the file name

**"CUDA out of memory"**
- Reduce batch size in config file
- Use gradient accumulation
- Close other GPU processes

**"Poor validation accuracy"**
- Check data quality and keypoint extraction
- Verify train/val splits don't have data leakage
- Try different learning rates or augmentation settings

**"Training loss not decreasing"**
- Check if keypoints are properly normalized
- Verify action class labels are correct
- Try lower learning rates

### Getting Help

1. Check the detailed documentation: `docs/custom_action_training.md`
2. Validate your dataset: use `--validate` flag in conversion tool
3. Start with a small subset of your data to test the pipeline
4. Use dummy keypoints first to ensure the training pipeline works

## Expected Results

With proper setup, you should see:
- Training loss steadily decreasing
- Validation accuracy improving over epochs
- Best model saved as `best_epoch.bin`
- TensorBoard logs in checkpoint directory

Typical training times:
- Small dataset (1000 videos): 2-4 hours on single GPU
- Medium dataset (10000 videos): 1-2 days on single GPU
- Large dataset (100000 videos): Several days, consider multi-GPU setup

## Next Steps

After successful training:
1. Use `infer_action.py` for inference on new videos
2. Fine-tune hyperparameters based on validation results
3. Consider ensemble methods for better performance
4. Deploy the model for real-time inference
