"""
Driving Behavior Dataset Converter for DriveBERT Action Recognition
Specialized converter for automotive behavior datasets with temporal annotations
"""

import os
import json
import pickle
import numpy as np
import cv2
import pandas as pd
import argparse
from tqdm import tqdm
from collections import defaultdict
import sys

# Add the lib directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

class DrivingBehaviorConverter:
    """
    Converter for driving behavior datasets with temporal annotations
    """
    
    def __init__(self):
        # Define activity mapping from your dataset
        self.activity_list = [
            'sitting_still',
            'eating', 
            'fetching_an_object',
            'placing_an_object',
            'reading_magazine',
            'reading_newspaper',
            'using_multimedia_display',
            'interacting_with_phone',
            'working_on_laptop',
            'talking_on_phone',
            'writing',
            'pressing_automation_button',
            'putting_on_jacket',
            'fastening_seat_belt',
            'drinking',
            'taking_off_jacket',
            'opening_bottle',
            'looking_or_moving_around (e.g. searching)',
            'closing_bottle',
            'unfastening_seat_belt',
            'putting_on_sunglasses',
            'taking_off_sunglasses',
            'preparing_food',
            'opening_laptop',
            'exiting_car',
            'entering_car',
            'closing_laptop',
            'opening_door_inside',
            'opening_door_outside',
            'closing_door_inside',
            'opening_backpack',
            'putting_laptop_into_backpack',
            'closing_door_outside',
            'taking_laptop_from_backpack',
            'looking_back_left_shoulder',
            'closing_backpack',
            'moving_towards_door',
            'standing_by_the_door',
            'looking_back_right_shoulder'
        ]
        
        self.activity_to_id = {activity: idx for idx, activity in enumerate(self.activity_list)}
        self.num_classes = len(self.activity_list)
    
    def load_annotations(self, csv_path):
        """Load annotations from CSV file"""
        df = pd.read_csv(csv_path)
        return df
    
    def load_timestamps(self, timestamp_path):
        """Load timestamps from file"""
        with open(timestamp_path, 'r') as f:
            timestamps = [float(line.strip()) for line in f.readlines()]
        return np.array(timestamps)
    
    def load_calibration(self, calibration_path):
        """Load camera calibration data"""
        with open(calibration_path, 'r') as f:
            calibration = json.load(f)
        return calibration
    
    def extract_segments(self, df, min_duration_frames=30, max_duration_frames=243):
        """
        Extract video segments with consistent activities
        
        Args:
            df: DataFrame with annotations
            min_duration_frames: Minimum segment duration in frames
            max_duration_frames: Maximum segment duration in frames
        """
        segments = []
        
        for _, row in df.iterrows():
            frame_start = int(row['frame_start'])
            frame_end = int(row['frame_end'])
            duration = frame_end - frame_start
            
            if duration < min_duration_frames:
                continue  # Skip too short segments
            
            if duration <= max_duration_frames:
                # Single segment
                segments.append({
                    'participant_id': row['participant_id'],
                    'file_id': row['file_id'],
                    'annotation_id': row['annotation_id'],
                    'frame_start': frame_start,
                    'frame_end': frame_end,
                    'activity': row['activity'],
                    'activity_id': self.activity_to_id.get(row['activity'], 0),
                    'chunk_id': row['chunk_id'],
                    'duration': duration
                })
            else:
                # Split long segments into chunks
                num_chunks = (duration + max_duration_frames - 1) // max_duration_frames
                chunk_size = duration // num_chunks
                
                for i in range(num_chunks):
                    chunk_start = frame_start + i * chunk_size
                    chunk_end = frame_start + (i + 1) * chunk_size
                    if i == num_chunks - 1:  # Last chunk gets remaining frames
                        chunk_end = frame_end
                    
                    if chunk_end - chunk_start >= min_duration_frames:
                        segments.append({
                            'participant_id': row['participant_id'],
                            'file_id': row['file_id'],
                            'annotation_id': f"{row['annotation_id']}_chunk_{i}",
                            'frame_start': chunk_start,
                            'frame_end': chunk_end,
                            'activity': row['activity'],
                            'activity_id': self.activity_to_id.get(row['activity'], 0),
                            'chunk_id': f"{row['chunk_id']}_{i}",
                            'duration': chunk_end - chunk_start
                        })
        
        return segments
    
    def extract_keypoints_from_video(self, video_path, frame_start, frame_end, target_frames=243):
        """
        Extract keypoints from video segment using MediaPipe
        """
        try:
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
        except ImportError:
            print("MediaPipe not available, using dummy keypoints")
            return self.create_dummy_keypoints(target_frames)
        
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}, using dummy keypoints")
            return self.create_dummy_keypoints(target_frames)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}, using dummy keypoints")
            return self.create_dummy_keypoints(target_frames)
        
        # Set video to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        
        keypoints = []
        frame_count = 0
        target_duration = frame_end - frame_start
        
        while frame_count < target_duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Extract landmark coordinates and visibility
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    # Convert to normalized coordinates [-1, 1]
                    x = (lm.x - 0.5) * 2
                    y = (lm.y - 0.5) * 2
                    landmarks.append([x, y, lm.visibility])
                keypoints.append(landmarks)
            else:
                # Add zero keypoints for frames without detection
                keypoints.append(np.zeros((33, 3)).tolist())
            
            frame_count += 1
        
        cap.release()
        
        if len(keypoints) == 0:
            return self.create_dummy_keypoints(target_frames)
        
        keypoints = np.array(keypoints)  # (T, 33, 3)
        
        # Convert MediaPipe to H36M format
        h36m_keypoints = self.mediapipe_to_h36m(keypoints)
        
        # Resample to target frames
        if len(h36m_keypoints[0]) != target_frames:
            indices = np.linspace(0, len(h36m_keypoints[0])-1, target_frames, dtype=int)
            h36m_keypoints = h36m_keypoints[:, indices, :, :]
        
        return h36m_keypoints
    
    def create_dummy_keypoints(self, num_frames, num_joints=17):
        """Create dummy keypoints for testing"""
        # Create more realistic dummy keypoints for driving scenarios
        keypoints = np.random.rand(1, num_frames, num_joints, 3) * 0.4 - 0.2  # Smaller range for sitting
        keypoints[:, :, :, 2] = 0.8  # Set confidence to 0.8
        
        # Ensure we have 2 people (pad with zeros for second person)
        fake_person = np.zeros_like(keypoints)
        keypoints = np.concatenate([keypoints, fake_person], axis=0)
        
        return keypoints
    
    def mediapipe_to_h36m(self, mp_keypoints):
        """Convert MediaPipe keypoints to H36M format"""
        # MediaPipe pose landmark indices mapping to H36M
        mp_to_h36m_map = {
            0: [11, 12],    # root (hip center)
            1: 24,          # rhip
            2: 26,          # rkne  
            3: 28,          # rank
            4: 23,          # lhip
            5: 25,          # lkne
            6: 27,          # lank
            7: [11, 12, 23, 24],  # belly
            8: [11, 12],    # neck
            9: 0,           # nose
            10: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # head
            11: 11,         # lsho
            12: 13,         # lelb
            13: 15,         # lwri
            14: 12,         # rsho
            15: 14,         # relb
            16: 16          # rwri
        }
        
        T = mp_keypoints.shape[0]
        h36m_keypoints = np.zeros((1, T, 17, 3))  # Single person format
        
        for h36m_idx, mp_idx in mp_to_h36m_map.items():
            if isinstance(mp_idx, list):
                # Average multiple MediaPipe points
                coords = mp_keypoints[:, mp_idx, :]
                h36m_keypoints[0, :, h36m_idx, :] = np.mean(coords, axis=1)
            else:
                # Direct mapping
                h36m_keypoints[0, :, h36m_idx, :] = mp_keypoints[:, mp_idx, :]
        
        # Add second person (zeros)
        fake_person = np.zeros_like(h36m_keypoints)
        h36m_keypoints = np.concatenate([h36m_keypoints, fake_person], axis=0)
        
        return h36m_keypoints
    
    def create_splits(self, segments, split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15}):
        """Create train/val/test splits stratified by activity"""
        np.random.seed(42)
        
        # Group segments by activity for stratified splitting
        activity_segments = defaultdict(list)
        for segment in segments:
            activity_segments[segment['activity']].append(segment)
        
        splits = {'train': [], 'val': [], 'test': []}
        
        for activity, activity_segs in activity_segments.items():
            np.random.shuffle(activity_segs)
            n_segments = len(activity_segs)
            
            n_train = int(n_segments * split_ratios['train'])
            n_val = int(n_segments * split_ratios['val'])
            
            train_segs = activity_segs[:n_train]
            val_segs = activity_segs[n_train:n_train+n_val]
            test_segs = activity_segs[n_train+n_val:]
            
            splits['train'].extend([f"{seg['file_id']}_{seg['annotation_id']}" for seg in train_segs])
            splits['val'].extend([f"{seg['file_id']}_{seg['annotation_id']}" for seg in val_segs])
            splits['test'].extend([f"{seg['file_id']}_{seg['annotation_id']}" for seg in test_segs])
        
        return splits, {'train': [], 'val': [], 'test': []}  # Return empty splits for segments
    
    def convert_dataset(self, csv_path, video_dir, output_path, 
                       use_dummy_keypoints=True, target_frames=243):
        """
        Convert driving behavior dataset to DriveBERT format
        """
        print("Loading annotations...")
        df = self.load_annotations(csv_path)
        
        print(f"Found {len(df)} annotations with {self.num_classes} activity classes")
        
        # Extract segments
        print("Extracting segments...")
        segments = self.extract_segments(df, min_duration_frames=30, max_duration_frames=target_frames)
        
        print(f"Created {len(segments)} segments")
        
        # Create splits
        splits, _ = self.create_splits(segments)
        
        # Process segments
        print("Processing segments...")
        annotations = []
        
        for segment in tqdm(segments):
            segment_id = f"{segment['file_id']}_{segment['annotation_id']}"
            
            if use_dummy_keypoints:
                # Use dummy keypoints for quick testing
                keypoints = self.create_dummy_keypoints(target_frames)
            else:
                # Extract from actual video (implement video loading logic)
                video_path = os.path.join(video_dir, f"{segment['file_id']}.mp4")  # Adjust extension as needed
                keypoints = self.extract_keypoints_from_video(
                    video_path, segment['frame_start'], segment['frame_end'], target_frames
                )
            
            # Get video properties (use dummy values if video not available)
            img_height, img_width = 1024, 1280  # From calibration data
            
            annotation = {
                'frame_dir': segment_id,
                'total_frames': segment['duration'],
                'img_shape': (img_height, img_width),
                'keypoint': keypoints[:, :, :, :2],  # (M, T, J, 2) - x,y coordinates
                'keypoint_score': keypoints[:, :, :, 2],  # (M, T, J) - confidence scores
                'label': segment['activity_id'],
                'activity_name': segment['activity'],
                'participant_id': segment['participant_id'],
                'original_frame_start': segment['frame_start'],
                'original_frame_end': segment['frame_end']
            }
            
            annotations.append(annotation)
        
        # Create final dataset structure
        dataset = {
            'split': splits,
            'annotations': annotations,
            'activity_names': self.activity_list,
            'num_classes': self.num_classes
        }
        
        # Save dataset
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"Dataset saved to {output_path}")
        print(f"Dataset statistics:")
        print(f"  Total segments: {len(annotations)}")
        print(f"  Activity classes: {self.num_classes}")
        for split_name, split_files in splits.items():
            print(f"  {split_name}: {len(split_files)} segments")
        
        # Save activity names file
        action_names_path = output_path.replace('.pkl', '_actions.txt')
        with open(action_names_path, 'w') as f:
            for i, activity_name in enumerate(self.activity_list):
                f.write(f"{i}. {activity_name}\n")
        print(f"Activity names saved to {action_names_path}")
        
        return dataset

def main():
    parser = argparse.ArgumentParser(description='Convert driving behavior dataset to DriveBERT format')
    parser.add_argument('--csv_path', required=True, 
                       help='Path to CSV file with annotations (e.g., midlevel.chunks_90.csv)')
    parser.add_argument('--video_dir', default=None,
                       help='Directory containing video files')
    parser.add_argument('--output_path', required=True,
                       help='Output pickle file path')
    parser.add_argument('--use_dummy_keypoints', action='store_true', default=True,
                       help='Use dummy keypoints for testing (default: True)')
    parser.add_argument('--target_frames', type=int, default=243,
                       help='Target number of frames per segment')
    parser.add_argument('--validate', action='store_true',
                       help='Validate the created dataset')
    
    args = parser.parse_args()
    
    # Create converter
    converter = DrivingBehaviorConverter()
    
    # Convert dataset
    dataset = converter.convert_dataset(
        csv_path=args.csv_path,
        video_dir=args.video_dir,
        output_path=args.output_path,
        use_dummy_keypoints=args.use_dummy_keypoints,
        target_frames=args.target_frames
    )
    
    # Validate if requested
    if args.validate:
        validate_dataset(args.output_path)

def validate_dataset(dataset_path):
    """Validate the created dataset"""
    print(f"Validating dataset: {dataset_path}")
    
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    annotations = data['annotations']
    splits = data['split']
    activity_names = data['activity_names']
    
    print(f"Dataset contains {len(annotations)} segments")
    print(f"Activity classes: {len(activity_names)}")
    print(f"Splits: {list(splits.keys())}")
    
    # Show activity distribution
    activity_counts = defaultdict(int)
    for annotation in annotations:
        activity_name = annotation['activity_name']
        activity_counts[activity_name] += 1
    
    print("\nActivity distribution:")
    for activity, count in sorted(activity_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {activity}: {count}")
    
    # Validate data structure
    for i, sample in enumerate(annotations[:3]):
        keypoints = sample['keypoint']
        scores = sample['keypoint_score']
        
        print(f"\nSample {i}:")
        print(f"  Activity: {sample['activity_name']}")
        print(f"  Keypoints shape: {keypoints.shape}")
        print(f"  Scores shape: {scores.shape}")
        print(f"  Label: {sample['label']}")
        print(f"  Duration: {sample['total_frames']} frames")
        print(f"  Participant: {sample['participant_id']}")
        
        # Validate shapes
        assert len(keypoints.shape) == 4, f"Invalid keypoint shape: {keypoints.shape}"
        assert keypoints.shape[2] == 17, f"Expected 17 joints, got {keypoints.shape[2]}"
        assert keypoints.shape[3] == 2, f"Expected 2 coordinates, got {keypoints.shape[3]}"
    
    print("\nValidation completed successfully!")

if __name__ == "__main__":
    main()
