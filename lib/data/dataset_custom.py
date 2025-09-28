"""
Custom Dataset Implementation for Video-based Action Recognition
Author: DriveBERT Custom Implementation
"""

import os
import pickle
import numpy as np
import cv2
import json
from collections import defaultdict
from lib.data.dataset_action import ActionDataset, random_move
from lib.utils.utils_data import crop_scale

class CustomActionDataset(ActionDataset):
    """
    Custom dataset class for video-based action recognition
    Extends the base ActionDataset to handle custom video datasets
    """
    
    def __init__(self, data_path, data_split, n_frames=243, random_move=True, scale_range=[1,1]):
        """
        Initialize custom action dataset
        
        Args:
            data_path: Path to the dataset pickle file
            data_split: Split name ('train', 'val', 'test')
            n_frames: Number of frames to sample
            random_move: Whether to apply random motion augmentation
            scale_range: Range for scaling augmentation
        """
        super(CustomActionDataset, self).__init__(data_path, data_split, n_frames, random_move, scale_range)
        
        # Ensure each sample has 'total_frames' key
        if hasattr(self, 'samples'):
            for sample in self.samples:
                if 'total_frames' not in sample:
                    if 'keypoint' in sample and isinstance(sample['keypoint'], (list, np.ndarray)):
                        sample['total_frames'] = len(sample['keypoint'])
                    elif 'frames' in sample and isinstance(sample['frames'], (list, np.ndarray)):
                        sample['total_frames'] = len(sample['frames'])
                    else:
                        sample['total_frames'] = n_frames
        # Add custom preprocessing if needed
        self.preprocess_custom_data()
    
    def preprocess_custom_data(self):
        """
        Add any custom preprocessing specific to your dataset
        """
        # Handle missing keypoints by interpolation
        self._interpolate_missing_keypoints()
        
        # Normalize keypoints if not already normalized
        self._normalize_keypoints()
        
        # Temporarily disable quality filtering for DrivenAct to debug
        # self._filter_low_quality_samples()
        print(f"Dataset loaded: {len(self.motions)} samples (quality filtering disabled)")
    
    def _interpolate_missing_keypoints(self):
        """
        Interpolate missing keypoints (marked with confidence = 0)
        """
        for i, motion in enumerate(self.motions):
            M, T, J, C = motion.shape
            for m in range(M):
                for j in range(J):
                    # Get confidence scores (last channel)
                    if C > 2:  # Has confidence channel
                        confidence = motion[m, :, j, 2]
                        valid_frames = confidence > 0
                        
                        if np.any(valid_frames) and not np.all(valid_frames):
                            # Interpolate x and y coordinates
                            for coord in range(2):
                                valid_coords = motion[m, valid_frames, j, coord]
                                if len(valid_coords) > 1:
                                    # Simple linear interpolation
                                    motion[m, :, j, coord] = np.interp(
                                        range(T), 
                                        np.where(valid_frames)[0], 
                                        valid_coords
                                    )
    
    def _normalize_keypoints(self):
        """
        Ensure keypoints are normalized to camera coordinates [-1, 1]
        """
        for i, motion in enumerate(self.motions):
            # Check if already normalized (values between -1 and 1)
            if np.max(np.abs(motion[:, :, :, :2])) > 2:
                print(f"Warning: Sample {i} may need normalization")
    
    def _filter_low_quality_samples(self, min_confidence_ratio=0.1):
        """
        Filter out samples with too many missing keypoints
        More lenient for DrivenAct dataset which may have lower confidence scores
        """
        valid_indices = []
        
        for i, motion in enumerate(self.motions):
            if motion.shape[3] > 2:  # Has confidence channel
                confidence = motion[:, :, :, 2]
                valid_ratio = np.mean(confidence > 0)
                
                if valid_ratio >= min_confidence_ratio:
                    valid_indices.append(i)
                else:
                    print(f"Filtering out sample {i} (confidence ratio: {valid_ratio:.2f})")
            else:
                # If no confidence channel, keep all samples
                valid_indices.append(i)
        
        if len(valid_indices) < len(self.motions):
            # Apply filtering
            self.motions = [self.motions[i] for i in valid_indices]
            self.labels = [self.labels[i] for i in valid_indices]
            print(f"Filtered dataset: {len(valid_indices)}/{len(self.motions) + len(valid_indices)} samples kept")
    
    def __getitem__(self, idx):
        """
        Generate one sample of data
        
        Returns:
            motion: (M, T, J, C) tensor of keypoints
            label: Action class label
        """
        motion, label = self.motions[idx], self.labels[idx]  # (M,T,J,C)
        
        # Apply random motion augmentation during training
        if self.random_move:
            motion = random_move(motion)
        
        # Apply scaling augmentation
        if self.scale_range:
            result = crop_scale(motion, scale_range=self.scale_range)
        else:
            result = motion
            
        return result.astype(np.float32), label

class VideoActionDataset(CustomActionDataset):
    """
    Dataset class that handles raw video files and extracts keypoints on-the-fly
    Use this for smaller datasets or when you want to extract keypoints dynamically
    """
    
    def __init__(self, video_dir, annotation_file, keypoint_extractor='mediapipe', 
                 data_split='train', n_frames=243, random_move=True, scale_range=[1,1]):
        """
        Initialize video action dataset
        
        Args:
            video_dir: Directory containing video files
            annotation_file: JSON file with video annotations
            keypoint_extractor: Method to extract keypoints ('mediapipe', 'openpose', 'alphapose')
            data_split: Split name
            n_frames: Number of frames to sample
            random_move: Whether to apply augmentation
            scale_range: Scaling range for augmentation
        """
        self.video_dir = video_dir
        self.keypoint_extractor = keypoint_extractor
        self.n_frames = n_frames
        self.random_move = random_move and (data_split == 'train')
        self.scale_range = scale_range
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Filter by split
        if 'split' in self.annotations:
            split_samples = self.annotations['split'].get(data_split, [])
            self.samples = [sample for sample in self.annotations['samples'] 
                           if sample['video_name'] in split_samples]
        else:
            self.samples = self.annotations['samples']
        
        # Initialize keypoint extractor
        self._init_keypoint_extractor()
    
    def _init_keypoint_extractor(self):
        """Initialize the keypoint extraction method"""
        if self.keypoint_extractor == 'mediapipe':
            try:
                import mediapipe as mp
                self.mp_pose = mp.solutions.pose
                self.pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5
                )
            except ImportError:
                raise ImportError("MediaPipe not installed. Install with: pip install mediapipe")
        else:
            raise NotImplementedError(f"Keypoint extractor '{self.keypoint_extractor}' not implemented")
    
    def _extract_keypoints_mediapipe(self, video_path):
        """
        Extract keypoints from video using MediaPipe
        
        Returns:
            keypoints: (T, 33, 3) array - T frames, 33 landmarks, (x, y, visibility)
        """
        cap = cv2.VideoCapture(video_path)
        keypoints = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Extract landmark coordinates and visibility
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.visibility])
                keypoints.append(landmarks)
            else:
                # Add zero keypoints for frames without detection
                keypoints.append(np.zeros((33, 3)).tolist())
        
        cap.release()
        return np.array(keypoints)  # (T, 33, 3)
    
    def _mediapipe_to_h36m(self, mp_keypoints):
        """
        Convert MediaPipe keypoints to H36M format
        MediaPipe has 33 landmarks, we need to map to H36M's 17 joints
        """
        # MediaPipe pose landmark indices
        mp_to_h36m_map = {
            0: [11, 12],    # root (hip center) - average of left and right hip
            1: 24,          # rhip
            2: 26,          # rkne  
            3: 28,          # rank
            4: 23,          # lhip
            5: 25,          # lkne
            6: 27,          # lank
            7: [11, 12, 23, 24],  # belly - average of hips and shoulders
            8: [11, 12],    # neck - average of shoulders
            9: 0,           # nose
            10: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # head - average of face landmarks
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
        
        return h36m_keypoints
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        """
        sample = self.samples[idx]
        video_path = os.path.join(self.video_dir, sample['video_name'])
        label = sample['label']
        
        # Extract keypoints from video
        if self.keypoint_extractor == 'mediapipe':
            mp_keypoints = self._extract_keypoints_mediapipe(video_path)
            motion = self._mediapipe_to_h36m(mp_keypoints)
        else:
            raise NotImplementedError(f"Extractor {self.keypoint_extractor} not implemented")
        
        # Resample to target number of frames
        from lib.utils.utils_data import resample
        T = motion.shape[1]
        if T != self.n_frames:
            resample_idx = resample(T, self.n_frames, randomness=self.random_move)
            motion = motion[:, resample_idx, :, :]
        
        # Convert coordinates to camera coordinates (normalize to [-1, 1])
        motion[:, :, :, 0] = (motion[:, :, :, 0] - 0.5) * 2  # x coordinates
        motion[:, :, :, 1] = (motion[:, :, :, 1] - 0.5) * 2  # y coordinates
        # Keep visibility as confidence score
        
        # Ensure we have exactly 2 people (pad with zeros if needed)
        if motion.shape[0] == 1:
            fake_person = np.zeros_like(motion)
            motion = np.concatenate([motion, fake_person], axis=0)
        
        # Apply augmentations
        if self.random_move:
            motion = random_move(motion)
        
        if self.scale_range:
            motion = crop_scale(motion, scale_range=self.scale_range)
        
        return motion.astype(np.float32), label


def create_action_names_file(action_classes, output_path):
    """
    Create action names file compatible with DriveBERT format
    
    Args:
        action_classes: List of action class names
        output_path: Path to save the action names file
    """
    with open(output_path, 'w') as f:
        for i, action_name in enumerate(action_classes):
            f.write(f"{i}. {action_name}\n")
    
    print(f"Action names file saved to {output_path}")
