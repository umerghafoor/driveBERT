import torch
import numpy as np
import os
import random
import copy
from torch.utils.data import Dataset, DataLoader
from lib.utils.utils_data import crop_scale, resample
from lib.utils.tools import read_pkl
    
def get_action_names(file_path = "data/action/ntu_actions.txt"):
    f = open(file_path, "r")
    s = f.read()
    actions = s.split('\n')
    action_names = []
    for a in actions:
        action_names.append(a.split('.')[1][1:])
    return action_names

def make_cam(x, img_shape):
    '''
        Input: x (M x T x V x C)
               img_shape (height, width)
    '''
    h, w = img_shape
    if w >= h:
        x_cam = x / w * 2 - 1
    else:
        x_cam = x / h * 2 - 1
    return x_cam

def coco2h36m(x):
    '''
        Input: x (M x T x V x C)
        
        COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
        
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    y = np.zeros(x.shape)
    y[:,:,0,:] = (x[:,:,11,:] + x[:,:,12,:]) * 0.5
    y[:,:,1,:] = x[:,:,12,:]
    y[:,:,2,:] = x[:,:,14,:]
    y[:,:,3,:] = x[:,:,16,:]
    y[:,:,4,:] = x[:,:,11,:]
    y[:,:,5,:] = x[:,:,13,:]
    y[:,:,6,:] = x[:,:,15,:]
    y[:,:,8,:] = (x[:,:,5,:] + x[:,:,6,:]) * 0.5
    y[:,:,7,:] = (y[:,:,0,:] + y[:,:,8,:]) * 0.5
    y[:,:,9,:] = x[:,:,0,:]
    y[:,:,10,:] = (x[:,:,1,:] + x[:,:,2,:]) * 0.5
    y[:,:,11,:] = x[:,:,5,:]
    y[:,:,12,:] = x[:,:,7,:]
    y[:,:,13,:] = x[:,:,9,:]
    y[:,:,14,:] = x[:,:,6,:]
    y[:,:,15,:] = x[:,:,8,:]
    y[:,:,16,:] = x[:,:,10,:]
    return y
    
def random_move(data_numpy,
                angle_range=[-10., 10.],
                scale_range=[0.9, 1.1],
                transform_range=[-0.1, 0.1],
                move_time_candidate=[1]):
    data_numpy = np.transpose(data_numpy, (3,1,2,0)) # M,T,V,C-> C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)
    A = np.random.uniform(angle_range[0], angle_range[1], num_node)
    S = np.random.uniform(scale_range[0], scale_range[1], num_node)
    T_x = np.random.uniform(transform_range[0], transform_range[1], num_node)
    T_y = np.random.uniform(transform_range[0], transform_range[1], num_node)
    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)
    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1], node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1], node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1], node[i + 1] - node[i])
    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])
    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)
    data_numpy = np.transpose(data_numpy, (3,1,2,0)) # C,T,V,M -> M,T,V,C
    return data_numpy    

def human_tracking(x):
    M, T = x.shape[:2]
    if M==1:
        return x
    else:
        diff0 = np.sum(np.linalg.norm(x[0,1:] - x[0,:-1], axis=-1), axis=-1)        # (T-1, V, C) -> (T-1)
        diff1 = np.sum(np.linalg.norm(x[0,1:] - x[1,:-1], axis=-1), axis=-1)
        x_new = np.zeros(x.shape)
        sel = np.cumsum(diff0 > diff1) % 2
        sel = sel[:,None,None]
        x_new[0][0] = x[0][0]
        x_new[1][0] = x[1][0]
        x_new[0,1:] = x[1,1:] * sel + x[0,1:] * (1-sel)
        x_new[1,1:] = x[0,1:] * sel + x[1,1:] * (1-sel)
        return x_new

class ActionDataset(Dataset):
    def __init__(self, data_path, data_split, n_frames=243, random_move=True, scale_range=[1,1], check_split=True):   # data_split: train/test etc.
        np.random.seed(0)
        dataset = read_pkl(data_path)
        if check_split:
            assert data_split in dataset['split'].keys()
            self.split = dataset['split'][data_split]
        annotations = dataset['annotations']
        # Ensure each sample has 'total_frames' key
        for sample in annotations:
            if 'total_frames' not in sample:
                if 'keypoint' in sample and isinstance(sample['keypoint'], np.ndarray):
                    sample['total_frames'] = sample['keypoint'].shape[0]
                else:
                    sample['total_frames'] = n_frames
        self.random_move = random_move
        self.is_train = "train" in data_split or (check_split==False)
        if "oneshot" in data_split:
            self.is_train = False
        self.scale_range = scale_range
        motions = []
        labels = []
        for sample in annotations:
            if check_split and (not sample['frame_dir'] in self.split):
                continue
            resample_id = resample(ori_len=sample['total_frames'], target_len=n_frames, randomness=self.is_train)
            motion_cam = make_cam(x=sample['keypoint'], img_shape=sample['img_shape'])
            motion_cam = human_tracking(motion_cam)
            motion_cam = coco2h36m(motion_cam)
            motion_conf = sample['keypoint_score'][..., None]
            motion = np.concatenate((motion_cam[:,resample_id], motion_conf[:,resample_id]), axis=-1)
            if motion.shape[0]==1:                                  # Single person, make a fake zero person
                fake = np.zeros(motion.shape)
                motion = np.concatenate((motion, fake), axis=0)
            motions.append(motion.astype(np.float32)) 
            labels.append(sample['label'])
        self.motions = np.array(motions)
        self.labels = np.array(labels)
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.motions)

    def __getitem__(self, index):
        raise NotImplementedError 

class NTURGBD(ActionDataset):
    def __init__(self, data_path, data_split, n_frames=243, random_move=True, scale_range=[1,1]):
        super(NTURGBD, self).__init__(data_path, data_split, n_frames, random_move, scale_range)
        
    def __getitem__(self, idx):
        'Generates one sample of data'
        motion, label = self.motions[idx], self.labels[idx] # (M,T,J,C)
        if self.random_move:
            motion = random_move(motion)
        if self.scale_range:
            result = crop_scale(motion, scale_range=self.scale_range)
        else:
            result = motion
        return result.astype(np.float32), label
    
class NTURGBD1Shot(ActionDataset):
    def __init__(self, data_path, data_split, n_frames=243, random_move=True, scale_range=[1,1], check_split=False):
        super(NTURGBD1Shot, self).__init__(data_path, data_split, n_frames, random_move, scale_range, check_split)
        oneshot_classes = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114]
        new_classes = set(range(120)) - set(oneshot_classes)
        old2new = {}
        for i, cid in enumerate(new_classes):
            old2new[cid] = i
        filtered = [not (x in oneshot_classes) for x in self.labels]
        self.motions = self.motions[filtered]
        filtered_labels = self.labels[filtered]
        self.labels = [old2new[x] for x in filtered_labels]
        
    def __getitem__(self, idx):
        'Generates one sample of data'
        motion, label = self.motions[idx], self.labels[idx] # (M,T,J,C)
        if self.random_move:
            motion = random_move(motion)
        if self.scale_range:
            result = crop_scale(motion, scale_range=self.scale_range)
        else:
            result = motion
        return result.astype(np.float32), label

class DrivenAct(Dataset):
    """
    DrivenAct dataset loader for driving behavior recognition
    """
    def __init__(self, data_path, data_split, n_frames=90, random_move=True, scale_range=[1,1]):
        """
        Args:
            data_path: Path to DrivenAct pickle file
            data_split: 'train', 'val', or 'test'
            n_frames: Number of frames per clip (90 for DrivenAct)
            random_move: Whether to apply random movement augmentation
            scale_range: Scale augmentation range
        """
        self.data_path = data_path
        self.data_split = data_split
        self.n_frames = n_frames
        self.random_move = random_move
        self.scale_range = scale_range
        
        # Load DrivenAct data
        print(f"Loading DrivenAct data from: {data_path}")
        data = read_pkl(data_path)
        
        # Extract samples for the specified split
        if data_split in data:
            split_data = data[data_split]
        else:
            raise ValueError(f"Split '{data_split}' not found in data. Available: {list(data.keys())}")
        
        print(f"Processing {len(split_data)} samples for {data_split} split")
        
        motions = []
        labels = []
        
        # Build activity mapping from actions file
        activity_mapping = self._build_activity_mapping(data_path.replace('.pkl', '_actions.txt'))
        
        for sample in split_data:
            # Get keypoints and action
            keypoints = sample['keypoints']  # (M, T, J, 2) format from our converter
            action = sample['action']
            
            # Convert action name to label index
            if action in activity_mapping:
                label = activity_mapping[action]
            else:
                print(f"Warning: Unknown activity '{action}', skipping sample")
                continue
            
            # Add confidence channel (using sample confidence if available)
            if 'confidence' in sample:
                confidence = sample['confidence']  # (T, J)
                confidence = confidence[np.newaxis, :, :, np.newaxis]  # (1, T, J, 1)
            else:
                # Create default confidence of 0.8 for all keypoints
                M, T, J, C = keypoints.shape
                confidence = np.full((M, T, J, 1), 0.8, dtype=np.float32)
            
            # Combine keypoints and confidence: (M, T, J, 3) -> x, y, confidence
            motion = np.concatenate([keypoints, confidence], axis=-1)
            
            # Ensure we have exactly n_frames
            if motion.shape[1] != n_frames:
                # Resample to target frames
                old_frames = motion.shape[1]
                indices = np.linspace(0, old_frames - 1, n_frames)
                resampled_motion = np.zeros((motion.shape[0], n_frames, motion.shape[2], motion.shape[3]))
                for m in range(motion.shape[0]):
                    for j in range(motion.shape[2]):
                        for c in range(motion.shape[3]):
                            resampled_motion[m, :, j, c] = np.interp(
                                indices, np.arange(old_frames), motion[m, :, j, c]
                            )
                motion = resampled_motion
            
            motions.append(motion.astype(np.float32))
            labels.append(label)
        
        self.motions = np.array(motions)
        self.labels = np.array(labels)
        
        print(f"Loaded {len(self.motions)} samples, {len(np.unique(self.labels))} unique classes")
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.motions)
    
    def _build_activity_mapping(self, actions_file):
        """Build activity name to index mapping from actions file"""
        activity_mapping = {}
        if os.path.exists(actions_file):
            with open(actions_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '. ' in line:
                        idx_str, activity = line.split('. ', 1)
                        activity_mapping[activity] = int(idx_str)
        else:
            print(f"Warning: Actions file not found: {actions_file}")
        return activity_mapping
    
    def __getitem__(self, idx):
        'Generates one sample of data'
        motion, label = self.motions[idx], self.labels[idx] # (M,T,J,C)
        if self.random_move:
            motion = random_move(motion)
        if self.scale_range:
            result = crop_scale(motion, scale_range=self.scale_range)
        else:
            result = motion
        return result.astype(np.float32), label