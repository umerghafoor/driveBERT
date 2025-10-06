"""
Test script to verify DrivenAct dataset structure and basic functionality
"""

import os
import pandas as pd
import cv2

def test_dataset_structure():
    """Test if the dataset structure is as expected"""
    
    # Check dataset paths
    dataset_root = '/mnt/1C00FF7F00FF5DE8/Users/datasets/Drivenact'
    csv_file = '/mnt/1C00FF7F00FF5DE8/Users/datasets/Drivenact/activities_3s/kinect_color/midlevel.chunks_90.split_0.train.csv'
    video_dir = os.path.join(dataset_root, 'a_column_co_driver')
    
    print("Testing DrivenAct dataset structure...")
    print(f"Dataset root: {dataset_root}")
    print(f"Dataset root exists: {os.path.exists(dataset_root)}")
    print(f"CSV file: {csv_file}")
    print(f"CSV file exists: {os.path.exists(csv_file)}")
    print(f"Video directory: {video_dir}")
    print(f"Video directory exists: {os.path.exists(video_dir)}")
    
    if os.path.exists(csv_file):
        # Load CSV and check structure
        df = pd.read_csv(csv_file)
        print(f"\nCSV file loaded successfully!")
        print(f"Number of samples: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print(f"Sample row:")
        print(df.iloc[0])
        
        # Check unique activities
        activities = df['activity'].unique()
        print(f"\nNumber of unique activities: {len(activities)}")
        print(f"Sample activities: {activities[:10]}")
        
        # Test a video file
        if len(df) > 0:
            row = df.iloc[0]
            video_path = os.path.join(
                dataset_root, 'a_column_co_driver',
                f"vp{row['participant_id']}", 
                f"{row['file_id']}.mp4"
            )
            print(f"\nTesting video file: {video_path}")
            print(f"Video file exists: {os.path.exists(video_path)}")
            
            if os.path.exists(video_path):
                # Try to open video
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    print(f"Video opened successfully!")
                    print(f"Total frames: {frame_count}")
                    print(f"FPS: {fps}")
                    print(f"Duration: {frame_count/fps:.2f} seconds")
                    
                    # Test frame extraction
                    cap.set(cv2.CAP_PROP_POS_FRAMES, row['frame_start'])
                    ret, frame = cap.read()
                    if ret:
                        print(f"Frame extraction successful!")
                        print(f"Frame shape: {frame.shape}")
                    else:
                        print("Frame extraction failed!")
                    
                    cap.release()
                else:
                    print("Could not open video file!")
    else:
        print("CSV file not found!")

if __name__ == "__main__":
    test_dataset_structure()