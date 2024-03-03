import cv2
import os
import argparse
from tqdm import tqdm
import numpy as np
import json
import re

"""
This module is designed to reformat the Multiview video dataset into the LLFF data format for NeRF's variants. 
The purpose of this transformation is to make the NV3D/MPEG dataset compatible with the ReRF model.

Author: jhpark
"""

parser = argparse.ArgumentParser(description="Convert multiview video dataset to LLFF format")
parser.add_argument("--video_path", type=str, required=True, help="Path to the NV3D dataset directory")
parser.add_argument("--llff_path", type=str, required=True, help="Path to the output LLFF dataset directory")
parser.add_argument("--num_frames", type=int, default=200, help="Number of frames to extract from each video")

args = parser.parse_args()

video_path = args.video_path
llff_path = args.llff_path
num_frames = args.num_frames

def create_bbox(llff_dataset_path):
    bbox = {
        "xyz_min": [-1.5, -1.5, -1.0],
        "xyz_max": [1.5, 1.5, 1.0]
    }

    with open(os.path.join(llff_dataset_path, 'bbox.json'), 'w') as json_file:
        json.dump(bbox, json_file, indent=4)


def extract_number(filename):
    """Extracts the number from the filename."""
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0  # default if no number is found

def convert_nv3d_to_llff(video_dataset_path, llff_dataset_path, num_frames):
    # Load poses and bounds
    poses_bounds = np.load(os.path.join(video_dataset_path, 'poses_bounds.npy'))

    # Get the list of video files and sort them based on camera number
    video_files = sorted(os.listdir(video_dataset_path), key=extract_number)

    # Filter out non-MP4 files
    video_files = [f for f in video_files if f.endswith('.mp4')]

    # For each video file (camera view)
    for view_id, video_file in enumerate(video_files):

        # Open the video
        video = cv2.VideoCapture(os.path.join(video_dataset_path, video_file))

        # Seek to the correct frame
        #video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        # For each frame to be extracted
        for frame_id in tqdm(range(num_frames), desc="Processing frames"):
            # Create directories for the frame
            frame_dir = os.path.join(llff_dataset_path, str(frame_id))
            os.makedirs(frame_dir, exist_ok=True)
            image_dir = os.path.join(frame_dir, 'images')
            os.makedirs(image_dir, exist_ok=True)

       
        
            
            

            
            # Check if the current frame position is correct
            #current_frame_id = int(video.get(cv2.CAP_PROP_POS_FRAMES))
            #if current_frame_id != frame_id:
            #    print(f"Warning: Skipping frame {frame_id} in video {video_file}. Could not seek to the correct position.")
            #    continue
            
            # Read the frame
            success, image = video.read()
            
            if success:
                # Save the frame as an image
                cv2.imwrite(os.path.join(image_dir, f'image_{str(view_id).zfill(4)}.jpg'), image, [cv2.IMWRITE_PNG_COMPRESSION, 2])

        # Save the corresponding poses and bounds
            np.save(os.path.join(frame_dir, 'poses_bounds.npy'), poses_bounds)
convert_nv3d_to_llff(video_path, llff_path, num_frames)
create_bbox(llff_path)