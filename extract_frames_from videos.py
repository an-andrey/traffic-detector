import cv2
import os
from pathlib import Path

# Configuration
video_folder = "data/videos"  # Change to your video folder path
output_folder = "data/extracted_frames"  # Where to save frames

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get all video files
video_files = sorted([f for f in os.listdir(video_folder)])

print(f"Found {len(video_files)} videos to process")

# Process each video
for video_idx, video_file in enumerate(video_files):
    video_path = os.path.join(video_folder, video_file)
    
    # Get video name without extension (e.g., "0000" from "0000.mp4")
    video_name = Path(video_file).stem
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video: {video_file}")
        continue
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Create filename: video_name-frame_number.jpg
        frame_filename = f"{video_name}-{frame_count}.jpg"
        frame_path = os.path.join(output_folder, frame_filename)
        
        # Save frame
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    
    cap.release()
    
    print(f"✅ Processed {video_file}: {frame_count} frames extracted ({video_idx+1}/{len(video_files)})")

print(f"Done! Total videos processed: {len(video_files)}")
print(f"Frames saved to: {output_folder}/")
