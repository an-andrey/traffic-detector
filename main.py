import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import cv2
import argparse
import cv2, json
from google import genai
from google.genai import types

# --- 1. UTILITY: NUMPY MODE & TEMPORAL SMOOTHING FUNCTIONS ---

def numpy_mode(arr):
    """
    Calculates the mode of a 1D NumPy array using bincount.
    Assumes array contains non-negative integers (0s and 1s).
    """
    if len(arr) == 0:
        return 0 # Default to 'No Crash' if array is empty
    
    # Count occurrences of each number (0 and 1)
    counts = np.bincount(arr)
    
    # The mode is the index with the highest count.
    # np.argmax finds the index (0 or 1) of the largest count.
    return np.argmax(counts)

def smooth_predictions(raw_predictions, window_size=9):
    """
    Applies a sliding window mode filter (temporal smoothing) to reduce flicker.
    Uses the stable numpy_mode function.
    """
    if window_size % 2 == 0:
        window_size += 1 # Ensure odd size
        
    predictions = np.array(raw_predictions) # Convert to NumPy array
    
    # Check for empty array
    if len(predictions) == 0:
        return []
    
    # If not enough predictions for a full window, return the list version
    if len(predictions) < window_size:
        return predictions.tolist()
        
    smoothed_predictions = predictions.copy()
    padding = window_size // 2
    
    # Initialize the first few predictions (padding at the start)
    for i in range(padding):
         # Use the mode of the available starting segment
         smoothed_predictions[i] = numpy_mode(predictions[:i + padding + 1])

    # Iterate through the sequence where a full window is possible
    for i in range(padding, len(predictions) - padding):
        start = i - padding
        end = i + padding + 1
        window = predictions[start:end]
        smoothed_predictions[i] = numpy_mode(window)

    # Fill the last few predictions (padding at the end)
    for i in range(len(predictions) - padding, len(predictions)):
         # Use the mode of the available ending segment
         smoothed_predictions[i] = numpy_mode(predictions[i - padding:])
        
    return smoothed_predictions.tolist() # Final return as a list

# --- 2. MAIN EXECUTION PIPELINE ---
def main(video_path, model_path, fps_sampling=10, smooth_window=5):
    
    # Setup Device (Checks for local GPU, defaults to CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_loc = 'cuda' if device.type == 'cuda' else 'cpu'
    print(f"Using device: {device}")

    # --- Load Model Structure and Weights ---
    model = resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    # Load the state dictionary, mapping to the appropriate device
    model.load_state_dict(torch.load(model_path, map_location=map_loc))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}.")

    # --- Define Transforms (MUST match training transforms) ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # --- Initialize Video Capture ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {video_path}")
        return

    # Calculate frame sampling interval
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0: 
         original_fps = 30 # Default if reading fails
    frame_interval = max(1, int(round(original_fps / fps_sampling)))
    
    # Prediction tracking
    raw_predictions = []
    
    print(f"Original FPS: {original_fps:.2f}, Sampling every {frame_interval} frames to achieve ~{fps_sampling} FPS.")

    # --- Start Video Processing Loop ---
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        is_sampled_frame = (frame_idx % frame_interval == 0)
        
        # --- Prediction Logic ---
        current_prediction = 0 # Default display state
        if is_sampled_frame:
            # 1. Preprocess: BGR (OpenCV) to RGB (PIL/Torch)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = transform(image).unsqueeze(0).to(device)

            # 2. Predict
            with torch.no_grad():
                output = model(image_tensor)
                _, pred = torch.max(output, 1)
                raw_predictions.append(pred.item())
            
            # 3. Smooth the prediction based on the history
            smoothed = smooth_predictions(raw_predictions, window_size=smooth_window)
            if smoothed:
                 current_prediction = smoothed[-1]
        
        # --- Visualization Logic ---
        
        # Only use the latest prediction if it exists; otherwise, use default (0/No Crash)
        if raw_predictions:
            display_prediction = "CRASH DETECTED" if current_prediction == 1 else "No Crash"
        else:
            display_prediction = "Loading..."
            
        color = (0, 0, 255) if display_prediction == "CRASH DETECTED" else (0, 255, 0) # Red/Green
        
        # --- START CORRECTED BLACK BLOCK LOGIC ---
        font_scale = 1.5
        font_thickness = 3
        text_pos = (10, 50) # The desired bottom-left start point for the text
        
        # 1. Get the size of the text string
        (text_width, text_height), baseline = cv2.getTextSize(
            display_prediction, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            font_thickness
        )
        
        # 2. Define the coordinates for the background rectangle
        # Padding to give some space around the text
        padding = 5 
        
        # Top-Left Corner (Start): X-coord is 10 - padding. Y-coord is above the text (50 - height - padding)
        rect_start = (text_pos[0] - padding, text_pos[1] - text_height - baseline - padding)
        
        # Bottom-Right Corner (End): X-coord is 10 + width + padding. Y-coord is slightly below the text baseline (50 + padding)
        rect_end = (text_pos[0] + text_width + padding, text_pos[1] + padding)
        
        # 3. Draw the filled black rectangle first (behind the text)
        cv2.rectangle(frame, rect_start, rect_end, (0, 0, 0), -1) # (0,0,0) is black, -1 fills it
        # --- END CORRECTED BLACK BLOCK LOGIC ---
        
        # Overlay the prediction text on the video frame
        cv2.putText(frame, 
                    display_prediction, 
                    (10, 50), # Position
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5, # Font Scale
                    color, 
                    3, # Thickness
                    cv2.LINE_AA)
        
        # Display the frame
        cv2.imshow('Crash Detection Pipeline', frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(50) & 0xFF == ord('q'): # Changed to 1ms wait for smoother playback
            break

        frame_idx += 1

    # --- Cleanup ---
    print("Video finished. Press any key in the video window to close.")
    cv2.waitKey(0)


    cap.release()
    cv2.destroyAllWindows()
    print("\nVideo analysis complete.")


# --- 3. ARGPARSE SETUP ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time crash detection from video file.")
    parser.add_argument('video_file', type=str, help='Path to the input MP4 video file.')
    parser.add_argument('model_weights', type=str, help='Path to the trained PyTorch .pth model weights file.')
    
    args = parser.parse_args()
    
    main(args.video_file, args.model_weights)