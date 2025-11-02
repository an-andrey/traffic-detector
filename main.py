import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import cv2, json
import argparse
from crashresume import create_report
import os

# --- 1. UTILITY: NUMPY MODE & TEMPORAL SMOOTHING FUNCTIONS ---

def numpy_mode(arr):
    """Calculates the mode of a 1D NumPy array."""
    if len(arr) == 0:
        return 0
    counts = np.bincount(arr)
    return np.argmax(counts)

def smooth_predictions(raw_predictions, max_gap_size=9):
    """
    Fills gaps of a specific size or less (up to max_gap_size) 
    where a sequence of zeros is surrounded by ones.
    """
    predictions = np.array(raw_predictions)
    
    if len(predictions) < 2:
        return predictions.tolist()

    smoothed = predictions.copy()
    start_of_gap = -1
    
    for i in range(len(predictions)):
        # Look for the start of a gap (a '0') immediately following a '1'
        if predictions[i] == 0:
            if i > 0 and predictions[i-1] == 1:
                start_of_gap = i
        
        # Check for the end of a gap
        if predictions[i] == 1:
            if start_of_gap != -1:
                gap_length = i - start_of_gap
                
                if 1 <= gap_length <= max_gap_size:
                    # Fill the gap (the zeros only)
                    smoothed[start_of_gap:i] = 1 
                
                start_of_gap = -1
        
        # Handle a gap that exceeds the max_gap_size
        elif start_of_gap != -1:
            current_gap_length = i - start_of_gap + 1
            if current_gap_length > max_gap_size:
                start_of_gap = -1
    
    return smoothed.tolist()

# --- 2. MAIN EXECUTION PIPELINE (With Playback Delay) ---

def main(video_path, model_path, fps_sampling=10, max_gap_size=3):
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_loc = 'cuda' if device.type == 'cuda' else 'cpu'
    print(f"Using device: {device}")

    # Load Model
    model = resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=map_loc))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}.")

    # Define Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # Initialize Video Capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0: 
         original_fps = 30
    frame_interval = max(1, int(round(original_fps / fps_sampling)))
    
    # Buffers and State
    raw_predictions = []
    smoothed_predictions = [] 
    frame_buffer = [] 
    prediction_delay = max_gap_size
    
    # AI Reporting State Variables
    CRASH_REPORT_THRESHOLD = 10 # Number of consecutive '1's required for a report
    is_in_crash_event = False
    
    print(f"Original FPS: {original_fps:.2f}, Sampling every {frame_interval} frames (~{fps_sampling} FPS).")
    print(f"Gap-filling up to size: {max_gap_size}. Playback delay: {prediction_delay} samples.")
    print(f"AI Report Threshold: {CRASH_REPORT_THRESHOLD} consecutive '1's.")

    # --- Start Video Processing Loop ---
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        is_sampled_frame = (frame_idx % frame_interval == 0)
        
        # 1. Frame Buffer Logic: Store the current frame
        frame_buffer.append(frame)

        # 2. Prediction Logic
        if is_sampled_frame:
            # Predict
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image_tensor)
                _, pred = torch.max(output, 1)
                raw_predictions.append(pred.item())
            
            # Smooth the entire history
            smoothed_predictions = smooth_predictions(raw_predictions, max_gap_size=max_gap_size)

        
        # 3. Playback and Display Logic
        if len(smoothed_predictions) > prediction_delay:
            
            # Get the frame and prediction for the delayed display time
            display_frame_bgr = frame_buffer.pop(0)
            prediction_index = len(smoothed_predictions) - prediction_delay - 1
            current_prediction = smoothed_predictions[prediction_index]
            
            # --- AI CRASH REPORTING LOGIC ---
            
            # Check for a sustained crash event using the smoothed predictions
            current_history = smoothed_predictions[:len(smoothed_predictions) - prediction_delay]
            
            if len(current_history) >= CRASH_REPORT_THRESHOLD:
                
                # Check the last CRASH_REPORT_THRESHOLD predictions
                last_predictions = current_history[-CRASH_REPORT_THRESHOLD:]
                
                if all(p == 1 for p in last_predictions) and not is_in_crash_event:
                    # Sustained crash detected! Trigger the report.
                    is_in_crash_event = True
                    print("\n!!! SUSTAINED CRASH DETECTED !!!")
                    
                    # --- DIRECTORY SETUP ---
                    UPLOAD_DIR = 'website/static/uploads'
                    if not os.path.exists(UPLOAD_DIR):
                        os.makedirs(UPLOAD_DIR)
                        print(f"Created directory: {UPLOAD_DIR}")
                    
                    # --- CALCULATE INDICES FOR 3 FRAMES (as you requested) ---
                    
                    # This calculates the index in the 'current_history' (which is the same length as raw_predictions)
                    middle_index_in_history = len(current_history) - (CRASH_REPORT_THRESHOLD // 2) - 1
                    
                    # We select: Middle - 2, Middle, Middle + 2 (as in your last code block)
                    sampled_indices_to_report = [
                        middle_index_in_history - 2,
                        middle_index_in_history,
                        middle_index_in_history + 2
                    ]

                    # Map sampled indices to the original video frame numbers
                    frame_indices_to_report = [
                        max(0, i * frame_interval) for i in sampled_indices_to_report
                    ]
                    
                    # --- RETRIEVE, SAVE, AND COLLECT FILE PATHS ---
                    report_file_paths = []
                    all_frames_retrieved = True
                    
                    cap_report = cv2.VideoCapture(video_path)
                    
                    for i, frame_num in enumerate(frame_indices_to_report):
                        cap_report.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                        ret_r, report_frame_bgr = cap_report.read()
                        
                        if ret_r:
                            # SAVE the image to the 'upload/' directory
                            timestamp = int(cap_report.get(cv2.CAP_PROP_POS_MSEC))
                            filename = f"crash_frame_{timestamp}ms_p{i+1}.jpg"
                            file_path = os.path.join(UPLOAD_DIR, filename)
                            
                            cv2.imwrite(file_path, report_frame_bgr)
                            report_file_paths.append(file_path)
                            print(f"Saved frame {i+1} to: {file_path}")
                        else:
                            all_frames_retrieved = False
                            print(f"Failed to re-read frame {frame_num} for reporting.")
                            break
                            
                    cap_report.release() # Release the temporary capture
                    
                    # --- CALL GEMINI WITH THE LIST OF FILE PATHS ---
                    if all_frames_retrieved and len(report_file_paths) == 3:
                        # PASS THE LIST of 3 FILE PATHS
                        create_report(report_file_paths) 
                    else:
                        print("Skipping AI report due to insufficient/failed frame retrieval.")
                        
                elif not all(p == 1 for p in last_predictions):
                    # Crash event has ended
                    is_in_crash_event = False
            
            # --- END AI CRASH REPORTING LOGIC ---
            
            # Visualization Logic (on the displayed frame)
            display_prediction = "CRASH DETECTED" if current_prediction == 1 else "No Crash"
            color = (0, 0, 255) if display_prediction == "CRASH DETECTED" else (0, 255, 0)
            
            font_scale = 1.5
            font_thickness = 3
            text_pos = (10, 50)
            
            (text_width, text_height), baseline = cv2.getTextSize(
                display_prediction, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            padding = 5 
            rect_start = (text_pos[0] - padding, text_pos[1] - text_height - baseline - padding)
            rect_end = (text_pos[0] + text_width + padding, text_pos[1] + padding)
            
            cv2.rectangle(display_frame_bgr, rect_start, rect_end, (0, 0, 0), -1)
            
            cv2.putText(display_frame_bgr, 
                        display_prediction, 
                        (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.5, 
                        color, 
                        3, 
                        cv2.LINE_AA)
            
            # Display the frame
            cv2.imshow('Crash Detection Pipeline (Delayed)', display_frame_bgr)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

        frame_idx += 1

    # --- Post-Processing and Cleanup (Remaining Frames) ---
    print("\nProcessing remaining buffered frames...")

    for i in range(len(frame_buffer)):
        prediction_index = len(smoothed_predictions) - len(frame_buffer) + i
        current_prediction = smoothed_predictions[prediction_index]
        display_frame_bgr = frame_buffer[i]
        
        display_prediction = "CRASH DETECTED" if current_prediction == 1 else "No Crash"
        color = (0, 0, 255) if display_prediction == "CRASH DETECTED" else (0, 255, 0)
        
        # Overlay logic (simplified)
        cv2.putText(display_frame_bgr, display_prediction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)
        cv2.imshow('Crash Detection Pipeline (Delayed)', display_frame_bgr)
        
        if cv2.waitKey(int(1000/original_fps)) & 0xFF == ord('q'):
            break

    print("Video finished. Press any key in the video window to close.")
    cv2.waitKey(0)


    cap.release()
    cv2.destroyAllWindows()
    print("\nVideo analysis complete.")


# --- 3. ARGPARSE SETUP ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time crash detection and AI reporting.")
    parser.add_argument('video_file', type=str, help='Path to the input MP4 video file.')
    parser.add_argument('model_weights', type=str, help='Path to the trained PyTorch .pth model weights file.')
    parser.add_argument('--max_gap', type=int, default=3, help='Max size of zero gap to fill (in sampled frames). This also determines the playback delay.')
    
    args = parser.parse_args()
    
    # Note: max_gap_size=5 is used in the user's final example call, so I'll use that 
    # as the default here to match the user's intent, overriding the default of 3.
    main(args.video_file, args.model_weights, max_gap_size=5)