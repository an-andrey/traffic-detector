import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.models import resnet18
import os

# --- 1. UTILITY: NUMPY MODE & TEMPORAL SMOOTHING FUNCTIONS ---

def numpy_mode(arr):
    """Calculates the mode of a 1D NumPy array."""
    if len(arr) == 0:
        return 0
    counts = np.bincount(arr)
    return np.argmax(counts)

def smoothen_predictions(raw_predictions, max_gap_size=9):
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

############################################
# Loads the PyTorch model with the weights #
############################################
def load_transforms():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    
    return transform

def load_model_and_tranforms(model_path):
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
    
    return model, transform, device


def generate_video_frames(video_path, model_path, fps_sampling=10, max_gap_size=3):
    
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