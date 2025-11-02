import cv2
from google import genai
from google.genai import types
import requests
import os

def send_report_to_dashboard(description, image_paths):   
    # The URL of your Flask server's API endpoint
    api_url = "http://127.0.0.1:5000/api/add_report"
    
    # The data to send, formatted as JSON
    payload = {
        "description": description,
        "image_paths": image_paths  # Send the list of file paths
    }
    
    print(f"Sending report to dashboard: {description[:50]}...")
    
    try:
        # Send the POST request
        response = requests.post(api_url, json=payload)
        
        if response.status_code == 201:
            print("Report successfully added to dashboard.")
        else:
            print(f"Error: Server responded with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the dashboard server.")
        print("Please ensure the 'app.py' Flask server is running.")

def describe_crash_with_gemini(*image_paths):
    prompt = (
        "You are working with 911 as a dashcam surveillance operator. You have been provided with the key pictures from the scene" \
        "Provide a short description that will be relayed to the 911 operator paired with the images. If you are uncertain of an information" \
        "Make sure to precise if you're uncertain of key information (like plate numbers etc.) if you're mentionning key info."
    )

    contents = [prompt]

    if not image_paths:
        return "Error: No image paths were provided."

    successful_images = 0
    # Loop over the tuple of image paths
    for path in image_paths:
        # Read the image from the path
        frame_bgr = cv2.imread(path)
        
        if frame_bgr is None:
            print(f"Warning: Could not read image at {path}. Skipping.")
            continue

        # Encode the image
        ok, jpg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            print(f"Warning: Could not encode image at {path}. Skipping.")
            continue
            
        # Add the image part to our contents list
        image_part = types.Part.from_bytes(data=jpg.tobytes(), mime_type="image/jpeg")
        contents.append(image_part)
        successful_images += 1
        
    if successful_images == 0:
        return "Error: All image paths were invalid. No images to process."

    client = genai.Client(api_key="")
    resp = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=contents,
    )

    if resp.text:
        return resp.text
    else:
        return "Unable to provide an adequate description for the scene."

path = "data/extracted_frames/000004-42.jpg"
frame_bgr = cv2.imread(path)  # BGR array from disk

if frame_bgr is None:
    raise FileNotFoundError(f"Could not read {path}")

report_description = describe_crash_with_gemini(frame_bgr)

print("--- Generated Report ---")
print(report_description)
print("------------------------")

# 2. Get the *absolute path* of the image
# The server needs this to find and copy the file
image_abs_path = os.path.abspath(path)

# 3. Send the report to the dashboard
# We pass the description and a *list* of image paths
send_report_to_dashboard(
    description=report_description, 
    image_paths=[image_abs_path] 
)
