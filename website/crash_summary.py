import cv2
from google import genai
from google.genai import types
import requests
import os
from dotenv import load_dotenv
import time

load_dotenv()

def create_report(image_path_list):
    prompt = (
        "You are working with 911 as a dashcam surveillance operator. You have been provided with the key pictures from the scene" \
        "Provide a short description that will be relayed to the 911 operator paired with the images. If you are uncertain of an information" \
        "Make sure to precise if you're uncertain of key information (like plate numbers etc.) if you're mentionning key info." \
        "This message will be put as text inside of an HTML file, so if you do any formatting, do HTML compatible formatting"

    )

    contents = [prompt]

    if not image_path_list:
        return "Error: No image paths were provided."

    successful_images = 0
    # Loop over the list of image paths
    for path in image_path_list:
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
            
        # Add the image part to contents list
        image_part = types.Part.from_bytes(data=jpg.tobytes(), mime_type="image/jpeg")
        contents.append(image_part)
        successful_images += 1
        
    if successful_images == 0:
        return "Error: All image paths were invalid. No images to process."

    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=GEMINI_API_KEY)
    start = time.time()
    resp = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=contents,
    )
    print(f"gemini took {time.time() - start} time to reply")
    if resp.text:
        return resp.text.strip().strip("```html").strip("```")
    else:
        return "Unable to provide an adequate description for the scene."