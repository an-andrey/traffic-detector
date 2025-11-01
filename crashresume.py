import cv2, json
from google import genai
from google.genai import types

def describe_crash_with_gemini(frame_bgr):
    ok, jpg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    image_part = types.Part.from_bytes(data=jpg.tobytes(), mime_type="image/jpeg")

    prompt = (
        "You are an incident triage assistant for traffic authorities.\n"
        "Describe only what is visible in this frame and avoid speculation.\n"
        "Return strict JSON with keys: summary, vehicles_involved (count), "
        "vehicle_types (list), impact_type, lanes_blocked (list), hazards (list), "
        "severity_estimate (minor|moderate|severe|unknown), recommended_actions (list).\n"
        "If uncertain, use 'unknown'."
    )

    client = genai.Client(api_key="")
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, image_part],
    )
    text = resp.text or "{}"
    try:
        return json.loads(text)
    except Exception:
        # fallback: wrap free text into a JSON envelope
        return {"summary": text.strip(), "severity_estimate": "unknown",
                "vehicles_involved": None, "vehicle_types": [], "impact_type": "unknown",
                "lanes_blocked": [], "hazards": [], "recommended_actions": []}

path = "000004-42.jpg"
frame_bgr = cv2.imread(path)  # BGR array from disk

if frame_bgr is None:
    raise FileNotFoundError(f"Could not read {path}")
report = describe_crash_with_gemini(frame_bgr)
print(report)

# frame_bgr = ...  # obtain from your video at the crash frame (OpenCV BGR image)
# report = describe_crash_with_gemini(frame_bgr)
# print_report(report, camera_id="cam12", timestamp="2025-11-01T21:05:10Z")
