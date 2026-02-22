from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
import numpy as np
import cv2
import base64

# ---- MediaPipe Tasks API ----
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
from gtts import gTTS

def text_to_speech_base64(text, lang='hi'):
    """Converts text to speech using gTTS and returns base64 encoded audio"""
    try:
        tts = gTTS(text=text, lang=lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_b64 = "data:audio/mp3;base64," + base64.b64encode(fp.read()).decode("utf-8")
        return audio_b64
    except Exception as e:
        print(f"gTTS error: {e}")
        return None

# ---- Gemini API ----
import google.generativeai as genai
import os

app = FastAPI(title="NrityaVaani API")

GEMINI_KEY_PATH = "api.bin"
gemini_model = None
gemini_api_key = os.environ.get("GEMINI_API_KEY")

try:
    if not gemini_api_key:
        with open(GEMINI_KEY_PATH, "r") as f:
            gemini_api_key = f.read().strip()
            
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    else:
        print("Warning: Neither GEMINI_API_KEY environment variable nor api.bin were found. Gemini Voice features will be disabled.")
except FileNotFoundError:
    print(f"Warning: {GEMINI_KEY_PATH} not found. Gemini Voice features will be disabled.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- CONFIG ----------
MODEL_PATH = "nrityavaani_mobilenet.pth"
DATA_DIR = "final_dataset/train"
HAND_MODEL = "hand_landmarker.task"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load classes
if os.path.exists(DATA_DIR):
    classes = sorted(os.listdir(DATA_DIR))
else:
    # Fallback to known classes
    classes = ['Alapadma', 'Ardhapataka', 'Chandrakala', 'Kartarimukha', 'Mayura', 'Pataka', 'Shikhara', 'Simhamukha', 'Suchi', 'Tripataka']

# Load model
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load hand landmarker
def load_hand_landmarker():
    base_options = python.BaseOptions(model_asset_path=HAND_MODEL)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.IMAGE
    )
    return vision.HandLandmarker.create_from_options(options)

hand_landmarker = load_hand_landmarker()

@app.post("/predict")
async def predict_mudra(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image type")

    try:
        # Read up to 5MB to prevent memory exhaustion attacks
        MAX_FILE_SIZE = 5 * 1024 * 1024
        content = await file.read(MAX_FILE_SIZE + 1)
        
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 5MB.")
            
        file_bytes = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        h, w, _ = frame.shape
        
        # Convert to MediaPipe format
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        # Detect hands
        result = hand_landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return {
                "success": False,
                "error": "hand_not_detected",
                "message": "Hand not detected. Please ensure your full hand is visible with adequate lighting."
            }

        # Crop to the detected hand bounding box
        lm = result.hand_landmarks[0]
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]

        x1 = max(0, int(min(xs) * w) - 30)
        y1 = max(0, int(min(ys) * h) - 30)
        x2 = min(w, int(max(xs) * w) + 30)
        y2 = min(h, int(max(ys) * h) + 30)

        hand_crop = frame[y1:y2, x1:x2]
        
        if hand_crop.size == 0:
             return {
                 "success": False, 
                 "error": "crop_failed", 
                 "message": "Failed to crop hand image properly."
             }

        cropped_img = Image.fromarray(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))
        
        # Base64 Encode the cropped image for frontend display
        buffered = io.BytesIO()
        cropped_img.save(buffered, format="JPEG")
        cropped_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Model Prediction
        x = transform(cropped_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(x)
            prob = torch.softmax(out, dim=1)
            idx = prob.argmax().item()
            confidence = prob[0][idx].item()

        predicted_mudra = classes[idx]
        confidence_pct = confidence * 100
        
        gemini_message = ""
        audio_url = None
        
        # Determine success threshold
        if confidence_pct >= 70:
            if gemini_model:
                try:
                    prompt = f"In one short, simple sentence in Hindi, what is the classical meaning or symbolism of the {predicted_mudra} mudra in Bharatanatyam?"
                    response = gemini_model.generate_content(prompt)
                    gemini_message = response.text.replace("\n", " ").strip()
                    audio_url = text_to_speech_base64(gemini_message)
                except Exception as e:
                    print(f"Gemini error: {e}")
                    gemini_message = f"यह {predicted_mudra} मुद्रा है।"
                    audio_url = text_to_speech_base64(gemini_message)

            return {
                "success": True,
                "mudra": predicted_mudra,
                "confidence": round(confidence_pct, 2),
                "cropped_image": cropped_b64,
                "gemini_message": gemini_message,
                "audio_url": audio_url
            }
        else:
            if gemini_model:
                try:
                    prompt = f"A student is trying to perform the {predicted_mudra} mudra in Bharatanatyam but is struggling. In one short, encouraging sentence in Hindi, tell them exactly how to position their fingers correctly to achieve this mudra."
                    response = gemini_model.generate_content(prompt)
                    gemini_message = response.text.replace("\n", " ").strip()
                    audio_url = text_to_speech_base64(gemini_message)
                except Exception as e:
                    print(f"Gemini error: {e}")
                    gemini_message = f"कृपया {predicted_mudra} के लिए अपनी उंगलियों की स्थिति का अभ्यास करें।"
                    audio_url = text_to_speech_base64(gemini_message)

            return {
                "success": False,
                "error": "low_confidence",
                "message": "Low confidence. Please try again with clear hand posture.",
                "mudra": predicted_mudra,
                "confidence": round(confidence_pct, 2),
                "cropped_image": cropped_b64,
                "gemini_message": gemini_message,
                "audio_url": audio_url
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "NrityaVaani API is running. Use the /predict endpoint for inference."}

@app.get("/health")
def health_check():
    return {"status": "ok"}
