from fastapi import FastAPI, File, UploadFile, HTTPException, Form
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
from fastapi.staticfiles import StaticFiles

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
        gemini_model = genai.GenerativeModel('gemini-1.5-pro')
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
async def predict_mudra(file: UploadFile = File(...), target_mudra: str = Form(None)):
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
        
        # Fallback descriptions in case Gemini API fails
        fallback_descriptions = {
            "Alapadma": "यह अलपद्म मुद्रा है, जो पूर्ण खिले हुए कमल का प्रतीक है।",
            "Ardhapataka": "यह अर्धपताका मुद्रा है, जो आधा झंडा, पत्तियां या चाकू को दर्शाती है।",
            "Chandrakala": "यह चंद्रकला मुद्रा है, जो अर्धचंद्र या आभूषण को दर्शाती है।",
            "Kartarimukha": "यह कर्तरीमुख मुद्रा है, जो कैंची के समान अलगाव या विरोध दर्शाती है।",
            "Mayura": "यह मयूर मुद्रा है, जो मोर या सुंदरता का प्रतीक है।",
            "Pataka": "यह पताका मुद्रा है, जिसका अर्थ झंडा है। इसका उपयोग बादल या जंगल को दर्शाने में होता है।",
            "Shikhara": "यह शिखर मुद्रा है, जो धनुष या दृढ़ निश्चय का प्रतीक है।",
            "Simhamukha": "यह सिंहमुख मुद्रा है, जो शेर के चेहरे या साहस को दर्शाती है।",
            "Suchi": "यह सूची मुद्रा है, जिसका उपयोग सुई या किसी चीज़ को इंगित करने के लिए होता है।",
            "Tripataka": "यह त्रिपताका मुद्रा है, जो मुकुट, पेड़ या तीर को दर्शाने के लिए उपयोग की जाती है।"
        }
        fallback_desc = fallback_descriptions.get(predicted_mudra, f"यह {predicted_mudra} मुद्रा है।")
        
        gemini_message = ""
        audio_url = None
        
        # Determine success threshold
        if confidence_pct >= 70:
            # Check if practice target was set and missed
            if target_mudra and target_mudra != predicted_mudra:
                if gemini_model:
                    try:
                        prompt = f"The student wanted to form the '{target_mudra}' mudra but formed '{predicted_mudra}' instead. Act as an expert Bharatanatyam teacher. Gently correct them in Hindi, telling them what they actually formed, and clearly instruct how to position their fingers to correctly achieve the intended '{target_mudra}'."
                        response = gemini_model.generate_content(prompt)
                        gemini_message = response.text.replace("\n", " ").strip()
                        audio_url = text_to_speech_base64(gemini_message)
                    except Exception as e:
                        print(f"Gemini error: {e}")
                        gemini_message = f"आपने {predicted_mudra} बनाया है। कृपया {target_mudra} का अभ्यास करें।"
                        audio_url = text_to_speech_base64(gemini_message)

                return {
                    "success": False,
                    "error": "wrong_mudra",
                    "message": f"You formed {predicted_mudra} instead of {target_mudra}. Listen to the AI teacher's instructions to improve.",
                    "mudra": target_mudra,
                    "confidence": round(confidence_pct, 2),
                    "cropped_image": cropped_b64,
                    "gemini_message": gemini_message,
                    "audio_url": audio_url
                }
            
            # Correct mudra or free capture
            if gemini_model:
                try:
                    prompt = f"The user flawlessly formed the '{predicted_mudra}' mudra in Bharatanatyam. Describe the physical visual structure of this mudra and explain its classical symbolism in 2 elegant Hindi sentences."
                    response = gemini_model.generate_content(prompt)
                    gemini_message = response.text.replace("\n", " ").strip()
                    audio_url = text_to_speech_base64(gemini_message)
                except Exception as e:
                    print(f"Gemini error: {e}")
                    gemini_message = fallback_desc
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
                    practice_mudra = target_mudra if target_mudra else predicted_mudra
                    prompt = f"A student is doing Bharatanatyam and struggling to perform the '{practice_mudra}' mudra. Act as an expert dance teacher. In 2 encouraging Hindi sentences, provide clear, step-by-step physical instructions on exactly how to fold, position, and align their fingers to master this specific mudra."
                    response = gemini_model.generate_content(prompt)
                    gemini_message = response.text.replace("\n", " ").strip()
                    audio_url = text_to_speech_base64(gemini_message)
                except Exception as e:
                    print(f"Gemini error: {e}")
                    gemini_message = f"कृपया {practice_mudra} के लिए अपनी उंगलियों की स्थिति का अभ्यास करें।"
                    audio_url = text_to_speech_base64(gemini_message)

            return {
                "success": False,
                "error": "low_confidence",
                "message": f"Low confidence. Please follow the teacher's instructions to improve your posture for {target_mudra if target_mudra else predicted_mudra}.",
                "mudra": target_mudra if target_mudra else predicted_mudra,
                "confidence": round(confidence_pct, 2),
                "cropped_image": cropped_b64,
                "gemini_message": gemini_message,
                "audio_url": audio_url
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Mount frontend static files
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
else:
    @app.get("/")
    def read_root():
        return {"message": "NrityaVaani API is running. Use the /predict endpoint for inference. (Static frontend not found)"}
