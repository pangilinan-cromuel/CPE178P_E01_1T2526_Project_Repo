import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64
import json
import logging
import cv2
import numpy as np

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Model setup
# ----------------------------
MODEL_PATH = "best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

try:
    # Use EfficientNet B0
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # 2 classes: drug_user, not_user

    # Load weights
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")

except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# ----------------------------
# Image Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ----------------------------
# Face Detection Setup
# ----------------------------
# Load OpenCV Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face(pil_image):
    """Detects if a human face is present in the image."""
    cv_img = np.array(pil_image.convert("RGB"))[:, :, ::-1]  # Convert to OpenCV BGR
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return len(faces) > 0

# ----------------------------
# Prediction Function
# ----------------------------
def predict_image(image):
    try:
        # Check if a face is detected first
        if not detect_face(image):
            logger.warning("No face detected in image")
            return "no_face_detected", 0.0

        # Transform and predict
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        confidence = probabilities[0][predicted.item()].item()

        label = "drug_user" if predicted.item() == 0 else "not_user"

        logger.info(f"Prediction: {label}, Confidence: {confidence:.4f}")
        return label, confidence

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

# ----------------------------
# FastAPI Setup
# ----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# POST Endpoint for File Upload
# ----------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Read the uploaded file
        contents = await file.read()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Predict
        label, confidence = predict_image(image)
        
        if label == "no_face_detected":
            return {
                "result": "No face detected",
                "confidence": 0.0,
                "error": "No face detected in the image. Please upload a clear face image."
            }
        else:
            # Convert label to readable format
            readable_result = "Drug User" if label == "drug_user" else "Not a Drug User"
            
            return {
                "result": readable_result,
                "confidence": confidence,
                "prediction": label
            }
            
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# ----------------------------
# WebSocket Endpoint
# ----------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")

    try:
        while True:
            try:
                data = await websocket.receive_text()
            except WebSocketDisconnect:
                logger.info("Client disconnected")
                break

            try:
                message = json.loads(data)

                if "image" in message:
                    image_data = message["image"]
                elif "data" in message:
                    image_data = message["data"]
                else:
                    await websocket.send_text(json.dumps({"error": "No image data found"}))
                    continue

                # Decode image
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                # Predict or detect face
                label, confidence = predict_image(image)

                if label == "no_face_detected":
                    response = {"error": "No face detected in the image"}
                else:
                    response = {
                        "prediction": label,
                        "confidence": round(confidence, 4)
                    }

                await websocket.send_text(json.dumps(response))
                logger.info(f"Response sent: {response}")

            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                await websocket.send_text(json.dumps({"error": str(e)}))

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket connection closed")

# ----------------------------
# Health Check Endpoint
# ----------------------------
@app.get("/")
async def health_check():
    return {"status": "healthy", "model_loaded": True}