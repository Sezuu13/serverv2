"""
FSL (Filipino Sign Language) Prediction API
Loads the trained Grouped BiLSTM Keras model and serves predictions.
Expects 30 frames × 285 landmark features as JSON input.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import base64
import cv2
import mediapipe as mp

# ── Custom layer needed to deserialise the .keras model ──────────────
class SliceLayer(layers.Layer):
    def __init__(self, start, end, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.end = end

    def call(self, inputs):
        return inputs[..., self.start:self.end]

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({"start": self.start, "end": self.end})
        return config

# ── Constants ────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 30
N_FEATURES = 285
LABELS = [
    "Ingat",
    "Magandang Gabi",
    "Magandang Hapon",
    "Magandang Umaga",
    "Mahal Kita",
    "Paalam",
]

# ── Load model at startup ───────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "asl_final_model_fixed.keras")

print(f"Loading model from {MODEL_PATH} ...")
model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={"SliceLayer": SliceLayer},
)
print(f"[OK] Model loaded. Input shape: {model.input_shape}")

# ── FastAPI app ──────────────────────────────────────────────────────
app = FastAPI(
    title="FSL Prediction API",
    description="Filipino Sign Language detection using Grouped BiLSTM",
    version="1.0.0",
)

# Initialize mediapipe holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False
)

def extract_landmarks_from_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    results = holistic.process(frame_rgb)

    landmarks = []
    # Left hand
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)

    # Right hand
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)

    # Pose
    if results.pose_world_landmarks:
        for lm in results.pose_world_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 99)

    # Face
    key_indices = [0, 1, 4, 5, 9, 10, 13, 14, 17, 18, 21, 33, 36, 39, 42, 45, 48, 51, 54, 57]
    if results.face_landmarks:
        for idx in key_indices:
            if idx < len(results.face_landmarks.landmark):
                lm = results.face_landmarks.landmark[idx]
                landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([0.0, 0.0, 0.0])
    else:
        landmarks.extend([0.0] * 60)

    return landmarks

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response schemas ───────────────────────────────────────
class PredictionRequest(BaseModel):
    landmarks: List[List[float]]  # shape: [30][285]

class NV21Frame(BaseModel):
    base64_data: str
    width: int
    height: int

class FramesRequest(BaseModel):
    frames: List[NV21Frame]

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    all_predictions: dict  # label → confidence for every class

# ── Endpoints ────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        landmarks = request.landmarks

        # Validate shape
        if len(landmarks) != SEQUENCE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {SEQUENCE_LENGTH} frames, got {len(landmarks)}",
            )
        for i, frame in enumerate(landmarks):
            if len(frame) != N_FEATURES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Frame {i}: expected {N_FEATURES} features, got {len(frame)}",
                )

        # Convert to numpy and reshape for model
        input_data = np.array(landmarks, dtype=np.float32).reshape(
            1, SEQUENCE_LENGTH, N_FEATURES
        )

        # Run inference
        output = model.predict(input_data, verbose=0)
        probabilities = output[0]

        pred_index = int(np.argmax(probabilities))
        pred_label = LABELS[pred_index]
        confidence = float(probabilities[pred_index])

        all_preds = {
            LABELS[i]: round(float(probabilities[i]), 4)
            for i in range(len(LABELS))
        }

        return PredictionResponse(
            label=pred_label,
            confidence=round(confidence, 4),
            all_predictions=all_preds,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_frames", response_model=PredictionResponse)
async def predict_frames(request: FramesRequest):
    try:
        if len(request.frames) != SEQUENCE_LENGTH:
            raise HTTPException(status_code=400, detail=f"Expected {SEQUENCE_LENGTH} frames")

        all_landmarks = []
        for frame_data in request.frames:
            raw_bytes = base64.b64decode(frame_data.base64_data)
            yuv = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((int(frame_data.height * 1.5), frame_data.width))
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
            landmarks = extract_landmarks_from_frame(bgr)
            all_landmarks.append(landmarks)
            
        input_data = np.array(all_landmarks, dtype=np.float32).reshape(1, SEQUENCE_LENGTH, N_FEATURES)
        output = model.predict(input_data, verbose=0)
        probabilities = output[0]
        pred_index = int(np.argmax(probabilities))
        pred_label = LABELS[pred_index]
        confidence = float(probabilities[pred_index])
        all_preds = {LABELS[i]: round(float(probabilities[i]), 4) for i in range(len(LABELS))}
        
        return PredictionResponse(
            label=pred_label,
            confidence=round(confidence, 4),
            all_predictions=all_preds,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "service": "FSL Prediction API",
        "version": "1.0.0",
        "labels": LABELS,
        "input_shape": f"[{SEQUENCE_LENGTH}, {N_FEATURES}]",
    }
