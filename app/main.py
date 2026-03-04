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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

# ── Static files directory ───────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Initialize mediapipe holistic for /predict_frames (NV21 video frames — sequential)
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False
)

def extract_landmarks_from_frame(frame_bgr):
    """Extract landmarks from a single frame using the global holistic instance.
    Used for sequential NV21 video frames (predict_frames endpoint)."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    results = holistic.process(frame_rgb)
    return _results_to_landmarks(results)


def extract_landmarks_static(frame_bgr):
    """Extract landmarks in static image mode — best for independent JPEG frames
    from the web (predict_web_frames endpoint). Creates its own holistic each time
    a batch is processed for thread safety."""
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=True
    ) as h:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = h.process(frame_rgb)
        return _results_to_landmarks(results)


def _results_to_landmarks(results):
    """Convert MediaPipe holistic results to 285-element landmark array."""
    landmarks = []
    # Left hand (21 × 3 = 63)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)

    # Right hand (21 × 3 = 63)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)

    # Pose world landmarks (33 × 3 = 99)
    if results.pose_world_landmarks:
        for lm in results.pose_world_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 99)

    # Face (20 selected landmarks × 3 = 60)
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


def orient_frame(frame_bgr, rotation=0, is_front_camera=False):
    """Rotate and flip a frame so it is upright for MediaPipe.

    Mobile camera sensors are typically rotated 90° or 270° relative to
    the device's natural orientation.  The Flutter camera plugin reports
    the sensor orientation but does NOT auto-rotate the raw NV21 bytes,
    so the server must correct it here.
    """
    if rotation == 90:
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 180:
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)
    elif rotation == 270:
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    if is_front_camera:
        frame_bgr = cv2.flip(frame_bgr, 1)  # horizontal mirror
    return frame_bgr

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
    rotation: Optional[int] = 0
    is_front_camera: Optional[bool] = False

class FramesRequest(BaseModel):
    frames: List[NV21Frame]

class WebFrame(BaseModel):
    base64_data: str  # JPEG base64 from browser canvas

class WebFramesRequest(BaseModel):
    frames: List[WebFrame]

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
        non_zero_frames = 0
        for i, frame_data in enumerate(request.frames):
            raw_bytes = base64.b64decode(frame_data.base64_data)
            yuv = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(
                (int(frame_data.height * 1.5), frame_data.width)
            )
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)

            # Rotate / flip to upright orientation
            bgr = orient_frame(
                bgr,
                rotation=frame_data.rotation or 0,
                is_front_camera=frame_data.is_front_camera or False,
            )

            if i == 0:
                print(
                    f"[predict_frames] frame-0 decoded to {bgr.shape}, "
                    f"rotation={frame_data.rotation}, front={frame_data.is_front_camera}"
                )

            landmarks = extract_landmarks_from_frame(bgr)
            non_zero = sum(1 for v in landmarks if v != 0.0)
            if non_zero > 0:
                non_zero_frames += 1
            all_landmarks.append(landmarks)

        print(f"[predict_frames] {non_zero_frames}/{SEQUENCE_LENGTH} frames had non-zero landmarks")

        input_data = np.array(all_landmarks, dtype=np.float32).reshape(1, SEQUENCE_LENGTH, N_FEATURES)
        output = model.predict(input_data, verbose=0)
        probabilities = output[0]
        pred_index = int(np.argmax(probabilities))
        pred_label = LABELS[pred_index]
        confidence = float(probabilities[pred_index])
        all_preds = {LABELS[i]: round(float(probabilities[i]), 4) for i in range(len(LABELS))}

        print(f"[predict_frames] Prediction: {pred_label} ({confidence:.4f})")

        return PredictionResponse(
            label=pred_label,
            confidence=round(confidence, 4),
            all_predictions=all_preds,
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[predict_frames] ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_web_frames", response_model=PredictionResponse)
async def predict_web_frames(request: WebFramesRequest):
    """Accept JPEG base64 frames from browser webcam, extract landmarks, predict."""
    try:
        if len(request.frames) != SEQUENCE_LENGTH:
            raise HTTPException(status_code=400, detail=f"Expected {SEQUENCE_LENGTH} frames")

        all_landmarks = []
        non_zero_frames = 0
        for i, frame_data in enumerate(request.frames):
            # Remove data URL prefix if present
            b64 = frame_data.base64_data
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            raw_bytes = base64.b64decode(b64)
            nparr = np.frombuffer(raw_bytes, np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise HTTPException(status_code=400, detail=f"Failed to decode frame {i}")
            # Use static mode for independent JPEG frames from web
            landmarks = extract_landmarks_static(bgr)
            non_zero = sum(1 for v in landmarks if v != 0.0)
            if non_zero > 0:
                non_zero_frames += 1
            all_landmarks.append(landmarks)

        print(f"[predict_web_frames] {non_zero_frames}/{SEQUENCE_LENGTH} frames had non-zero landmarks")

        input_data = np.array(all_landmarks, dtype=np.float32).reshape(1, SEQUENCE_LENGTH, N_FEATURES)
        output = model.predict(input_data, verbose=0)
        probabilities = output[0]
        pred_index = int(np.argmax(probabilities))
        pred_label = LABELS[pred_index]
        confidence = float(probabilities[pred_index])
        all_preds = {LABELS[i]: round(float(probabilities[i]), 4) for i in range(len(LABELS))}

        print(f"[predict_web_frames] Prediction: {pred_label} ({confidence:.4f})")

        return PredictionResponse(
            label=pred_label,
            confidence=round(confidence, 4),
            all_predictions=all_preds,
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[predict_web_frames] ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug_frame")
async def debug_frame(frame: NV21Frame):
    """Diagnostic: decode a single NV21 frame, extract landmarks, return stats."""
    try:
        raw_bytes = base64.b64decode(frame.base64_data)
        yuv = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(
            (int(frame.height * 1.5), frame.width)
        )
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
        bgr = orient_frame(bgr, rotation=frame.rotation or 0, is_front_camera=frame.is_front_camera or False)

        landmarks = extract_landmarks_from_frame(bgr)
        non_zero = sum(1 for v in landmarks if v != 0.0)
        sample = {f"idx_{i}": round(v, 4) for i, v in enumerate(landmarks) if v != 0.0}
        sample = dict(list(sample.items())[:10])  # first 10 non-zero

        return {
            "frame_shape": list(bgr.shape),
            "rotation_applied": frame.rotation,
            "is_front_camera": frame.is_front_camera,
            "total_landmarks": len(landmarks),
            "non_zero_count": non_zero,
            "sample_non_zero": sample,
            "has_body": non_zero > 0,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/info")
async def api_info():
    return {
        "service": "FSL Prediction API",
        "version": "1.0.0",
        "labels": LABELS,
        "input_shape": f"[{SEQUENCE_LENGTH}, {N_FEATURES}]",
    }


@app.get("/")
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "service": "FSL Prediction API",
        "version": "1.0.0",
        "labels": LABELS,
        "input_shape": f"[{SEQUENCE_LENGTH}, {N_FEATURES}]",
    }

# ── Mount static files LAST (catch-all) ──────────────────────────────
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
