#!/usr/bin/env python3
"""
Real-time Sign Language Recognition using a pre-trained TFLite model.
Model: asl_final_model_float32.tflite
Label encoder: label_encoder.pkl
Features:
- 30‑frame sliding window buffer
- On‑screen buffering progress
- Live MediaPipe landmark overlay
"""

import cv2
import numpy as np
import mediapipe as mp
import pickle
import tensorflow as tf
from collections import deque

# ---------------------------
# Constants (must match training)
# ---------------------------
SEQUENCE_LENGTH = 30          # number of frames per sample
N_FEATURES = 285              # landmark features per frame
TARGET_CLASSES = ['Ingat', 'Magandang Gabi', 'Magandang Hapon',
                  'Magandang Umaga', 'Mahal Kita', 'Paalam']

# ---------------------------
# Load label encoder
# ---------------------------
with open('New Model2/v8/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
classes = label_encoder.classes_
print("Loaded classes:", classes)

# ---------------------------
# Load TFLite model
# ---------------------------
interpreter = tf.lite.Interpreter(model_path='New Model2/v8/asl_final_model_float32.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']      # expected [1, 30, 285]
print(f"Model input shape: {input_shape}")

# ---------------------------
# MediaPipe Holistic setup
# ---------------------------
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False
)
mp_drawing = mp.solutions.drawing_utils

# ---------------------------
# Helper: extract landmarks from a single frame
# ---------------------------
def extract_landmarks_from_frame(frame, holistic):
    """Process one frame and return a flattened 285‑element landmark array."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    results = holistic.process(frame_rgb)

    landmarks = []
    # Left hand (21 × 3 = 63)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)

    # Right hand (63)
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
    key_indices = [0, 1, 4, 5, 9, 10, 13, 14, 17, 18,
                   21, 33, 36, 39, 42, 45, 48, 51, 54, 57]
    if results.face_landmarks:
        for idx in key_indices:
            if idx < len(results.face_landmarks.landmark):
                lm = results.face_landmarks.landmark[idx]
                landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([0.0, 0.0, 0.0])
    else:
        landmarks.extend([0.0] * 60)

    return np.array(landmarks, dtype=np.float32)

# ---------------------------
# Real‑time video loop with buffer display
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
print("Starting real‑time recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)          # mirror view
    landmarks = extract_landmarks_from_frame(frame, holistic)
    frame_buffer.append(landmarks)

    # Display buffering progress while collecting initial frames
    if len(frame_buffer) < SEQUENCE_LENGTH:
        progress_text = f"Buffering: {len(frame_buffer)}/{SEQUENCE_LENGTH}"
        cv2.putText(frame, progress_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        # Buffer is full – run inference on the sliding window
        input_data = np.array(frame_buffer, dtype=np.float32).reshape(1, SEQUENCE_LENGTH, N_FEATURES)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        pred_index = np.argmax(output_data[0])
        pred_label = classes[pred_index]
        confidence = output_data[0][pred_index]

        # Show prediction
        text = f"{pred_label} ({confidence:.2f})"
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

    # (Optional) Draw MediaPipe landmarks for visual feedback
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1))

    cv2.imshow('Sign Language Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
holistic.close()