import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import os
import time
from collections import deque
# Load Model 
model_path = os.path.join(os.path.dirname(__file__), "predictWaste12.h5")
model = tf.keras.models.load_model(model_path)
# Classes 
classes = [
    "cardboard", "glass", "metal", "paper", "plastic", "trash",
    "clothes", "green-waste", "shoes", "food", "battery", "others"
]
# Mediapipe Setup 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
#  FPS & Smoothing 
prev_time = 0
smooth_window = 5
pred_buffer = deque(maxlen=smooth_window)
roi_buffer = deque(maxlen=3)  # Smooth ROI movement
#  Helper Functions
def is_pointing(landmarks):
    tips = [8, 12, 16, 20]
    pip_joints = [6, 10, 14, 18]
    extended = landmarks[tips[0]].y < landmarks[pip_joints[0]].y
    folded = all(landmarks[t].y > landmarks[p].y for t, p in zip(tips[1:], pip_joints[1:]))
    return extended and folded

def segment_object(roi):
    """
    Robust segmentation using color + edge detection.
    Returns masked ROI of largest object.
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Color mask (adjust range if background is complex)
    lower = np.array([0, 30, 30])
    upper = np.array([179, 255, 255])
    mask_color = cv2.inRange(hsv, lower, upper)

    # Edge mask
    edges = cv2.Canny(gray, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    # Combine masks
    combined = cv2.bitwise_or(mask_color, edges_dilated)

    # Find largest contour
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest], -1, 255, -1)
        segmented = cv2.bitwise_and(roi, roi, mask=mask)
        return segmented
    return roi  # fallback

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark

                if is_pointing(landmarks):
                    # Compute dynamic ROI 
                    index_tip = landmarks[8]
                    wrist = landmarks[0]
                    cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                    wx, wy = int(wrist.x * w), int(wrist.y * h)

                    dx, dy = cx - wx, cy - wy
                    hand_distance = np.hypot(dx, dy)
                    
                    # Use hand Z for depth scaling (closer = larger ROI)
                    z_scale = max(1 - index_tip.z, 0.5)
                    roi_size = int(hand_distance * 2.0 * z_scale)
                    
                    roi_center_x = cx + dx
                    roi_center_y = cy + dy

                    xmin = max(int(roi_center_x - roi_size // 2), 0)
                    ymin = max(int(roi_center_y - roi_size // 2), 0)
                    xmax = min(int(roi_center_x + roi_size // 2), w)
                    ymax = min(int(roi_center_y + roi_size // 2), h)

                    roi_buffer.append((xmin, ymin, xmax, ymax))
                    if roi_buffer:
                        xmin = int(np.mean([r[0] for r in roi_buffer]))
                        ymin = int(np.mean([r[1] for r in roi_buffer]))
                        xmax = int(np.mean([r[2] for r in roi_buffer]))
                        ymax = int(np.mean([r[3] for r in roi_buffer]))

                    # Draw fingertip & ROI
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)
                    cv2.line(frame, (cx, cy), ((xmin+xmax)//2, (ymin+ymax)//2), (255, 255, 0), 2)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

                    roi = frame[ymin:ymax, xmin:xmax]
                    if roi.size > 0:
                        segmented_roi = segment_object(roi)
                        img = cv2.resize(segmented_roi, (224, 224))
                        img = img.astype("float32") / 255.0
                        img = np.expand_dims(img, axis=0)

                        #  Predict 
                        pred = model.predict(img, verbose=0)[0]
                        pred_buffer.append(pred)
                        avg_pred = np.mean(pred_buffer, axis=0)
                        top3_idx = avg_pred.argsort()[-3:][::-1]

                        if np.max(avg_pred) < 0.3:
                            cv2.putText(frame, "Unknown", (xmin, ymin - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            for i, idx in enumerate(top3_idx):
                                label = f"{classes[idx]} ({avg_pred[idx]:.2f})"
                                y_offset = ymin - 10 - i*25
                                color = (0,255,0) if avg_pred[idx]>0.5 else (0,165,255)
                                cv2.putText(frame, label, (xmin, y_offset),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                bar_x, bar_y = xmin, ymax + 30 + i*25
                                bar_length = int(avg_pred[idx]*150)
                                cv2.rectangle(frame, (bar_x, bar_y-10), (bar_x+bar_length, bar_y), (0,255,0), -1)
        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time-prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),2)
        # Display
        frame_resized = cv2.resize(frame, (960, 720))
        cv2.imshow("Robust Segmented Pointing Classifier", frame_resized)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()
