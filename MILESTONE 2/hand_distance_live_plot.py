"""
hand_distance_live_plot.py
Real-time hand landmark distance + live plot.
"""

import time
import math
from collections import deque
import csv
import os

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

CAM_INDEX = 0
MAX_PTS = 200
PRINT_EVERY_N_FRAMES = 15
PINCH_PIXEL_THRESHOLD = 40
PINCH_NORM_THRESHOLD = 0.03
LOG_TO_CSV = True
CSV_FILENAME = "hand_distance_log.csv"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
)

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera index {CAM_INDEX}")

timestamps = deque(maxlen=MAX_PTS)
norm_dists = deque(maxlen=MAX_PTS)
pixel_dists = deque(maxlen=MAX_PTS)

plt.ion()
fig, ax = plt.subplots(figsize=(8, 4))
line_norm, = ax.plot([], [], label="norm_dist")
line_pix, = ax.plot([], [], label="pixel_dist")
ax.set_xlabel("time (s)")
ax.set_ylabel("distance")
ax.set_title("Thumb Tip ↔ Index Tip Distance (live)")
ax.legend(loc="upper right")
ax.grid(True)

start_time = time.time()
frame_count = 0

if LOG_TO_CSV:
    first_write = not os.path.exists(CSV_FILENAME)
    csv_file = open(CSV_FILENAME, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if first_write:
        csv_writer.writerow(["timestamp_iso", "elapsed_s", "norm_dist", "pixel_dist", "pinch"])

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        frame = cv2.flip(frame, 1)
        img_h, img_w = frame.shape[:2]
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        norm_dist = None
        pixel_dist = None
        pinch = False

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            lm_thumb = hand_landmarks.landmark[4]
            lm_index = hand_landmarks.landmark[8]

            dx_n = lm_thumb.x - lm_index.x
            dy_n = lm_thumb.y - lm_index.y
            dz_n = lm_thumb.z - lm_index.z
            norm_dist = math.sqrt(dx_n**2 + dy_n**2 + dz_n**2)

            tx_px, ty_px = int(lm_thumb.x * img_w), int(lm_thumb.y * img_h)
            ix_px, iy_px = int(lm_index.x * img_w), int(lm_index.y * img_h)
            pixel_dist = math.hypot(tx_px - ix_px, ty_px - iy_px)

            pinch = pixel_dist <= PINCH_PIXEL_THRESHOLD or (norm_dist and norm_dist <= PINCH_NORM_THRESHOLD)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.circle(frame, (tx_px, ty_px), 8, (0, 0, 255), -1)
            cv2.circle(frame, (ix_px, iy_px), 8, (255, 0, 0), -1)
            cv2.line(frame, (tx_px, ty_px), (ix_px, iy_px), (0, 255, 0), 2)

        cv2.putText(frame, f"Pixel: {pixel_dist:.1f}px" if pixel_dist else "No Hand",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Hand Distance (press 'q' to quit)", frame)

        timestamps.append(time.time() - start_time)
        norm_dists.append(norm_dist if norm_dist else np.nan)
        pixel_dists.append(pixel_dist if pixel_dist else np.nan)

        line_norm.set_data(timestamps, norm_dists)
        line_pix.set_data(timestamps, pixel_dists)
        ax.relim(); ax.autoscale_view()
        fig.canvas.draw(); fig.canvas.flush_events()
        plt.pause(0.001)

        if LOG_TO_CSV:
            csv_writer.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), timestamps[-1], norm_dist, pixel_dist, int(pinch)])
            if frame_count % 50 == 0:
                csv_file.flush()

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release(); cv2.destroyAllWindows(); hands.close()
    if LOG_TO_CSV: csv_file.close()
    print("Clean exit.")
