"""
Streamlit live Gesture Recognition & Distance Measurement
Now supports clean Start/Stop multiple times.
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
import time

# Page setup
st.set_page_config(page_title="Gesture Recognition", layout="wide")
st.title("🖐️ Gesture Recognition & Distance Measurement (Live Stream)")

st.markdown("""
This app measures the distance between **thumb** and **index finger tips**
using your webcam feed — with **live video**, **pinch detection**, and **real-time updates**.
""")

# Sidebar
st.sidebar.header("⚙️ Settings")
min_conf = st.sidebar.slider("Min Detection Confidence", 0.3, 1.0, 0.7)
pinch_thresh_px = st.sidebar.slider("Pinch Pixel Threshold", 10, 100, 40)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=min_conf,
    min_tracking_confidence=min_conf,
    max_num_hands=1
)

# Layout
frame_placeholder = st.empty()
col1, col2 = st.columns(2)
pinch_display = col1.metric("Pinch Detected", "NO")
distance_display = col2.metric("Distance (px)", "0.0")

# Maintain camera object and session state
if "run" not in st.session_state:
    st.session_state.run = False
if "camera" not in st.session_state:
    st.session_state.camera = None

# Buttons
colA, colB = st.columns(2)
start_button = colA.button("▶️ Start Camera")
stop_button = colB.button("⏹️ Stop Camera")

# Start logic
if start_button and not st.session_state.run:
    st.session_state.run = True
    st.session_state.camera = cv2.VideoCapture(0)
    if not st.session_state.camera.isOpened():
        st.error("❌ Could not access the webcam.")
        st.session_state.run = False

# Stop logic
if stop_button and st.session_state.run:
    st.session_state.run = False
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None
    st.success("✅ Camera stopped.")

# Main live loop
if st.session_state.run and st.session_state.camera is not None:
    st.info("🎥 Camera running — press 'Stop Camera' to exit.")

    cap = st.session_state.camera
    while st.session_state.run:
        success, frame = cap.read()
        if not success:
            st.warning("⚠️ Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        distance_px = 0
        pinch_state = "NO"

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            lm_thumb = hand_landmarks.landmark[4]
            lm_index = hand_landmarks.landmark[8]

            tx, ty = int(lm_thumb.x * w), int(lm_thumb.y * h)
            ix, iy = int(lm_index.x * w), int(lm_index.y * h)
            distance_px = math.hypot(tx - ix, ty - iy)
            pinch_state = "YES" if distance_px <= pinch_thresh_px else "NO"

            cv2.line(frame, (tx, ty), (ix, iy), (0, 255, 0), 3)
            cv2.circle(frame, (tx, ty), 10, (0, 0, 255), -1)
            cv2.circle(frame, (ix, iy), 10, (255, 0, 0), -1)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Distance: {distance_px:.1f}px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Pinch: {pinch_state}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if pinch_state == "YES" else (0, 0, 255), 2)

        frame_placeholder.image(frame, channels="BGR")
        pinch_display.metric("Pinch Detected", pinch_state)
        distance_display.metric("Distance (px)", f"{distance_px:.1f}")

        time.sleep(0.03)

    # when loop ends (user stops)
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None
        st.session_state.run = False
        st.success("✅ Camera stopped cleanly.")

else:
    st.warning("Press **Start Camera** to begin.")
