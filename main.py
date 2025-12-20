import cv2
import math
import numpy as np
import streamlit as st
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL, CoInitialize
from comtypes.client import CreateObject
from pycaw.pycaw import IAudioEndpointVolume, IMMDeviceEnumerator

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(layout="wide", page_title="Hand Gesture Mic Control")

# ==================================================
# DARK UI THEME
# ==================================================
st.markdown("""
<style>
html, body, .stApp {
    background-color: #0b1220;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background-color: #0f172a;
}
.stButton>button {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 18px;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e3a8a);
}
.stMetric {
    background: rgba(15,23,42,0.9);
    padding: 14px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# PARAMETERS (TUNED — DON’T TOUCH RANDOMLY)
# ==================================================
PINCH_RATIO_THRESHOLD = 0.15
MIN_RATIO = 0.10
MAX_RATIO = 1.80
SMOOTHING = 5

FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE_THICKNESS = 4

# ==================================================
# MEDIAPIPE INIT
# ==================================================
@st.cache_resource
def load_mediapipe():
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    return hands, mp.solutions.drawing_utils

hands, mp_draw = load_mediapipe()

# ==================================================
# MICROPHONE CONTROL (WINDOWS)
# ==================================================
def get_mic_interface():
    try:
        CoInitialize()
        clsid = "{BCDE0395-E52F-467C-8E3D-C4579291692E}"
        enumerator = CreateObject(clsid, interface=IMMDeviceEnumerator)
        device = enumerator.GetDefaultAudioEndpoint(1, 0)  # MIC
        interface = device.Activate(
            IAudioEndpointVolume._iid_,
            CLSCTX_ALL,
            None
        )
        return cast(interface, POINTER(IAudioEndpointVolume))
    except:
        return None

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.header("Settings")

    st.subheader("Manual Mic Control")

    def manual_change(delta):
        vc = get_mic_interface()
        if vc:
            try:
                curr = vc.GetMasterVolumeLevelScalar()
                vc.SetMasterVolumeLevelScalar(
                    min(max(curr + delta, 0.0), 1.0), None
                )
            except:
                pass

    c1, c2 = st.columns(2)
    if c1.button("Vol +10%"):
        manual_change(0.1)
    if c2.button("Vol -10%"):
        manual_change(-0.1)

# ==================================================
# HEADER
# ==================================================
st.title("🎤 GestureGain - Volume Control Using Hand Gestures")
st.markdown("**Mentor:** Dr. D. Bhanu Prakash")
st.markdown("**Developed by:** Anusuya B, Prashanti Hebbar, Shreyas S N, Siva Sankar")

col1, col2 = st.columns([1, 1])

# ==================================================
# DASHBOARD
# ==================================================
with col1:
    st.subheader("Dashboard")

    if "run_camera" not in st.session_state:
        st.session_state.run_camera = False

    def toggle():
        st.session_state.run_camera = not st.session_state.run_camera

    st.button(
        "STOP CAMERA" if st.session_state.run_camera else "START CAMERA",
        on_click=toggle
    )

    vol_metric = st.empty()
    ratio_metric = st.empty()
    status_msg = st.empty()

    if not st.session_state.run_camera:
        status_msg.info("Camera is OFF")

# ==================================================
# LIVE FEED
# ==================================================
with col2:
    st.subheader("Live Camera Feed")
    video_container = st.empty()

# ==================================================
# CAMERA LOOP
# ==================================================
if st.session_state.run_camera:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Camera not detected.")
        st.session_state.run_camera = False
    else:
        vc = get_mic_interface()

        while st.session_state.run_camera:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            ratio_val = 0.0
            muted = False

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_lms,
                        mp.solutions.hands.HAND_CONNECTIONS
                    )

                    p4 = hand_lms.landmark[4]   # thumb tip
                    p8 = hand_lms.landmark[8]   # index tip
                    p0 = hand_lms.landmark[0]   # wrist
                    p9 = hand_lms.landmark[9]   # middle MCP

                    pinch = math.hypot(p8.x - p4.x, p8.y - p4.y)
                    scale = math.hypot(p9.x - p0.x, p9.y - p0.y)
                    scale = max(scale, 0.01)

                    ratio = pinch / scale
                    ratio_val = ratio

                    # Pixel coords
                    x4, y4 = int(p4.x * w), int(p4.y * h)
                    x8, y8 = int(p8.x * w), int(p8.y * h)

                    color = (0, 255, 0)
                    if ratio <= PINCH_RATIO_THRESHOLD:
                        color = (0, 0, 255)

                    # Draw pinch visuals
                    cv2.circle(frame, (x4, y4), 8, (255, 0, 255), -1)
                    cv2.circle(frame, (x8, y8), 8, (255, 0, 255), -1)
                    cv2.line(frame, (x4, y4), (x8, y8), color, LINE_THICKNESS)

                    dist_px = int(math.hypot(x8 - x4, y8 - y4))
                    cv2.putText(
                        frame,
                        f"{dist_px}px",
                        (min(x4, x8), min(y4, y8) - 10),
                        FONT, 0.6, color, 2
                    )

                    if vc:
                        try:
                            if ratio <= PINCH_RATIO_THRESHOLD:
                                vc.SetMute(1, None)
                                muted = True
                            else:
                                vc.SetMute(0, None)
                                muted = False

                                vol = np.interp(
                                    ratio,
                                    [MIN_RATIO, MAX_RATIO],
                                    [0, 100]
                                )
                                vol = SMOOTHING * round(vol / SMOOTHING)
                                vol = int(np.clip(vol, 0, 100))
                                vc.SetMasterVolumeLevelScalar(vol / 100, None)
                        except:
                            pass

            # =======================
            # UI METRICS
            # =======================
            if vc:
                try:
                    curr_vol = int(vc.GetMasterVolumeLevelScalar() * 100)
                    vol_metric.metric("Mic Volume", f"{curr_vol}%")
                except:
                    curr_vol = 0

            ratio_metric.metric("Gesture Ratio", f"{ratio_val:.2f}")

            if muted:
                status_msg.error("MICROPHONE MUTED")
            else:
                status_msg.success("MICROPHONE ACTIVE")

            # =======================
            # VOLUME BAR OVERLAY
            # =======================
            bar_x, bar_y = 40, 60
            bar_w, bar_h = 25, 250

            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h),
                          (50, 50, 50), -1)

            fill_h = int((curr_vol / 100) * bar_h)
            cv2.rectangle(
                frame,
                (bar_x, bar_y + bar_h - fill_h),
                (bar_x + bar_w, bar_y + bar_h),
                (0, 200, 255), -1
            )

            cv2.putText(
                frame,
                f"{curr_vol}%",
                (bar_x - 10, bar_y + bar_h + 30),
                FONT, 0.6, (255, 255, 255), 2
            )

            video_container.image(frame, channels="BGR", width=650)

        cap.release()
        cv2.destroyAllWindows()
        st.write("Camera stopped.")