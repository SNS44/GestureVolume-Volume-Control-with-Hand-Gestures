# 🎙️ Hand Gesture–Based Volume Control

A real-time **gesture-controlled volume system** built with **Streamlit**, **MediaPipe**, **OpenCV**, and **PyCAW**. The application allows users to **mute/unmute** and **adjust microphone volume** using **thumb–index finger gestures**, with scale-invariant normalization for stability.

---

## 🚀 Features

* ✋ **Real-time hand tracking** using MediaPipe (single-hand, low-latency)
* 🔇 **Pinch-to-mute** gesture (thumb + index close)
* 🔊 **Continuous microphone volume control** using finger distance
* 📐 **Scale-invariant ratio calculation** (robust to hand distance from camera)
* 🎛️ **Manual volume controls** (+20% / −20%) in sidebar
* 🎥 **Live camera feed** with visual landmarks
* 🧊  built with Streamlit + custom CSS

---

## 🧠 How It Works (Core Logic)

1. Capture frames from webcam using OpenCV
2. Detect hand landmarks using MediaPipe Hands
3. Extract key landmarks:

   * Thumb tip (ID 4)
   * Index tip (ID 8)
   * Wrist (ID 0)
   * Middle finger MCP (ID 9)
4. Compute:

```
Pinch Ratio = Distance(Thumb, Index) / Distance(Wrist, Middle MCP)
```

5. Apply logic:

   * **Ratio ≤ 0.15** → Microphone muted
   * **Ratio > 0.15** → Map ratio to volume (0–100%)
6. Set microphone volume using **Windows Core Audio API (PyCAW)**

---

## 🖥️ System Requirements

| Requirement | Details                            |
| ----------- | ---------------------------------- |
| OS          | **Windows 10 / 11 only**           |
| Python      | 3.9 – 3.11 (3.10 recommended)      |
| Hardware    | Webcam + Microphone                |
| Permissions | Camera & Microphone access enabled |

> ⚠️ This project will **not work on macOS or Linux** due to Windows-specific audio APIs.

---

## 📦 Dependencies

Install the required libraries:

```bash
pip install streamlit opencv-python mediapipe numpy pycaw comtypes
```

---

## ▶️ How to Run

1. Clone the repository

```bash
git clone <repo-url>
cd <repo-folder>
```

2. (Recommended) Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

3. Run the Streamlit app

```bash
streamlit run main.py
```

4. Open the browser at:

```
http://localhost:8501
```

---

## 🎮 Usage Instructions

1. Click **START CAMERA**
2. Show **one hand only** to the webcam
3. Gestures:

   * 🤏 **Thumb + Index close** → Mute microphone
   * ↔️ **Increase distance** → Increase mic volume
   * ↕️ **Decrease distance** → Decrease mic volume
4. Use sidebar buttons for manual volume control if needed

---

## ⚙️ Configuration Parameters

You can fine-tune gesture sensitivity in the code:

```python
PINCH_RATIO_THRESHOLD = 0.15  # Mute threshold
MIN_RATIO = 0.15             # 0% volume
MAX_RATIO = 1.5              # 100% volume
SMOOTHING = 3                # Volume smoothing step
```

---

## 🧪 Known Limitations

* Single-hand tracking only
* Requires good lighting for accurate detection
* Background clutter may affect hand tracking
* Windows-only microphone control

---

## 🧠 System Architecture

### High-Level Architecture

```
┌──────────────┐
│   Webcam     │
│ (Video Feed) │
└──────┬───────┘
       │ OpenCV
       ▼
┌────────────────────────┐
│  Frame Preprocessing   │
│  - Flip
│  - RGB Conversion      │
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│  MediaPipe Hands       │
│  - Landmark Detection  │
│  - 21 key points       │
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│ Gesture Analysis Logic │
│ - Thumb–Index Distance│
│ - Wrist–MCP Scale     │
│ - Ratio Normalization │
└──────┬─────────────────┘
       │
       ▼
┌────────────────────────┐
│ Decision Engine        │
│ - Pinch → Mute         │
│ - Spread → Volume Map  │
└──────┬─────────────────┘
       │ PyCAW (COM API)
       ▼
┌────────────────────────┐
│ Windows Audio Engine   │
│ (Microphone Control)  │
└────────────────────────┘
       ▲
       │
┌────────────────────────┐
│ Streamlit UI Layer     │
│ - Metrics              │
│ - Status               │
│ - Live Video           │
└────────────────────────┘
```

### Architectural Design Choices

* **Scale-invariant ratio** prevents false volume changes due to hand depth
* **Single-hand tracking** reduces latency and CPU usage
* **Cached MediaPipe model** avoids repeated initialization
* **COM interface instantiated once** for stability
* **UI + processing loop tightly synchronized** via Streamlit session state

---

## 🔁 Processing Flowchart

### Runtime Execution Flow

```
START
  │
  ▼
Initialize Streamlit UI
  │
  ▼
Load MediaPipe Hand Model (cached)
  │
  ▼
User clicks START CAMERA
  │
  ▼
Open Webcam Stream
  │
  ▼
Read Frame
  │
  ▼
Convert BGR → RGB
  │
  ▼
Detect Hand Landmarks
  │
  ▼
Hand Detected?
  ├── NO → Show camera feed only
  │
  └── YES
        │
        ▼
   Extract Key Landmarks
        │
        ▼
   Compute Pinch Ratio
        │
        ▼
   Ratio ≤ Threshold?
        ├── YES → MUTE MICROPHONE
        │
        └── NO  → Map Ratio → Volume %
                  │
                  ▼
             Set Mic Volume
        
  │
  ▼
Update UI Metrics & Status
  │
  ▼
Display Annotated Frame
  │
  ▼
STOP CAMERA?
  ├── NO → Loop
  └── YES → Release Camera & Exit
```

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit** – UI & app framework
* **MediaPipe** – Hand landmark detection
* **OpenCV** – Video processing
* **PyCAW** – Windows microphone control
* **NumPy** – Numeric mapping & interpolation

---

## 📜 License

This project is intended for **educational and academic use**.

---

## ✅ Evaluation Readiness

✔ Real-time system
✔ Hardware interaction
✔ OS-level API usage
✔ Gesture normalization
✔ Production-style UI

This is **not a toy project**. It demonstrates applied computer vision + system programming.
