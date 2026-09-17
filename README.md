# 🧠 Emotion Detection System

A real-time facial emotion recognition desktop application. It reads your
webcam feed, detects faces with OpenCV's Haar Cascade classifier, and
classifies each face into **Angry**, **Happy**, or **Sad** using a
Convolutional Neural Network (CNN) — all through a modern, dark-themed
Tkinter GUI.

![Python](https://img.shields.io/badge/Python-3.9%20--%203.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Model details](#model-details)
- [Logging](#logging)
- [Snapshots](#snapshots)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

This project turns a laptop webcam into a live emotion analyzer. It combines
a classical computer-vision face detector (Haar Cascade) with a deep-learning
classifier (a small CNN built on Keras/TensorFlow) to label each detected
face with an emotion and a confidence score, in real time, inside a desktop
GUI — no browser or server required.

It was built as an end-to-end demo of a computer-vision pipeline: **capture →
detect → preprocess → classify → visualize → log**.

## Features

- 🎥 **Live webcam feed** with colored bounding boxes and emotion labels
  drawn directly on the video
- 📊 **Real-time confidence bars** for every emotion class (Angry / Happy /
  Sad), updated frame by frame
- 📈 **Session stats panel**: live FPS, session duration, total faces
  detected, and the most frequently detected emotion
- 🟢 **Status indicators** for camera state (running/stopped) and model load
  state
- 📷 **One-click snapshot capture** — saves the current annotated frame to
  `snapshots/`
- 🗂️ **CSV detection logging** — every detection is appended to
  `logs/emotion_log.csv` with a timestamp and confidence score, ready for
  further analysis in Excel/Pandas
- 🖥️ **Responsive video panel** that scales the feed to fit the window while
  preserving aspect ratio
- 🎨 A clean, dark, professional UI built entirely with Tkinter (no external
  UI framework)

## How it works

```
Webcam frame
     │
     ▼
Grayscale conversion (cv2.cvtColor)
     │
     ▼
Face detection (Haar Cascade — face_detector.xml)
     │  for each detected face:
     ▼
Crop face → resize to 48×48 → normalize (0–1)
     │
     ▼
CNN forward pass (emotion_model.h5)  →  [P(Angry), P(Happy), P(Sad)]
     │
     ▼
argmax → label + confidence
     │
     ├──▶ Draw box + label on frame → show in GUI
     ├──▶ Update live probability bars / session stats
     └──▶ Append row to logs/emotion_log.csv
```

The whole loop runs on Tkinter's `after()` scheduler (no extra threads
needed), so the UI stays responsive while frames are continuously grabbed,
processed, and rendered roughly every 10 ms.

## Tech stack

| Layer            | Library / Tool                        |
|-------------------|----------------------------------------|
| GUI               | Tkinter (Python standard library)      |
| Video capture      | OpenCV (`cv2.VideoCapture`)           |
| Face detection     | OpenCV Haar Cascade (`CascadeClassifier`) |
| Emotion classifier  | TensorFlow / Keras CNN (`.h5` model)  |
| Image handling      | Pillow (`PIL.Image`, `ImageTk`)       |
| Numerics            | NumPy                                 |

## Project structure

```
EmotionDetect/
├── detect_gui.py         # Main application — GUI + real-time detection loop
├── train.py               # Model training script (placeholder — see note below)
├── emotion_model.h5        # Pre-trained Keras CNN (Angry / Happy / Sad), 48×48 grayscale input
├── face_detector.xml       # OpenCV Haar Cascade for frontal face detection
├── requirements.txt         # Pinned Python dependencies
├── logs/
│   └── emotion_log.csv      # Generated at runtime — timestamp, emotion, confidence
├── snapshots/                # Generated at runtime — saved PNG captures
└── dataset/                   # (not tracked in git — see "Dataset" note below)
```

> **Dataset note:** the `dataset/` folder used to originally train the model
> (FER2013-style face crops, split by emotion) is intentionally excluded from
> version control via `.gitignore` — it's thousands of small image files and
> isn't needed to *run* the app, since the trained model (`emotion_model.h5`)
> is already included.

> **About `train.py`:** trains a CNN from scratch on `dataset/train` and
> `dataset/test` (grayscale 48×48 crops, `angry`/`happy`/`sad` folders) and
> saves the result as an `.h5` model compatible with `detect_gui.py`. See
> [Training](#training) below.

## Installation

**Requirements:** Python 3.9 – 3.12 and a webcam. (TensorFlow does not yet
publish wheels for Python 3.13+, so a newer interpreter will fail to install
it — see [Troubleshooting](#troubleshooting).)

```bash
# 1. Clone the repo
git clone https://github.com/awaishassan321/emotion-detection-system.git
cd emotion-detection-system

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python detect_gui.py
```

1. Click **▶ Start Camera** to open the webcam and begin live detection.
2. Detected faces are outlined with a colored box and an emotion + confidence
   label; the sidebar shows live per-emotion probability bars.
3. Click **📷 Save Snapshot** at any time to save the current annotated frame
   to `snapshots/`.
4. Click **■ Stop Camera** to release the webcam, or **✕ Exit** to close the
   app.

The footer shows status messages (e.g. snapshot saved) and the path to the
active log file.

## Training

`train.py` retrains the CNN from scratch using the images in `dataset/train`
and `dataset/test` (not included in this repo — see the note above).

```bash
python train.py                                # default: 40 epochs, batch size 64
python train.py --epochs 25 --batch-size 32
python train.py --output emotion_model.h5      # overwrite the shipped model directly
```

By default the trained model is saved to `emotion_model_trained.h5` (not
`emotion_model.h5`) so a training run never silently overwrites the working
demo model — copy/rename it yourself once you're happy with the result, or
pass `--output emotion_model.h5` to replace it directly.

Key details:

- Class imbalance (train set: ~1.1k *angry* vs. ~7.2k *happy* vs. ~4.8k *sad*)
  is corrected with computed `class_weight`s.
- Data augmentation (random flip/rotation/zoom) is applied only at train time.
- `EarlyStopping` + `ReduceLROnPlateau` avoid overfitting and stop training
  once validation accuracy plateaus.
- Per-epoch metrics are written to `logs/training_history.csv`.
- Training runs on CPU by default (no CUDA/DirectML setup required) — a full
  run takes roughly a few minutes per epoch on a typical laptop CPU.

## Model details

- **Input:** 48×48 grayscale face crop, pixel values normalized to `[0, 1]`
- **Output:** 3-class softmax — `["Angry", "Happy", "Sad"]`
- **Format:** Keras `.h5` (`tensorflow.keras.models.load_model`)
- **Face detector:** OpenCV's stock Haar Cascade frontal-face model, tuned
  with `scaleFactor=1.3`, `minNeighbors=5`

## Logging

Every detection is appended to `logs/emotion_log.csv` (writes are throttled
to at most ~3 per second to keep the file manageable):

| column       | description                              |
|--------------|-------------------------------------------|
| `timestamp`  | `YYYY-MM-DD HH:MM:SS` of the detection    |
| `emotion`    | `Angry`, `Happy`, or `Sad`                |
| `confidence` | Softmax probability of the predicted class (0–1) |

This makes it easy to load the log into Pandas/Excel for post-session
analysis (e.g. mood trends over a call, a study session, etc.).

## Snapshots

Snapshots are saved as `snapshot_YYYYMMDD_HHMMSS.png` inside `snapshots/`,
capturing the frame exactly as shown on screen (with boxes and labels
burned in).

## Troubleshooting

- **`Could not find a version that satisfies the requirement tensorflow`** —
  you're on a Python version TensorFlow doesn't support yet (e.g. 3.13/3.14).
  Install Python 3.11 or 3.12 and recreate the virtual environment.
- **`AttributeError: module 'cv2' has no attribute 'CascadeClassifier'`** —
  you have a pre-release `opencv-python` (e.g. `5.0.0`) that doesn't ship the
  Haar Cascade module. Pin to a stable release:
  `pip install opencv-python==4.10.0.84`.
- **"Could not open the camera"** — another application may be using the
  webcam, or the wrong camera index is selected. Close other apps using the
  camera and retry.
- **Model file not found** — make sure `emotion_model.h5` sits next to
  `detect_gui.py` (the app resolves paths relative to its own file location).

## Known limitations

- Only three emotion classes are supported (Angry, Happy, Sad) — no
  Neutral/Surprise/Fear/Disgust classes yet.
- Haar Cascade face detection is fast but less robust than modern DNN-based
  detectors under poor lighting, extreme angles, or occlusion.
- Single-threaded capture/inference loop — very high-resolution webcams may
  reduce the effective FPS.
- No GPU acceleration on native Windows for TensorFlow ≥ 2.11 (CPU-only
  inference; still real-time for this model's small input size).

## Roadmap

- [x] Add a real `train.py` pipeline (load `dataset/`, train, evaluate, export)
- [ ] Expand to more emotion classes (Neutral, Surprise, Fear, Disgust)
- [ ] Swap Haar Cascade for a DNN-based face detector for better accuracy
- [ ] Add a session summary/report view (charts from `logs/emotion_log.csv`)
- [ ] Camera source selector in the UI (multiple webcams)

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for
details.
