# Emotion Detection System

A real-time facial emotion recognition desktop app. It captures video from your
webcam, detects faces with a Haar Cascade classifier, and classifies each face
into **Angry**, **Happy**, or **Sad** using a CNN trained on 48x48 grayscale
face crops (FER2013-style data).

## Features

- Live webcam feed with bounding boxes and emotion labels
- Per-emotion confidence bars, updated in real time
- Session stats: FPS, elapsed session time, total faces detected, most
  frequent emotion
- One-click snapshot capture (saved to `snapshots/`)
- CSV logging of every detection (`logs/emotion_log.csv`) for later analysis
- Dark, modern desktop UI built with Tkinter

## Requirements

- Python 3.9 – 3.12 (TensorFlow does not yet support 3.13+)
- A webcam

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Run

```bash
python detect_gui.py
```

Click **Start Camera** to begin detection, **Save Snapshot** to capture the
current annotated frame, and **Stop Camera** to release the webcam.

## Project structure

```
EmotionDetect/
├── detect_gui.py        # Main application (Tkinter GUI + detection loop)
├── emotion_model.h5      # Pre-trained Keras CNN (Angry / Happy / Sad)
├── face_detector.xml     # OpenCV Haar Cascade for face detection
├── requirements.txt
├── logs/                 # emotion_log.csv is generated at runtime
└── snapshots/            # Saved snapshots are written here
```

> Note: the `dataset/` folder used to train the model is not included in this
> repository (it's a standard FER2013-style angry/happy/sad face dataset).

## Tech stack

TensorFlow / Keras, OpenCV, Pillow, NumPy, Tkinter.
