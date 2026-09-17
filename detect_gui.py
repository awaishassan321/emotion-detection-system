import os
import csv
import time
from datetime import datetime

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# ==================================================================
#  CONFIG
# ==================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.h5")
CASCADE_PATH = os.path.join(BASE_DIR, "face_detector.xml")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_PATH = os.path.join(LOG_DIR, "emotion_log.csv")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")

EMOTIONS = ["Angry", "Happy", "Sad"]
IMG_SIZE = 48

EMOTION_COLOR = {"Angry": "#e05263", "Happy": "#2ecc71", "Sad": "#5b8def"}
EMOTION_EMOJI = {"Angry": "\U0001F620", "Happy": "\U0001F60A", "Sad": "\U0001F622"}

LOG_MIN_INTERVAL = 0.3  # seconds between CSV writes (avoid flooding the log)

# ==================================================================
#  THEME
# ==================================================================
BG_APP = "#12141c"
BG_SIDEBAR = "#181b26"
BG_CARD = "#1f2330"
BG_CARD_ALT = "#262b3a"
BORDER = "#2f3342"
ACCENT = "#5b8def"
ACCENT_HOVER = "#4676c9"
SUCCESS = "#2ecc71"
SUCCESS_HOVER = "#27ae60"
DANGER = "#e05263"
DANGER_HOVER = "#c8455a"
NEUTRAL = "#3a3f52"
NEUTRAL_HOVER = "#4a4f66"
TEXT_PRIMARY = "#eef0f5"
TEXT_SECONDARY = "#8a8f9c"
TEXT_MUTED = "#5f6474"

FONT = "Segoe UI"


def ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "emotion", "confidence"])


class FlatButton(tk.Button):
    """A flat, hover-aware button matching the app's dark theme."""

    def __init__(self, master, bg, hover_bg, **kwargs):
        super().__init__(
            master,
            bg=bg,
            fg=TEXT_PRIMARY,
            activebackground=hover_bg,
            activeforeground=TEXT_PRIMARY,
            disabledforeground=TEXT_MUTED,
            relief="flat",
            bd=0,
            cursor="hand2",
            font=(FONT, 10, "bold"),
            highlightthickness=0,
            **kwargs,
        )
        self._bg = bg
        self._hover = hover_bg
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _on_enter(self, _):
        if self["state"] != "disabled":
            self.config(bg=self._hover)

    def _on_leave(self, _):
        if self["state"] != "disabled":
            self.config(bg=self._bg)


class ProbabilityBar(tk.Frame):
    """A labeled horizontal bar showing an emotion's live confidence."""

    def __init__(self, master, name, color, width=196, height=10):
        super().__init__(master, bg=BG_CARD)
        self.width = width
        self.height = height
        self.color = color

        top = tk.Frame(self, bg=BG_CARD)
        top.pack(fill="x")
        tk.Label(
            top, text=f"{EMOTION_EMOJI[name]}  {name}", bg=BG_CARD,
            fg=TEXT_PRIMARY, font=(FONT, 10), anchor="w",
        ).pack(side="left")
        self.pct_label = tk.Label(
            top, text="0%", bg=BG_CARD, fg=TEXT_SECONDARY,
            font=(FONT, 10, "bold"), anchor="e",
        )
        self.pct_label.pack(side="right")

        self.canvas = tk.Canvas(
            self, width=width, height=height, bg=BG_CARD_ALT,
            highlightthickness=0, bd=0,
        )
        self.canvas.pack(fill="x", pady=(4, 10))
        self.bar = self.canvas.create_rectangle(0, 0, 0, height, fill=color, width=0)

    def set_value(self, fraction):
        fraction = max(0.0, min(1.0, fraction))
        self.canvas.coords(self.bar, 0, 0, self.width * fraction, self.height)
        self.pct_label.config(text=f"{int(fraction * 100)}%")


class Card(tk.Frame):
    """A titled card container used throughout the sidebar."""

    def __init__(self, master, title):
        super().__init__(master, bg=BG_CARD, highlightbackground=BORDER,
                          highlightthickness=1, bd=0)
        header = tk.Frame(self, bg=BG_CARD)
        header.pack(fill="x", padx=14, pady=(12, 6))
        tk.Label(
            header, text=title.upper(), bg=BG_CARD, fg=TEXT_MUTED,
            font=(FONT, 9, "bold"),
        ).pack(side="left")
        self.body = tk.Frame(self, bg=BG_CARD)
        self.body.pack(fill="both", expand=True, padx=14, pady=(0, 14))


class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detection System")
        self.root.geometry("1180x700")
        self.root.minsize(1020, 620)
        self.root.configure(bg=BG_APP)
        self.root.protocol("WM_DELETE_WINDOW", self.exit_app)

        ensure_dirs()

        self.cap = None
        self.running = False
        self.camera_index = 0
        self.last_frame = None
        self.last_log_time = 0.0
        self.session_start = None
        self.frame_times = []
        self.face_count_total = 0
        self.emotion_counts = {name: 0 for name in EMOTIONS}

        self.model = None
        self.model_error = None
        self._load_model()

        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        self.cascade_error = self.face_cascade.empty()

        self._build_gui()
        self._tick_clock()
        self._tick_session_timer()

        if self.model_error:
            messagebox.showerror("Model Error", self.model_error)
        if self.cascade_error:
            messagebox.showerror("Error", "Face detector cascade file could not be loaded.")

    # ---------------- MODEL ----------------
    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            self.model_error = f"Model file not found:\n{MODEL_PATH}"
            return
        try:
            self.model = load_model(MODEL_PATH)
        except Exception as exc:  # noqa: BLE001 - surface any load failure to the user
            self.model_error = f"Failed to load model:\n{exc}"

    # ================================================================
    #  GUI CONSTRUCTION
    # ================================================================
    def _build_gui(self):
        self._build_header()

        body = tk.Frame(self.root, bg=BG_APP)
        body.pack(fill="both", expand=True)

        self._build_sidebar(body)
        self._build_video_area(body)

        self._build_footer()

    def _build_header(self):
        header = tk.Frame(self.root, bg=BG_SIDEBAR, height=64)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        left = tk.Frame(header, bg=BG_SIDEBAR)
        left.pack(side="left", padx=20)
        tk.Label(
            left, text="\U0001F9E0  Emotion Detection System", bg=BG_SIDEBAR,
            fg=TEXT_PRIMARY, font=(FONT, 15, "bold"),
        ).pack(anchor="w", pady=(10, 0))
        tk.Label(
            left, text="Real-time facial emotion recognition  •  CNN + Haar Cascade",
            bg=BG_SIDEBAR, fg=TEXT_SECONDARY, font=(FONT, 9),
        ).pack(anchor="w")

        right = tk.Frame(header, bg=BG_SIDEBAR)
        right.pack(side="right", padx=20)
        self.clock_label = tk.Label(
            right, text="--:--:--", bg=BG_SIDEBAR, fg=TEXT_PRIMARY,
            font=(FONT, 13, "bold"),
        )
        self.clock_label.pack(anchor="e", pady=(10, 0))
        tk.Label(
            right, text=datetime.now().strftime("%A, %d %B %Y"), bg=BG_SIDEBAR,
            fg=TEXT_SECONDARY, font=(FONT, 9),
        ).pack(anchor="e")

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

    def _build_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=BG_APP, width=280)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        pad = dict(fill="x", padx=16, pady=(16, 0))

        # ---- Status card ----
        status_card = Card(sidebar, "Status")
        status_card.pack(**pad)

        row = tk.Frame(status_card.body, bg=BG_CARD)
        row.pack(fill="x", pady=2)
        self.status_dot = tk.Canvas(row, width=10, height=10, bg=BG_CARD, highlightthickness=0)
        self.status_dot.pack(side="left", padx=(0, 8))
        self.status_dot_id = self.status_dot.create_oval(0, 0, 10, 10, fill=DANGER, width=0)
        self.status_text = tk.Label(
            row, text="Camera stopped", bg=BG_CARD, fg=TEXT_SECONDARY, font=(FONT, 10),
        )
        self.status_text.pack(side="left")

        model_state = "Loaded" if self.model is not None else "Failed to load"
        model_color = SUCCESS if self.model is not None else DANGER
        tk.Label(
            status_card.body, text=f"Model: {model_state}", bg=BG_CARD,
            fg=model_color, font=(FONT, 9, "bold"),
        ).pack(anchor="w", pady=(8, 0))

        # ---- Controls card ----
        controls_card = Card(sidebar, "Controls")
        controls_card.pack(**pad)

        self.start_btn = FlatButton(
            controls_card.body, SUCCESS, SUCCESS_HOVER, text="▶  Start Camera",
            command=self.start_camera,
        )
        self.start_btn.pack(fill="x", ipady=8, pady=(0, 8))

        self.stop_btn = FlatButton(
            controls_card.body, DANGER, DANGER_HOVER, text="■  Stop Camera",
            command=self.stop_camera, state="disabled",
        )
        self.stop_btn.pack(fill="x", ipady=8, pady=(0, 8))

        self.snapshot_btn = FlatButton(
            controls_card.body, NEUTRAL, NEUTRAL_HOVER, text="\U0001F4F7  Save Snapshot",
            command=self.save_snapshot, state="disabled",
        )
        self.snapshot_btn.pack(fill="x", ipady=8, pady=(0, 8))

        FlatButton(
            controls_card.body, BG_CARD_ALT, NEUTRAL_HOVER, text="✕  Exit",
            command=self.exit_app,
        ).pack(fill="x", ipady=8)

        # ---- Live emotion card ----
        live_card = Card(sidebar, "Live Emotion")
        live_card.pack(**pad)

        self.emotion_big_label = tk.Label(
            live_card.body, text="—", bg=BG_CARD, fg=TEXT_PRIMARY,
            font=(FONT, 22, "bold"),
        )
        self.emotion_big_label.pack(anchor="w")
        self.confidence_label = tk.Label(
            live_card.body, text="No face detected", bg=BG_CARD, fg=TEXT_SECONDARY,
            font=(FONT, 9),
        )
        self.confidence_label.pack(anchor="w", pady=(0, 10))

        self.bars = {}
        for name in EMOTIONS:
            bar = ProbabilityBar(live_card.body, name, EMOTION_COLOR[name])
            bar.pack(fill="x")
            self.bars[name] = bar

        # ---- Session stats card ----
        stats_card = Card(sidebar, "Session Stats")
        stats_card.pack(**pad)

        self.session_time_label = self._stat_row(stats_card.body, "Session time", "00:00:00")
        self.fps_label = self._stat_row(stats_card.body, "FPS", "0.0")
        self.faces_label = self._stat_row(stats_card.body, "Faces detected", "0")
        self.dominant_label = self._stat_row(stats_card.body, "Most frequent", "—")

    def _stat_row(self, parent, label, value):
        row = tk.Frame(parent, bg=BG_CARD)
        row.pack(fill="x", pady=3)
        tk.Label(row, text=label, bg=BG_CARD, fg=TEXT_SECONDARY, font=(FONT, 9)).pack(side="left")
        val = tk.Label(row, text=value, bg=BG_CARD, fg=TEXT_PRIMARY, font=(FONT, 9, "bold"))
        val.pack(side="right")
        return val

    def _build_video_area(self, parent):
        wrapper = tk.Frame(parent, bg=BG_APP)
        wrapper.pack(side="right", fill="both", expand=True, padx=16, pady=16)

        video_frame = tk.Frame(wrapper, bg="#000000", highlightbackground=BORDER,
                                highlightthickness=1)
        video_frame.pack(fill="both", expand=True)

        self.video_label = tk.Label(video_frame, bg="#000000")
        self.video_label.pack(fill="both", expand=True)
        self._show_placeholder("Press ▶ Start Camera to begin")

    def _show_placeholder(self, message):
        self.video_label.configure(
            image="", text=message, fg=TEXT_MUTED, bg="#000000",
            font=(FONT, 13),
        )

    def _build_footer(self):
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x", side="bottom")
        footer = tk.Frame(self.root, bg=BG_SIDEBAR, height=28)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)

        self.footer_status = tk.Label(
            footer, text="Ready.", bg=BG_SIDEBAR, fg=TEXT_SECONDARY, font=(FONT, 9),
        )
        self.footer_status.pack(side="left", padx=14)

        tk.Label(
            footer, text=f"Log: {os.path.relpath(LOG_PATH, BASE_DIR)}",
            bg=BG_SIDEBAR, fg=TEXT_MUTED, font=(FONT, 9),
        ).pack(side="right", padx=14)

    def set_status(self, message):
        self.footer_status.config(text=message)

    # ================================================================
    #  CLOCK / SESSION TIMER
    # ================================================================
    def _tick_clock(self):
        self.clock_label.config(text=datetime.now().strftime("%H:%M:%S"))
        self.root.after(1000, self._tick_clock)

    def _tick_session_timer(self):
        if self.running and self.session_start:
            elapsed = int(time.time() - self.session_start)
            h, rem = divmod(elapsed, 3600)
            m, s = divmod(rem, 60)
            self.session_time_label.config(text=f"{h:02d}:{m:02d}:{s:02d}")
        self.root.after(1000, self._tick_session_timer)

    # ================================================================
    #  CAMERA CONTROL
    # ================================================================
    def start_camera(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded, cannot start detection.")
            return
        if self.cascade_error:
            messagebox.showerror("Error", "Face detector not loaded, cannot start detection.")
            return
        if self.running:
            return

        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open the camera.")
            self.cap = None
            return

        self.running = True
        self.session_start = time.time()
        self.frame_times.clear()

        self.status_dot.itemconfig(self.status_dot_id, fill=SUCCESS)
        self.status_text.config(text="Camera running", fg=SUCCESS)
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.snapshot_btn.config(state="normal")
        self.set_status("Camera started — detecting faces in real time.")

        self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.status_dot.itemconfig(self.status_dot_id, fill=DANGER)
        self.status_text.config(text="Camera stopped", fg=TEXT_SECONDARY)
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.snapshot_btn.config(state="disabled")
        self.session_time_label.config(text="00:00:00")
        self._show_placeholder("Press ▶ Start Camera to begin")
        self.set_status("Camera stopped.")

    # ================================================================
    #  FRAME PROCESSING LOOP
    # ================================================================
    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        self._track_fps()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        top_emotion, top_conf, top_probs = None, 0.0, None

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = face / 255.0
            face = np.reshape(face, (1, IMG_SIZE, IMG_SIZE, 1))

            preds = self.model.predict(face, verbose=0)[0]
            idx = int(np.argmax(preds))
            emotion = EMOTIONS[idx]
            confidence = float(preds[idx])

            color_hex = EMOTION_COLOR[emotion]
            color_bgr = self._hex_to_bgr(color_hex)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)
            label = f"{emotion} {confidence * 100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y - th - 14), (x + tw + 10, y), color_bgr, -1)
            cv2.putText(frame, label, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

            if top_emotion is None:
                top_emotion, top_conf, top_probs = emotion, confidence, preds

            self.emotion_counts[emotion] += 1
            self.face_count_total += 1
            self._maybe_log(emotion, confidence)

        self._update_stats_panel(len(faces), top_emotion, top_conf, top_probs)

        self.last_frame = frame
        self._render_frame(frame)

        self.root.after(10, self.update_frame)

    def _hex_to_bgr(self, hex_color):
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        return (b, g, r)

    def _track_fps(self):
        now = time.time()
        self.frame_times.append(now)
        cutoff = now - 2.0
        self.frame_times = [t for t in self.frame_times if t >= cutoff]
        if len(self.frame_times) >= 2:
            span = self.frame_times[-1] - self.frame_times[0]
            fps = (len(self.frame_times) - 1) / span if span > 0 else 0.0
            self.fps_label.config(text=f"{fps:.1f}")

    def _maybe_log(self, emotion, confidence):
        now = time.time()
        if now - self.last_log_time >= LOG_MIN_INTERVAL:
            self.last_log_time = now
            with open(LOG_PATH, "a", newline="") as f:
                csv.writer(f).writerow(
                    [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), emotion, f"{confidence:.4f}"]
                )

    def _update_stats_panel(self, face_count, emotion, confidence, probs):
        if emotion is not None:
            self.emotion_big_label.config(
                text=f"{EMOTION_EMOJI[emotion]}  {emotion}", fg=EMOTION_COLOR[emotion],
            )
            self.confidence_label.config(text=f"Confidence: {confidence * 100:.1f}%")
            for i, name in enumerate(EMOTIONS):
                self.bars[name].set_value(float(probs[i]))
        else:
            self.emotion_big_label.config(text="—", fg=TEXT_PRIMARY)
            self.confidence_label.config(text="No face detected")
            for name in EMOTIONS:
                self.bars[name].set_value(0.0)

        self.faces_label.config(text=str(self.face_count_total))
        if any(self.emotion_counts.values()):
            dominant = max(self.emotion_counts, key=self.emotion_counts.get)
            self.dominant_label.config(text=f"{EMOTION_EMOJI[dominant]} {dominant}")

    def _render_frame(self, frame):
        label_w = max(self.video_label.winfo_width(), 640)
        label_h = max(self.video_label.winfo_height(), 480)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        img_ratio = img.width / img.height
        box_ratio = label_w / label_h
        if img_ratio > box_ratio:
            new_w = label_w
            new_h = int(label_w / img_ratio)
        else:
            new_h = label_h
            new_w = int(label_h * img_ratio)
        img = img.resize((max(new_w, 1), max(new_h, 1)), Image.LANCZOS)

        photo = ImageTk.PhotoImage(img)
        self.video_label.imgtk = photo
        self.video_label.configure(image=photo, text="")

    # ================================================================
    #  SNAPSHOT
    # ================================================================
    def save_snapshot(self):
        if self.last_frame is None:
            messagebox.showinfo("Snapshot", "No frame available to capture yet.")
            return
        filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(SNAPSHOT_DIR, filename)
        cv2.imwrite(path, self.last_frame)
        self.set_status(f"Snapshot saved: {filename}")

    # ================================================================
    #  EXIT
    # ================================================================
    def exit_app(self):
        self.stop_camera()
        self.root.destroy()


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()
