import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import csv
import time
import json
import os
import webbrowser
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from deepface import DeepFace
import math
import random
import queue

# ==============================
# DEEPFACE ASYNC WORKER
# ==============================
emotion_queue = queue.Queue(maxsize=15)
latest_emotions = {}


def emotion_worker():
    while True:
        try:
            item = emotion_queue.get()
            if item is None:
                break
            face_crop, detected_product, pid = item

            analysis = DeepFace.analyze(
                face_crop,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            if isinstance(analysis, list):
                analysis = analysis[0]
            dominant_emotion = analysis.get('dominant_emotion')

            if dominant_emotion:
                latest_emotions[pid] = dominant_emotion
                if detected_product and dominant_emotion in emotion_data.get(detected_product, {}):
                    emotion_data[detected_product][dominant_emotion] += 1
        except Exception as e:
            logging.error(f"Emotion analysis failed: {e}", exc_info=True)
        finally:
            emotion_queue.task_done()


# Start worker thread
import threading

threading.Thread(target=emotion_worker, daemon=True).start()

# ==============================
# INTERACTIVE SELECTION GLOBALS
# ==============================
drawing = False
ix, iy = -1, -1
temp_frame = None
display_frame = None
rect_box = None


def draw_mouse_cb(event, x, y, flags, param):
    global ix, iy, drawing, display_frame, temp_frame, rect_box

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        rect_box = (x, y, x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            display_frame = temp_frame.copy()
            cv2.rectangle(display_frame, (ix, iy), (x, y), (0, 255, 0), 2)
            rect_box = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        display_frame = temp_frame.copy()
        cv2.rectangle(display_frame, (ix, iy), (x, y), (0, 255, 0), 2)
        rect_box = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))


def select_region_interactive():
    global temp_frame, display_frame, rect_box, cap

    if cap is None or not cap.isOpened():
        temp_cap = cv2.VideoCapture(0)
        ret, frame = temp_cap.read()
        temp_cap.release()
    else:
        ret, frame = cap.read()

    if not ret:
        messagebox.showerror("Error", "Could not read from camera")
        return

    temp_frame = frame.copy()
    display_frame = frame.copy()
    rect_box = None

    window_name = "Draw Product Region (Press ENTER/SPACE to confirm, ESC to cancel)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_mouse_cb)

    while True:
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(20) & 0xFF
        if key == 13 or key == 32:
            break
        elif key == 27:
            rect_box = None
            break

    cv2.destroyWindow(window_name)

    if rect_box and (rect_box[2] > rect_box[0] and rect_box[3] > rect_box[1]):
        x1_entry.delete(0, tk.END)
        x1_entry.insert(0, str(rect_box[0]))
        y1_entry.delete(0, tk.END)
        y1_entry.insert(0, str(rect_box[1]))
        x2_entry.delete(0, tk.END)
        x2_entry.insert(0, str(rect_box[2]))
        y2_entry.delete(0, tk.END)
        y2_entry.insert(0, str(rect_box[3]))


# Trend analyzer functions (from trend_analyzer.py — integrated)
from trend_analyzer import load_sessions, aggregate, generate_report as _generate_report

# ==============================
# INITIAL SETUP
# ==============================
MODEL_PATH = "../../Downloads/face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    fallback_path = "face_landmarker.task"
    if os.path.exists(fallback_path):
        MODEL_PATH = fallback_path
    else:
        logging.warning(f"Model not found at {MODEL_PATH}. Face tracking may fail.")

# Support up to MAX_PERSONS at once
MAX_PERSONS = 8

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=MAX_PERSONS
)

face_detector = vision.FaceLandmarker.create_from_options(options)

cap = None
running = False
product_boxes = {}

# Per-product aggregated analytics (all persons combined)
gaze_data = {}
emotion_data = {}

# Per-person state: dict keyed by person index 0..N-1
person_state = {}

# Frame counter for throttled JSON writes
_frame_counter = 0
SESSION_JSON_PATH = os.path.join(os.path.dirname(__file__), "session_data.json")

# ==============================
# COLOR PALETTE — "Soft Daylight"
# ==============================
COLORS = {
    "bg_darkest": "#e8eaf2",
    "bg_dark": "#eef0f8",
    "bg_primary": "#f4f6ff",
    "bg_card": "#ffffff",
    "bg_card_alt": "#f0f2ff",
    "bg_elevated": "#e6e9f8",
    "bg_hover": "#dde1f5",
    "border": "#c9cde8",
    "border_glow": "#7c3aed",
    "accent": "#7c3aed",
    "accent_light": "#9d68f5",
    "accent_dim": "#6027cc",
    "accent_glow": "#c4b5fd",
    "success": "#059669",
    "success_glow": "#10b981",
    "danger": "#e11d48",
    "danger_glow": "#f43f5e",
    "warning": "#d97706",
    "text_primary": "#1e1b4b",
    "text_secondary": "#4c4882",
    "text_muted": "#8b89b0",
    "text_accent": "#7c3aed",
}

# ==============================
# SPLASH-ONLY — Neon Surveillance palette
# ==============================
SPLASH = {
    "bg": "#04080f",
    "grid": "#0a1520",
    "bracket": "#0097a7",
    "ring_outer": "#0097a7",
    "ring_inner": "#00e5ff",
    "icon": "#00e5ff",
    "title": "#e8f4fd",
    "subtitle": "#90b8d4",
    "muted": "#3d6278",
    "scan": "#00e5ff",
    "underline": "#00e5ff",
}

# Distinct per-person colors for the camera overlay (BGR)
PERSON_COLORS_BGR = [
    (124, 58, 237),
    (16, 185, 129),
    (249, 115, 22),
    (236, 72, 153),
    (59, 130, 246),
    (234, 179, 8),
    (239, 68, 68),
    (168, 85, 247),
]


# ==============================
# DATA BRIDGE — write shared JSON for dashboard_server.py
# ==============================
def save_session_json():
    """Write current session data to session_data.json for the live dashboard."""
    try:
        payload = {
            "timestamp": datetime.now().isoformat(),
            "active_persons": len(person_state),
            "max_persons": MAX_PERSONS,
            "gaze_data": {
                p: {"fixations": d["fixations"], "total_time": round(d["total_time"], 3)}
                for p, d in gaze_data.items()
            },
            "emotion_data": emotion_data,
        }
        with open(SESSION_JSON_PATH, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass  # never crash the main loop for a JSON write failure


# ==============================
# MODERN UI THEME — Soft Daylight
# ==============================
def apply_theme():
    style = ttk.Style()
    style.theme_use("clam")

    style.configure("TLabel",
                    background=COLORS["bg_primary"],
                    foreground=COLORS["text_primary"],
                    font=("Segoe UI", 10))

    style.configure("Header.TLabel",
                    font=("Segoe UI", 24, "bold"),
                    background=COLORS["bg_primary"],
                    foreground=COLORS["accent"])

    style.configure("Sub.TLabel",
                    font=("Segoe UI", 11),
                    background=COLORS["bg_primary"],
                    foreground=COLORS["text_secondary"])

    style.configure("CardTitle.TLabel",
                    font=("Segoe UI", 13, "bold"),
                    background=COLORS["bg_card"],
                    foreground=COLORS["accent"])

    style.configure("Dark.TEntry",
                    fieldbackground=COLORS["bg_elevated"],
                    foreground=COLORS["text_primary"],
                    insertcolor=COLORS["accent"],
                    borderwidth=0,
                    padding=8)

    style.map("Dark.TEntry",
              fieldbackground=[("focus", COLORS["bg_dark"])])

    style.configure("Treeview",
                    background=COLORS["bg_card"],
                    foreground=COLORS["text_primary"],
                    fieldbackground=COLORS["bg_card"],
                    rowheight=38,
                    font=("Segoe UI", 10),
                    borderwidth=0)

    style.configure("Treeview.Heading",
                    background=COLORS["bg_elevated"],
                    foreground=COLORS["accent"],
                    font=("Segoe UI", 11, "bold"),
                    borderwidth=0,
                    relief="flat")

    style.map("Treeview",
              background=[("selected", COLORS["accent_glow"])],
              foreground=[("selected", COLORS["accent_dim"])])

    style.configure("Accent.TButton",
                    background=COLORS["accent"],
                    foreground="#ffffff",
                    font=("Segoe UI", 10, "bold"),
                    padding=(16, 10),
                    borderwidth=0,
                    relief="flat")

    style.map("Accent.TButton",
              background=[("active", COLORS["accent_light"]),
                          ("pressed", COLORS["accent_dim"])],
              foreground=[("active", "#ffffff")])

    style.configure("Success.TButton",
                    background=COLORS["success"],
                    foreground="#ffffff",
                    font=("Segoe UI", 10, "bold"),
                    padding=(16, 10),
                    borderwidth=0,
                    relief="flat")

    style.map("Success.TButton",
              background=[("active", COLORS["success_glow"]),
                          ("pressed", "#047a54")],
              foreground=[("active", "#ffffff")])

    style.configure("Danger.TButton",
                    background=COLORS["danger"],
                    foreground="#ffffff",
                    font=("Segoe UI", 10, "bold"),
                    padding=(16, 10),
                    borderwidth=0,
                    relief="flat")

    style.map("Danger.TButton",
              background=[("active", COLORS["danger_glow"]),
                          ("pressed", "#be123c")],
              foreground=[("active", "#ffffff")])

    style.configure("Ghost.TButton",
                    background=COLORS["bg_elevated"],
                    foreground=COLORS["text_secondary"],
                    font=("Segoe UI", 10),
                    padding=(16, 10),
                    borderwidth=0,
                    relief="flat")

    style.map("Ghost.TButton",
              background=[("active", COLORS["bg_hover"])])


# ==============================
# UI HELPERS
# ==============================
def create_gradient(canvas, width, height, color1, color2, direction="vertical"):
    r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
    r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)

    steps = height if direction == "vertical" else width
    for i in range(steps):
        ratio = i / max(steps - 1, 1)
        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)
        color = f"#{r:02x}{g:02x}{b:02x}"
        if direction == "vertical":
            canvas.create_line(0, i, width, i, fill=color, width=1)
        else:
            canvas.create_line(i, 0, i, height, fill=color, width=1)


def create_rounded_frame(parent, bg_color, border_color=None, radius=12, pad=1):
    outer = tk.Frame(parent,
                     bg=border_color if border_color else COLORS["border"],
                     padx=pad, pady=pad)
    inner = tk.Frame(outer, bg=bg_color)
    inner.pack(fill="both", expand=True)
    return outer, inner


def create_glow_label(parent, text, font_spec, fg, bg, glow_color=None):
    container = tk.Frame(parent, bg=bg)

    if glow_color:
        shadow = tk.Label(container, text=text, font=font_spec,
                          bg=bg, fg=glow_color)
        shadow.place(x=1, y=1)

    label = tk.Label(container, text=text, font=font_spec, bg=bg, fg=fg)
    label.pack()

    return container, label


def create_icon_button(parent, text, icon, style, command):
    return ttk.Button(parent,
                      text=f"{icon}  {text}",
                      style=style,
                      command=command)


def create_status_indicator(parent, bg):
    frame = tk.Frame(parent, bg=bg)
    canvas = tk.Canvas(frame, width=12, height=12, bg=bg,
                       highlightthickness=0)
    canvas.pack(side="left", padx=(0, 6))
    dot = canvas.create_oval(2, 2, 10, 10, fill=COLORS["text_muted"], outline="")
    label = tk.Label(frame, text="Idle", font=("Segoe UI", 9),
                     bg=bg, fg=COLORS["text_muted"])
    label.pack(side="left")
    return frame, canvas, dot, label


# ==============================
# CAMERA LOOP  — MULTI-PERSON
# ==============================
def _safe_update_status(color, text, fg_color):
    """Safely update UI from camera thread"""
    if status_canvas and status_dot and status_label:
        try:
            def _update(c=color, t=text, f=fg_color):
                status_canvas.itemconfig(status_dot, fill=c)
                status_label.config(text=t, fg=f)

            status_label.after(0, _update)
        except Exception:
            pass


def camera_loop():
    global cap, running, person_state, _frame_counter

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Failed to open camera 0. Trying camera 1...")
        cap = cv2.VideoCapture(1)

    _safe_update_status(COLORS["success"], f"Camera Active  •  Multi-Person (max {MAX_PERSONS})", COLORS["success"])

    person_state = {}
    _frame_counter = 0

    window_name = "GazeStyle — Multi-Person AI Camera"
    cv2.namedWindow(window_name)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        _frame_counter += 1

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = face_detector.detect(mp_image)

        num_detected = len(result.face_landmarks) if result.face_landmarks else 0

        # ── Flush state for persons that are no longer in frame ──
        active_indices = set(range(num_detected))
        for pid in list(person_state.keys()):
            if pid not in active_indices:
                ps = person_state[pid]
                if ps["current_product"] and ps["gaze_start_time"] is not None:
                    duration = time.perf_counter() - ps["gaze_start_time"]
                    if duration > 0:
                        gaze_data[ps["current_product"]]["total_time"] += duration
                        gaze_data[ps["current_product"]]["fixations"] += 1
                del person_state[pid]

        # ── Overlay: person count badge ──
        badge_text = f"Persons: {num_detected}"
        (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (8, 8), (18 + bw, 18 + bh), (30, 20, 60), -1)
        cv2.putText(frame, badge_text, (13, 13 + bh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 200, 255), 2)

        if result.face_landmarks:
            for pid, face_landmarks in enumerate(result.face_landmarks):

                if pid not in person_state:
                    person_state[pid] = {
                        "current_product": None,
                        "gaze_start_time": None
                    }

                ps = person_state[pid]
                color_bgr = PERSON_COLORS_BGR[pid % len(PERSON_COLORS_BGR)]

                iris = face_landmarks[468]
                cx_iris = int(iris.x * w)
                cy_iris = int(iris.y * h)

                cv2.circle(frame, (cx_iris, cy_iris), 8, color_bgr, -1)
                cv2.circle(frame, (cx_iris, cy_iris), 10, color_bgr, 2)

                pid_label = f"P{pid + 1}"
                cv2.putText(frame, pid_label,
                            (cx_iris - 10, cy_iris - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2)

                xs = [int(lm.x * w) for lm in face_landmarks]
                ys = [int(lm.y * h) for lm in face_landmarks]
                x_min, x_max = max(0, min(xs)), min(w - 1, max(xs))
                y_min, y_max = max(0, min(ys)), min(h - 1, max(ys))

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                              color_bgr, 1)

                detected_product = None
                for name, (bx1, by1, bx2, by2) in product_boxes.items():
                    if bx1 <= cx_iris <= bx2 and by1 <= cy_iris <= by2:
                        detected_product = name
                        break

                if detected_product != ps["current_product"]:
                    if ps["current_product"] is not None and ps["gaze_start_time"] is not None:
                        duration = time.perf_counter() - ps["gaze_start_time"]
                        if duration > 0:
                            gaze_data[ps["current_product"]]["total_time"] += duration
                            gaze_data[ps["current_product"]]["fixations"] += 1

                    ps["gaze_start_time"] = time.perf_counter() if detected_product else None
                    ps["current_product"] = detected_product

                face_crop = frame[y_min:y_max, x_min:x_max]
                if face_crop.size > 0:
                    try:
                        emotion_queue.put_nowait((face_crop, detected_product, pid))
                    except queue.Full:
                        pass

                dominant_emotion = latest_emotions.get(pid)
                if dominant_emotion:
                    cv2.putText(frame, dominant_emotion,
                                (cx_iris + 12, cy_iris + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color_bgr, 2)

                if detected_product:
                    bx1, by1, bx2, by2 = product_boxes[detected_product]
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color_bgr, 3)

        # ── Draw all product boxes ──
        for name, (bx1, by1, bx2, by2) in product_boxes.items():
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (180, 60, 200), 2)
            cv2.putText(frame, name, (bx1, by1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 60, 200), 2)

        cv2.imshow(window_name, frame)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

        # ── Write shared JSON every 30 frames (~1s at 30fps) ──
        if _frame_counter % 30 == 0:
            save_session_json()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_camera()

    # ── Commit any open gaze sessions on exit ──
    for pid, ps in person_state.items():
        if ps["current_product"] and ps["gaze_start_time"] is not None:
            duration = time.perf_counter() - ps["gaze_start_time"]
            if duration > 0:
                gaze_data[ps["current_product"]]["total_time"] += duration
                gaze_data[ps["current_product"]]["fixations"] += 1
    person_state.clear()

    save_session_json()  # final flush for dashboard

    cap.release()
    cv2.destroyAllWindows()

    _safe_update_status(COLORS["text_muted"], "Idle", COLORS["text_muted"])


def start_camera():
    global running
    if not product_boxes:
        messagebox.showwarning("Warning", "Add at least one product region first")
        return
    running = True
    threading.Thread(target=camera_loop, daemon=True).start()


def stop_camera():
    global running
    running = False


# ==============================
# PRODUCT MANAGEMENT
# ==============================
def add_product():
    name = product_entry.get().strip()

    try:
        x1 = int(x1_entry.get())
        y1 = int(y1_entry.get())
        x2 = int(x2_entry.get())
        y2 = int(y2_entry.get())
    except Exception:
        messagebox.showwarning("Warning", "Enter valid bounding box values")
        return

    if not name:
        messagebox.showwarning("Warning", "Enter product name")
        return

    if x2 <= x1 or y2 <= y1:
        messagebox.showwarning("Warning", "Invalid box size")
        return

    product_boxes[name] = (x1, y1, x2, y2)
    gaze_data[name] = {"fixations": 0, "total_time": 0}
    emotion_data[name] = {
        "happy": 0, "sad": 0, "neutral": 0,
        "angry": 0, "surprise": 0,
        "fear": 0, "disgust": 0
    }

    product_list.insert(tk.END, f"  ●  {name}    ({x1},{y1}) → ({x2},{y2})")

    count = product_list.size()
    if product_count_label:
        product_count_label.config(text=f"{count} product{'s' if count != 1 else ''} configured")

    product_entry.delete(0, tk.END)
    x1_entry.delete(0, tk.END)
    y1_entry.delete(0, tk.END)
    x2_entry.delete(0, tk.END)
    y2_entry.delete(0, tk.END)


# ==============================
# DASHBOARD
# ==============================
def open_dashboard():
    dashboard = tk.Toplevel()
    dashboard.title("GazeStyle — Analytics Dashboard")
    dashboard.geometry("1200x720")
    dashboard.configure(bg=COLORS["bg_darkest"])
    dashboard.minsize(900, 520)

    dashboard.update_idletasks()
    x_c = int((dashboard.winfo_screenwidth() / 2) - 600)
    y_c = int((dashboard.winfo_screenheight() / 2) - 360)
    dashboard.geometry(f"1200x720+{x_c}+{y_c}")

    # Title bar
    title_bar = tk.Frame(dashboard, bg=COLORS["bg_primary"], height=80)
    title_bar.pack(fill="x")
    title_bar.pack_propagate(False)

    title_inner = tk.Frame(title_bar, bg=COLORS["bg_primary"])
    title_inner.pack(expand=True)

    tk.Label(title_inner,
             text="\U0001F4CA",
             font=("Segoe UI Emoji", 22),
             bg=COLORS["bg_primary"],
             fg=COLORS["accent"]).pack(side="left", padx=(0, 12))

    tk.Label(title_inner,
             text="Retail Attention Analytics  —  Multi-Person",
             font=("Segoe UI", 22, "bold"),
             bg=COLORS["bg_primary"],
             fg=COLORS["text_primary"]).pack(side="left")

    active_count = len(person_state)
    tk.Label(title_inner,
             text=f"  [{active_count} person{'s' if active_count != 1 else ''} active]",
             font=("Segoe UI", 11),
             bg=COLORS["bg_primary"],
             fg=COLORS["text_muted"]).pack(side="left")

    sep = tk.Canvas(dashboard, height=2, bg=COLORS["bg_darkest"],
                    highlightthickness=0)
    sep.pack(fill="x", padx=40, pady=(0, 5))
    sep.create_line(0, 1, 2000, 1, fill=COLORS["accent_dim"], width=1)

    stats_bar = tk.Frame(dashboard, bg=COLORS["bg_darkest"])
    stats_bar.pack(fill="x", padx=50, pady=(10, 5))

    total_products = len(gaze_data)
    total_fixations = sum(d["fixations"] for d in gaze_data.values())
    total_time = sum(d["total_time"] for d in gaze_data.values())

    for icon, label, value, color in [
        ("\U0001F4E6", "Products", str(total_products), COLORS["accent"]),
        ("\U0001F441", "Total Fixations", str(total_fixations), COLORS["success"]),
        ("\u23F1", "Total Gaze Time", f"{round(total_time, 1)}s", COLORS["warning"]),
        ("\U0001F465", "Max Persons", str(MAX_PERSONS), COLORS["text_secondary"]),
    ]:
        sf = tk.Frame(stats_bar, bg=COLORS["bg_card"], padx=20, pady=10)
        sf.pack(side="left", expand=True, fill="x", padx=6)
        tk.Label(sf, text=f"{icon} {label}",
                 font=("Segoe UI", 9),
                 bg=COLORS["bg_card"],
                 fg=COLORS["text_muted"]).pack(anchor="w")
        tk.Label(sf, text=value,
                 font=("Segoe UI", 18, "bold"),
                 bg=COLORS["bg_card"],
                 fg=color).pack(anchor="w")

    table_frame = tk.Frame(dashboard, bg=COLORS["bg_card"], padx=2, pady=2)
    table_frame.pack(fill="both", expand=True, padx=50, pady=15)

    global table
    table = ttk.Treeview(
        table_frame,
        columns=("Product", "Fixations (All Persons)",
                 "Avg Gaze Time", "Top Emotion"),
        show="headings"
    )

    for col in table["columns"]:
        table.heading(col, text=col)
        table.column(col, anchor="center", width=250)

    table.tag_configure("oddrow", background=COLORS["bg_card"])
    table.tag_configure("evenrow", background=COLORS["bg_card_alt"])
    table.pack(fill="both", expand=True, padx=1, pady=1)

    btn_frame = tk.Frame(dashboard, bg=COLORS["bg_darkest"])
    btn_frame.pack(pady=(5, 25))

    create_icon_button(btn_frame, "Refresh Data", "\U0001F504",
                       "Accent.TButton", update_table).pack(side="left", padx=12)

    create_icon_button(btn_frame, "Export CSV", "\U0001F4BE",
                       "Success.TButton", export_csv).pack(side="left", padx=12)

    update_table()


def update_table():
    for row in table.get_children():
        table.delete(row)

    for idx, (product, data) in enumerate(gaze_data.items()):
        fix = data["fixations"]
        total = data["total_time"]
        avg = total / fix if fix > 0 else 0
        top = max(emotion_data[product], key=emotion_data[product].get)

        tag = "oddrow" if idx % 2 == 0 else "evenrow"
        table.insert("", "end",
                     values=(product, fix,
                             f"{round(avg, 2)}s", top),
                     tags=(tag,))


def export_csv():
    filename = f"gaze_data_multiperson_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Product", "Total Fixations (All Persons)",
                         "Avg Gaze Time (s)", "Top Emotion"])

        for product, data in gaze_data.items():
            fix = data["fixations"]
            total = data["total_time"]
            avg = total / fix if fix > 0 else 0
            top = max(emotion_data[product], key=emotion_data[product].get)

            writer.writerow([product, fix, round(avg, 2), top])

    save_session_json()  # keep dashboard in sync after export
    messagebox.showinfo("Export", f"CSV Exported\n{filename}")


# ==============================
# TREND REPORT — integrated from trend_analyzer.py
# ==============================
def generate_trend_report():
    """Generate a multi-session trend HTML report and open it in the browser."""
    directory = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(directory, "trend_report.html")

    sessions = load_sessions(directory)

    if not sessions:
        messagebox.showwarning(
            "Trend Report",
            "No CSV exports found.\n\n"
            "Export at least one session first using the\n"
            "'Export CSV' button in the Analytics Dashboard."
        )
        return

    try:
        agg = aggregate(sessions)
        _generate_report(sessions, agg, out_path)
        messagebox.showinfo(
            "Trend Report",
            f"Report generated from {len(sessions)} session(s).\n\n"
            f"Opening in browser...\n{out_path}"
        )
        webbrowser.open(f"file:///{out_path}")
    except Exception as e:
        messagebox.showerror("Trend Report Error", str(e))


# ==============================
# MODERN PRODUCT WINDOW
# ==============================
def open_product_window():
    splash.destroy()
    apply_theme()

    global product_entry, x1_entry, y1_entry, x2_entry, y2_entry, product_list
    global status_canvas, status_dot, status_label, product_count_label

    root = tk.Tk()
    root.title("GazeStyle — AI Retail Setup  [Multi-Person Mode]")
    root.geometry("1150x780")
    root.configure(bg=COLORS["bg_darkest"])
    root.minsize(900, 600)

    root.update_idletasks()
    x_c = int((root.winfo_screenwidth() / 2) - 575)
    y_c = int((root.winfo_screenheight() / 2) - 390)
    root.geometry(f"1150x780+{x_c}+{y_c}")

    # ─── TOP HEADER BAR ───
    header_bar = tk.Frame(root, bg=COLORS["bg_primary"], height=90)
    header_bar.pack(fill="x")
    header_bar.pack_propagate(False)

    header_inner = tk.Frame(header_bar, bg=COLORS["bg_primary"])
    header_inner.pack(expand=True)

    tk.Label(header_inner,
             text="\U0001F441\uFE0F",
             font=("Segoe UI Emoji", 26),
             bg=COLORS["bg_primary"],
             fg=COLORS["accent"]).pack(side="left", padx=(0, 14))

    title_text_frame = tk.Frame(header_inner, bg=COLORS["bg_primary"])
    title_text_frame.pack(side="left")

    tk.Label(title_text_frame,
             text="GazeStyle AI Retail System",
             font=("Segoe UI", 22, "bold"),
             bg=COLORS["bg_primary"],
             fg=COLORS["text_primary"]).pack(anchor="w")

    tk.Label(title_text_frame,
             text=f"Multi-Person Gaze Recognition  •  Up to {MAX_PERSONS} persons simultaneously",
             font=("Segoe UI", 10),
             bg=COLORS["bg_primary"],
             fg=COLORS["text_muted"]).pack(anchor="w")

    legend_frame = tk.Frame(header_inner, bg=COLORS["bg_primary"])
    legend_frame.pack(side="left", padx=(30, 0))
    for i in range(min(MAX_PERSONS, 6)):
        r, g, b = PERSON_COLORS_BGR[i][2], PERSON_COLORS_BGR[i][1], PERSON_COLORS_BGR[i][0]
        hex_c = f"#{r:02x}{g:02x}{b:02x}"
        tk.Label(legend_frame, text=f"P{i + 1}",
                 font=("Segoe UI", 8, "bold"),
                 bg=hex_c, fg="white",
                 padx=5, pady=2).pack(side="left", padx=2)

    accent_line = tk.Canvas(root, height=3, bg=COLORS["bg_darkest"],
                            highlightthickness=0)
    accent_line.pack(fill="x")
    accent_line.create_rectangle(0, 0, 2000, 3, fill=COLORS["accent_dim"], outline="")

    # ─── MAIN CONTENT ───
    main_container = tk.Frame(root, bg=COLORS["bg_darkest"])
    main_container.pack(fill="both", expand=True, padx=35, pady=20)

    # ─── LEFT PANEL: Add Product ───
    left_outer = tk.Frame(main_container, bg=COLORS["border"], padx=1, pady=1)
    left_outer.pack(side="left", fill="both", expand=True, padx=(0, 12))
    left_panel = tk.Frame(left_outer, bg=COLORS["bg_card"], padx=28, pady=24)
    left_panel.pack(fill="both", expand=True)

    card_header_left = tk.Frame(left_panel, bg=COLORS["bg_card"])
    card_header_left.pack(fill="x", pady=(0, 16))

    tk.Label(card_header_left,
             text="\u2B1B",
             font=("Segoe UI", 14),
             bg=COLORS["bg_card"],
             fg=COLORS["accent"]).pack(side="left", padx=(0, 8))

    tk.Label(card_header_left,
             text="Add Product Region",
             font=("Segoe UI", 14, "bold"),
             bg=COLORS["bg_card"],
             fg=COLORS["accent_light"]).pack(side="left")

    div1 = tk.Canvas(left_panel, height=1, bg=COLORS["bg_card"], highlightthickness=0)
    div1.pack(fill="x", pady=(0, 16))
    div1.create_line(0, 0, 800, 0, fill=COLORS["border"], width=1)

    tk.Label(left_panel, text="PRODUCT NAME",
             font=("Segoe UI", 8, "bold"),
             bg=COLORS["bg_card"],
             fg=COLORS["text_muted"]).pack(anchor="w", pady=(0, 4))

    name_frame = tk.Frame(left_panel, bg=COLORS["bg_darkest"], padx=10, pady=8)
    name_frame.pack(fill="x", pady=(0, 16))

    product_entry = tk.Entry(name_frame,
                             font=("Segoe UI", 11),
                             bg=COLORS["bg_darkest"],
                             fg=COLORS["text_primary"],
                             insertbackground=COLORS["accent"],
                             relief="flat", bd=0)
    product_entry.pack(fill="x")

    tk.Label(left_panel, text="BOUNDING BOX COORDINATES",
             font=("Segoe UI", 8, "bold"),
             bg=COLORS["bg_card"],
             fg=COLORS["text_muted"]).pack(anchor="w", pady=(0, 8))

    box_frame = tk.Frame(left_panel, bg=COLORS["bg_card"])
    box_frame.pack(fill="x", pady=(0, 16))

    labels = ["x1", "y1", "x2", "y2"]
    label_icons = ["↖", "↖", "↘", "↘"]
    entries = []

    for i, (lbl, ico) in enumerate(zip(labels, label_icons)):
        cell = tk.Frame(box_frame, bg=COLORS["bg_card"])
        cell.grid(row=0, column=i, padx=6, pady=4, sticky="ew")
        box_frame.columnconfigure(i, weight=1)

        tk.Label(cell, text=f"{ico} {lbl.upper()}",
                 font=("Segoe UI", 8, "bold"),
                 bg=COLORS["bg_card"],
                 fg=COLORS["text_muted"]).pack(anchor="w", pady=(0, 3))

        ef = tk.Frame(cell, bg=COLORS["bg_darkest"], padx=8, pady=6)
        ef.pack(fill="x")

        entry = tk.Entry(ef,
                         font=("Segoe UI", 11),
                         bg=COLORS["bg_darkest"],
                         fg=COLORS["text_primary"],
                         insertbackground=COLORS["accent"],
                         relief="flat", bd=0, width=8, justify="center")
        entry.pack(fill="x")
        entries.append(entry)

    x1_entry, y1_entry, x2_entry, y2_entry = entries

    btn_container = tk.Frame(left_panel, bg=COLORS["bg_card"])
    btn_container.pack(fill="x", pady=(8, 0))

    create_icon_button(btn_container, "Draw Region", "\u270F\uFE0F",
                       "Ghost.TButton", select_region_interactive).pack(fill="x", pady=(0, 6))

    create_icon_button(btn_container, "Add Product", "\u2795",
                       "Accent.TButton", add_product).pack(fill="x")

    # ─── RIGHT PANEL: Product List ───
    right_outer = tk.Frame(main_container, bg=COLORS["border"], padx=1, pady=1)
    right_outer.pack(side="right", fill="both", expand=True, padx=(12, 0))
    right_panel = tk.Frame(right_outer, bg=COLORS["bg_card"], padx=28, pady=24)
    right_panel.pack(fill="both", expand=True)

    card_header_right = tk.Frame(right_panel, bg=COLORS["bg_card"])
    card_header_right.pack(fill="x", pady=(0, 16))

    tk.Label(card_header_right,
             text="\U0001F4CB",
             font=("Segoe UI Emoji", 14),
             bg=COLORS["bg_card"],
             fg=COLORS["accent"]).pack(side="left", padx=(0, 8))

    tk.Label(card_header_right,
             text="Configured Products",
             font=("Segoe UI", 14, "bold"),
             bg=COLORS["bg_card"],
             fg=COLORS["accent_light"]).pack(side="left")

    div2 = tk.Canvas(right_panel, height=1, bg=COLORS["bg_card"], highlightthickness=0)
    div2.pack(fill="x", pady=(0, 12))
    div2.create_line(0, 0, 800, 0, fill=COLORS["border"], width=1)

    product_count_label = tk.Label(right_panel,
                                   text="0 products configured",
                                   font=("Segoe UI", 9),
                                   bg=COLORS["bg_card"],
                                   fg=COLORS["text_muted"])
    product_count_label.pack(anchor="w", pady=(0, 8))

    list_container = tk.Frame(right_panel, bg=COLORS["bg_darkest"], padx=2, pady=2)
    list_container.pack(fill="both", expand=True)

    product_list = tk.Listbox(list_container,
                              bg=COLORS["bg_darkest"],
                              fg=COLORS["text_primary"],
                              font=("Segoe UI", 10),
                              selectbackground=COLORS["accent_dim"],
                              selectforeground="white",
                              activestyle="none",
                              relief="flat", bd=0,
                              highlightthickness=0)
    product_list.pack(fill="both", expand=True, side="left")

    scrollbar = ttk.Scrollbar(list_container, orient="vertical",
                              command=product_list.yview)
    scrollbar.pack(fill="y", side="right")
    product_list.config(yscrollcommand=scrollbar.set)

    # ─── BOTTOM CONTROL BAR ───
    bottom_bar = tk.Frame(root, bg=COLORS["bg_primary"], height=70)
    bottom_bar.pack(fill="x", side="bottom")
    bottom_bar.pack_propagate(False)

    bottom_accent = tk.Canvas(bottom_bar, height=1, bg=COLORS["bg_primary"],
                              highlightthickness=0)
    bottom_accent.pack(fill="x")
    bottom_accent.create_line(0, 0, 2000, 0, fill=COLORS["border"], width=1)

    bottom_inner = tk.Frame(bottom_bar, bg=COLORS["bg_primary"])
    bottom_inner.pack(expand=True)

    status_frame, status_canvas, status_dot, status_label = \
        create_status_indicator(bottom_inner, COLORS["bg_primary"])
    status_frame.pack(side="left", padx=(0, 30))

    create_icon_button(bottom_inner, "Start Camera", "▶",
                       "Success.TButton", start_camera).pack(side="left", padx=8)

    create_icon_button(bottom_inner, "Stop Camera", "■",
                       "Danger.TButton", stop_camera).pack(side="left", padx=8)

    create_icon_button(bottom_inner, "Open Dashboard", "\U0001F4CA",
                       "Accent.TButton", open_dashboard).pack(side="left", padx=8)

    create_icon_button(bottom_inner, "Trend Report", "\U0001F4C8",
                       "Ghost.TButton", generate_trend_report).pack(side="left", padx=8)

    root.mainloop()


# ==============================
# SPLASH SCREEN — Neon Surveillance dark style
# ==============================
splash = tk.Tk()
splash.title("GazeStyle")
splash.configure(bg=SPLASH["bg"])
splash.attributes('-fullscreen', True)
splash.update_idletasks()

sw = splash.winfo_screenwidth()
sh = splash.winfo_screenheight()
cx = sw // 2
cy = sh // 2

sc = tk.Canvas(splash, width=sw, height=sh,
               bg=SPLASH["bg"], highlightthickness=0)
sc.pack(fill="both", expand=True)

for gx in range(0, sw, 55):
    sc.create_line(gx, 0, gx, sh, fill=SPLASH["grid"], width=1)
for gy in range(0, sh, 55):
    sc.create_line(0, gy, sw, gy, fill=SPLASH["grid"], width=1)

blen = 90
for (bx, by, dx, dy) in [
    (55, 55, blen, 0), (55, 55, 0, blen),
    (sw - 55, 55, -blen, 0), (sw - 55, 55, 0, blen),
    (55, sh - 55, blen, 0), (55, sh - 55, 0, -blen),
    (sw - 55, sh - 55, -blen, 0), (sw - 55, sh - 55, 0, -blen),
]:
    sc.create_line(bx, by, bx + dx, by + dy,
                   fill=SPLASH["bracket"], width=1)

ring_r = 105
ring_cy = cy - 100
sc.create_oval(cx - ring_r, ring_cy - ring_r,
               cx + ring_r, ring_cy + ring_r,
               outline=SPLASH["ring_outer"], width=1)
sc.create_oval(cx - ring_r + 16, ring_cy - ring_r + 16,
               cx + ring_r - 16, ring_cy + ring_r - 16,
               outline=SPLASH["ring_inner"], width=1)
for angle_deg in [0, 90, 180, 270]:
    rad = math.radians(angle_deg)
    x1s = cx + int((ring_r - 10) * math.cos(rad))
    y1s = ring_cy + int((ring_r - 10) * math.sin(rad))
    x2s = cx + int((ring_r + 10) * math.cos(rad))
    y2s = ring_cy + int((ring_r + 10) * math.sin(rad))
    sc.create_line(x1s, y1s, x2s, y2s, fill=SPLASH["ring_inner"], width=1)

sc.create_text(cx, ring_cy,
               text="\U0001F441",
               font=("Segoe UI Emoji", 28),
               fill=SPLASH["icon"])

sc.create_text(cx, cy + 12,
               text="GazeStyle",
               font=("Courier New", 52, "bold"),
               fill=SPLASH["title"])

sc.create_line(cx - 210, cy + 42, cx + 210, cy + 42,
               fill=SPLASH["underline"], width=2)

sc.create_text(cx, cy + 68,
               text="AI-Powered Retail Gaze & Emotion Analytics",
               font=("Consolas", 13),
               fill=SPLASH["subtitle"])

sc.create_text(cx, cy + 98,
               text=f"v3.0  •  Multi-Person Edition  •  Up to {MAX_PERSONS} simultaneous persons",
               font=("Consolas", 9),
               fill=SPLASH["muted"])

scan_sx = cx - 200
scan_ex = cx + 200
scan_y = cy + 124
sc.create_line(scan_sx, scan_y, scan_ex, scan_y,
               fill=SPLASH["grid"], width=1)
glow_line = sc.create_line(scan_sx, scan_y, scan_sx + 80, scan_y,
                           fill=SPLASH["scan"], width=2)

glow_anim_running = True
glow_line_half = 200


def animate_glow(pos=0):
    if not glow_anim_running:
        return
    total = scan_ex - scan_sx
    x_start = scan_sx + (pos % total)
    x_end = min(x_start + 80, scan_ex)
    sc.coords(glow_line, x_start, scan_y, x_end, scan_y)
    splash.after(22, animate_glow, pos + 4)


animate_glow()
apply_theme()

launch_btn = ttk.Button(splash,
                        text="\u26A1  Launch Application",
                        style="Accent.TButton",
                        command=lambda: [set_glow_stop(), open_product_window()])

sc.create_window(cx, cy + 160, window=launch_btn)

sc.create_text(cx, cy + 205,
               text="Press ESC to exit  •  Click Launch to begin",
               font=("Consolas", 9),
               fill=SPLASH["muted"])

sc.create_text(cx, sh - 28,
               text="© 2026 GazeStyle  •  Powered by MediaPipe & DeepFace  •  Multi-Person Mode",
               font=("Consolas", 9),
               fill=SPLASH["muted"])

splash.bind("<Escape>", lambda e: splash.destroy())


def set_glow_stop():
    global glow_anim_running
    glow_anim_running = False


# Initialize globals for status indicator
status_canvas = None
status_dot = None
status_label = None
product_count_label = None

splash.mainloop()
