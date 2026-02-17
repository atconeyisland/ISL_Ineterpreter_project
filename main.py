

import cv2
import mediapipe as mp
import pickle
import os
import time
import numpy as np
from collections import deque, Counter

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
WINDOW_NAME    = "ISL Interpreter"
FRAME_WIDTH    = 1280
FRAME_HEIGHT   = 720
PANEL_WIDTH    = 300
CAM_INDEX      = 0
ISL_MODEL_PATH = "isl_model.pkl"

SMOOTH_FRAMES     = 10    # Majority vote over last N frames
HOLD_TIME         = 1.5   # Seconds to hold sign before adding to buffer
MIN_CONFIDENCE    = 0.6   # Minimum confidence to accept prediction (FIX #3)

# UI Colors (BGR)
COLOR_BG         = (30, 30, 30)
COLOR_ACCENT     = (255, 180, 0)
COLOR_WHITE      = (255, 255, 255)
COLOR_GREEN      = (80, 200, 80)
COLOR_GRAY       = (160, 160, 160)
COLOR_BOX        = (50, 50, 50)
COLOR_RED        = (60, 60, 220)

HAND_COLORS = [
    (0, 255, 255),   # Hand 1 — cyan
    (255, 0, 255),   # Hand 2 — magenta
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ─────────────────────────────────────────────
#  LOAD ISL MODEL
# ─────────────────────────────────────────────
def load_isl_model(model_path):
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print(f"   Make sure isl_model.pkl is in the same folder as main.py")
        return None
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ ISL model loaded: {model_path}")
    print(f"   Classes ({len(model.classes_)}): {sorted(model.classes_)}")
    return model


# ─────────────────────────────────────────────
#  PREDICTION SMOOTHER
# ─────────────────────────────────────────────
class PredictionSmoother:
    def __init__(self, window_size=SMOOTH_FRAMES):
        self.history = deque(maxlen=window_size)

    def update(self, prediction):
        self.history.append(prediction)

    def get_smooth(self):
        if not self.history:
            return "---", 0.0
        counts = Counter(self.history)
        label, count = counts.most_common(1)[0]
        confidence = count / len(self.history)
        return label, confidence

    def clear(self):
        self.history.clear()


# ─────────────────────────────────────────────
#  DRAW LANDMARKS
# ─────────────────────────────────────────────
def draw_landmarks(frame, landmarks_px, hand_index=0):
    dot_color = HAND_COLORS[hand_index % len(HAND_COLORS)]
    for (a, b) in HAND_CONNECTIONS:
        if a < len(landmarks_px) and b < len(landmarks_px):
            cv2.line(frame, landmarks_px[a], landmarks_px[b], COLOR_WHITE, 2)
    for (px, py) in landmarks_px:
        cv2.circle(frame, (px, py), 5, dot_color, -1)
        cv2.circle(frame, (px, py), 5, COLOR_WHITE, 1)


def draw_landmark_indices(frame, landmarks_px, hand_index=0):
    dot_color = HAND_COLORS[hand_index % len(HAND_COLORS)]
    for i, (px, py) in enumerate(landmarks_px):
        cv2.putText(frame, str(i), (px + 6, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, dot_color, 1)


# ─────────────────────────────────────────────
#  EXTRACT LANDMARKS  ← FIX #1: Uses Solutions API format
# ─────────────────────────────────────────────
def extract_landmarks(hand_landmarks, frame_w, frame_h):
    """
    Extract landmarks from mp.solutions.hands result.
    Returns:
        coords     : flat list of 63 floats [x0,y0,z0, x1,y1,z1, ...]
        landmarks_px: list of (px, py) tuples for drawing
    """
    coords = []
    points = []
    for lm in hand_landmarks.landmark:        # ← .landmark (Solutions API)
        coords.extend([lm.x, lm.y, lm.z])
        px = int(lm.x * frame_w)
        py = int(lm.y * frame_h)
        points.append((px, py))
    return coords, points


# ─────────────────────────────────────────────
#  PREDICT SIGN
# ─────────────────────────────────────────────
def predict_sign(model, landmark_coords):
    """
    Predict ISL sign from 63 landmark features.
    Returns (label: str, confidence: float)
    """
    if not landmark_coords or len(landmark_coords) < 63:
        return "---", 0.0

    try:
        coords = np.array(landmark_coords[:63]).reshape(1, -1)
        coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)

        # FIX #4: Cast to plain str (numpy.str_ → str)
        prediction = str(model.predict(coords)[0])

        proba      = model.predict_proba(coords)[0]
        confidence = float(np.max(proba))

        return prediction, confidence

    except Exception as e:
        print(f"⚠️  Prediction error: {e}")
        return "---", 0.0


# ─────────────────────────────────────────────
#  DRAW PANEL
# ─────────────────────────────────────────────
def draw_panel(frame, prediction="---", confidence=0.0,
               word_buffer="", hand_detected=False,
               num_hands=0, hold_progress=0.0):

    h, w    = frame.shape[:2]
    panel_x = w - PANEL_WIDTH

    cv2.rectangle(frame, (panel_x, 0), (w, h), COLOR_BG, -1)
    cv2.line(frame, (panel_x, 0), (panel_x, h), COLOR_ACCENT, 2)

    # Title
    cv2.putText(frame, "ISL Interpreter", (panel_x + 10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_ACCENT, 2)
    cv2.line(frame, (panel_x + 10, 45), (w - 10, 45), COLOR_GRAY, 1)

    # Hand status
    status_color = COLOR_GREEN if hand_detected else COLOR_RED
    status_text  = f"Hands: {num_hands} DETECTED" if hand_detected else "Hands: NOT FOUND"
    cv2.putText(frame, status_text, (panel_x + 10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

    # Predicted sign
    cv2.putText(frame, "Sign Detected:", (panel_x + 10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)
    cv2.putText(frame, str(prediction), (panel_x + 10, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, COLOR_GREEN, 3)

    # Confidence bar
    cv2.putText(frame, "Confidence:", (panel_x + 10, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)
    bar_x, bar_y = panel_x + 10, 195
    bar_w, bar_h = PANEL_WIDTH - 20, 18
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), COLOR_BOX, -1)
    fill      = int(bar_w * confidence)
    bar_color = COLOR_GREEN if confidence >= MIN_CONFIDENCE else (0, 165, 255)
    if fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill, bar_y + bar_h), bar_color, -1)
    cv2.putText(frame, f"{int(confidence * 100)}%",
                (bar_x + bar_w + 4, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

    # Hold progress bar
    cv2.putText(frame, "Hold to add letter:", (panel_x + 10, 228),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1)
    hold_bar_y = 235
    cv2.rectangle(frame, (bar_x, hold_bar_y),
                  (bar_x + bar_w, hold_bar_y + 12), COLOR_BOX, -1)
    hold_fill = int(bar_w * min(hold_progress, 1.0))
    if hold_fill > 0:
        cv2.rectangle(frame, (bar_x, hold_bar_y),
                      (bar_x + hold_fill, hold_bar_y + 12), COLOR_ACCENT, -1)

    # Word buffer
    cv2.line(frame, (panel_x + 10, 260), (w - 10, 260), COLOR_GRAY, 1)
    cv2.putText(frame, "Word Buffer:", (panel_x + 10, 282),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)
    cv2.rectangle(frame, (panel_x + 10, 290), (w - 10, 340), COLOR_BOX, -1)
    display_word = word_buffer if word_buffer else "_"
    cv2.putText(frame, display_word, (panel_x + 18, 325),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)

    # Controls
    cv2.line(frame, (panel_x + 10, 355), (w - 10, 355), COLOR_GRAY, 1)
    cv2.putText(frame, "Controls:", (panel_x + 10, 375),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ACCENT, 1)
    controls = [
        ("SPACE", "Speak word (TTS)"),
        ("BKSP",  "Delete last letter"),
        ("ENTER", "Clear buffer"),
        ("F",     "Toggle fullscreen"),
        ("I",     "Toggle indices"),
        ("Q",     "Quit"),
    ]
    for i, (key, action) in enumerate(controls):
        y = 398 + i * 28
        cv2.putText(frame, f"[{key}]", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ACCENT, 1)
        cv2.putText(frame, action, (panel_x + 75, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_GRAY, 1)

    # Status dot
    cv2.circle(frame, (panel_x + 20, h - 20), 7,
               COLOR_GREEN if hand_detected else COLOR_RED, -1)
    cv2.putText(frame, "LIVE", (panel_x + 33, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GREEN, 1)

    return frame


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():

    # ── Load ISL model ────────────────────────
    isl_model = load_isl_model(ISL_MODEL_PATH)
    if isl_model is None:
        return

    # ── FIX #1: Setup MediaPipe Solutions API ─
    # OLD: mediapipe.tasks.python (Tasks API)  ← causes random predictions
    # NEW: mp.solutions.hands (Solutions API)  ← matches training data ✅
    mp_hands    = mp.solutions.hands
    mp_drawing  = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,                   # FIX #2: only 1 hand → no concat bug
        min_detection_confidence=0.5,
        min_tracking_confidence=0.3,
    )

    # ── Setup webcam ─────────────────────────
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("❌ Could not open webcam. Try changing CAM_INDEX to 1.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    print("\n✅ ISL Interpreter running!")
    print("   → Show a hand sign to detect it")
    print("   → Hold sign steady to add to word buffer")
    print("   → Press SPACE to speak the word")
    print("   → Press Q to quit\n")

    # ── State ────────────────────────────────
    word_buffer     = ""
    show_indices    = False
    is_fullscreen   = True
    smoother        = PredictionSmoother(SMOOTH_FRAMES)
    hold_start      = None
    last_prediction = "---"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        frame     = cv2.flip(frame, 1)
        h, w      = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── FIX #1: Use Solutions API ─────────
        results = hands.process(rgb_frame)

        hand_detected   = False
        landmark_coords = []
        num_hands       = 0

        if results.multi_hand_landmarks:
            hand_detected = True
            num_hands     = len(results.multi_hand_landmarks)

            # FIX #2: Only use FIRST hand → no landmark concat bug
            hand_lms = results.multi_hand_landmarks[0]
            landmark_coords, landmarks_px = extract_landmarks(hand_lms, w, h)

            # Draw landmarks
            draw_landmarks(frame, landmarks_px, 0)
            if show_indices:
                draw_landmark_indices(frame, landmarks_px, 0)

        else:
            # No hand → reset everything
            smoother.clear()
            hold_start      = None
            last_prediction = "---"
            panel_x = w - PANEL_WIDTH
            cv2.putText(frame, "Show your hand...",
                        (panel_x // 2 - 120, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GRAY, 2)

        # ── Predict ───────────────────────────
        if landmark_coords:
            raw_label, raw_conf = predict_sign(isl_model, landmark_coords)
            smoother.update(raw_label)

        smooth_prediction, smooth_confidence = smoother.get_smooth()

        # ── Hold timer ────────────────────────
        hold_progress = 0.0

        if hand_detected and smooth_prediction != "---":
            if smooth_prediction == last_prediction:
                if hold_start is None:
                    hold_start = time.time()
                elapsed       = time.time() - hold_start
                hold_progress = elapsed / HOLD_TIME

                # FIX #3: Only add letter if confidence is high enough
                if elapsed >= HOLD_TIME and smooth_confidence >= MIN_CONFIDENCE:
                    word_buffer    += smooth_prediction
                    print(f"✍️  Added '{smooth_prediction}' → buffer: '{word_buffer}'")
                    hold_start      = None   # reset for next letter
                    smoother.clear()         # clear history to avoid double-adding
            else:
                last_prediction = smooth_prediction
                hold_start      = None
        else:
            hold_start      = None
            last_prediction = smooth_prediction

        # ── Draw UI ───────────────────────────
        frame = draw_panel(
            frame,
            prediction    = smooth_prediction,
            confidence    = smooth_confidence,
            word_buffer   = word_buffer,
            hand_detected = hand_detected,
            num_hands     = num_hands,
            hold_progress = hold_progress,
        )

        cv2.imshow(WINDOW_NAME, frame)

        # ── Key handling ─────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q')):
            print("👋 Quitting.")
            break

        elif key in (ord('f'), ord('F')):
            is_fullscreen = not is_fullscreen
            prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, prop)
            print(f"⛶  {'Fullscreen' if is_fullscreen else 'Windowed'}")

        elif key in (ord('i'), ord('I')):
            show_indices = not show_indices
            print(f"🔢 Indices: {'ON' if show_indices else 'OFF'}")

        elif key == 8:    # Backspace
            word_buffer = word_buffer[:-1]
            print(f"⌫  Buffer: '{word_buffer}'")

        elif key == 13:   # Enter → clear buffer
            word_buffer = ""
            print("🗑️  Buffer cleared.")

        elif key == 32:   # Spacebar → TTS placeholder
            if word_buffer:
                print(f"🔊 [TTS placeholder] Would speak: '{word_buffer}'")
            else:
                print("⚠️  Buffer is empty!")

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()