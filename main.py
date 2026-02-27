import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import pickle
import os
import time
import numpy as np
from collections import deque, Counter

WINDOW_NAME    = "ISL Interpreter"
FRAME_WIDTH    = 1280
FRAME_HEIGHT   = 720
PANEL_WIDTH    = 300
CAM_INDEX      = 0
MP_MODEL_PATH  = "hand_landmarker.task"
ISL_MODEL_PATH = "isl_model.pkl"

SMOOTH_FRAMES  = 5      # REDUCED for faster response
HOLD_TIME      = 1.0    # REDUCED for faster letter addition
MIN_CONFIDENCE = 0.4    # LOWERED to accept more predictions

COLOR_BG         = (30, 30, 30)
COLOR_ACCENT     = (255, 180, 0)
COLOR_WHITE      = (255, 255, 255)
COLOR_GREEN      = (80, 200, 80)
COLOR_GRAY       = (160, 160, 160)
COLOR_BOX        = (50, 50, 50)
COLOR_CONNECTION = (255, 255, 255)
COLOR_RED        = (60, 60, 220)

HAND_COLORS = [(0, 255, 255), (255, 0, 255)]  # Cyan=left, Magenta=right

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def load_isl_model(model_path):
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        return None
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ ISL model loaded")
    print(f"   Features expected: {model.n_features_in_}")
    print(f"   Classes: {len(model.classes_)}")
    return model

def download_mp_model():
    if os.path.exists(MP_MODEL_PATH):
        return
    print("📥 Downloading MediaPipe model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, MP_MODEL_PATH)
    print("✅ Downloaded!")

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

def draw_landmarks(frame, landmarks_px, hand_index=0):
    dot_color = HAND_COLORS[hand_index % 2]
    for (a, b) in HAND_CONNECTIONS:
        if a < len(landmarks_px) and b < len(landmarks_px):
            cv2.line(frame, landmarks_px[a], landmarks_px[b], COLOR_CONNECTION, 2)
    for (px, py) in landmarks_px:
        cv2.circle(frame, (px, py), 5, dot_color, -1)
        cv2.circle(frame, (px, py), 5, COLOR_WHITE, 1)

def extract_landmarks(hand_landmarks, frame_w, frame_h):
    """Extract 63 features from one hand."""
    coords = []
    points = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])
        px = int(lm.x * frame_w)
        py = int(lm.y * frame_h)
        points.append((px, py))
    return coords, points

def predict_sign_two_hands(model, left_coords, right_coords):
    """
    FIX: Predict using BOTH hands (126 features).
    """
    # Fill missing hands with zeros
    if left_coords is None or len(left_coords) < 63:
        left_coords = [0.0] * 63
    if right_coords is None or len(right_coords) < 63:
        right_coords = [0.0] * 63
    
    # Combine: 63 left + 63 right = 126 features
    combined = np.array(left_coords[:63] + right_coords[:63]).reshape(1, -1)
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    
    try:
        prediction = str(model.predict(combined)[0])
        proba      = model.predict_proba(combined)[0]
        confidence = float(np.max(proba))
        return prediction, confidence
    except Exception as e:
        print(f"⚠️  Error: {e}")
        return "---", 0.0

def draw_panel(frame, prediction="---", confidence=0.0,
               word_buffer="", hands_detected=0, hold_progress=0.0):
    
    h, w    = frame.shape[:2]
    panel_x = w - PANEL_WIDTH

    cv2.rectangle(frame, (panel_x, 0), (w, h), COLOR_BG, -1)
    cv2.line(frame, (panel_x, 0), (panel_x, h), COLOR_ACCENT, 2)

    cv2.putText(frame, "ISL (2 Hands)", (panel_x + 10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_ACCENT, 2)
    cv2.line(frame, (panel_x + 10, 45), (w - 10, 45), COLOR_GRAY, 1)

    # Hand status
    status_color = COLOR_GREEN if hands_detected == 2 else (0, 165, 255) if hands_detected == 1 else COLOR_RED
    status_text  = f"Hands: {hands_detected}/2"
    cv2.putText(frame, status_text, (panel_x + 10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

    # Prediction
    cv2.putText(frame, "Sign:", (panel_x + 10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)
    cv2.putText(frame, str(prediction), (panel_x + 10, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, COLOR_GREEN, 3)

    # Confidence
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

    # Hold progress
    cv2.putText(frame, "Hold to add:", (panel_x + 10, 228),
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
    cv2.putText(frame, "Word:", (panel_x + 10, 282),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)
    cv2.rectangle(frame, (panel_x + 10, 290), (w - 10, 340), COLOR_BOX, -1)
    display_word = word_buffer[-12:] if word_buffer else "_"
    cv2.putText(frame, display_word, (panel_x + 18, 325),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)

    # Controls
    cv2.line(frame, (panel_x + 10, 355), (w - 10, 355), COLOR_GRAY, 1)
    cv2.putText(frame, "Controls:", (panel_x + 10, 375),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ACCENT, 1)
    controls = [
        ("BKSP",  "Delete"),
        ("ENTER", "Clear"),
        ("F",     "Fullscreen"),
        ("Q",     "Quit"),
    ]
    for i, (key, action) in enumerate(controls):
        y = 398 + i * 28
        cv2.putText(frame, f"[{key}] {action}", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_GRAY, 1)

    # Status
    cv2.circle(frame, (panel_x + 20, h - 20), 7,
               COLOR_GREEN if hands_detected == 2 else COLOR_RED, -1)
    cv2.putText(frame, "LIVE", (panel_x + 33, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GREEN, 1)

    return frame

def main():
    download_mp_model()

    isl_model = load_isl_model(ISL_MODEL_PATH)
    if isl_model is None:
        return

    # FIX: num_hands=2 to detect BOTH hands
    base_options = python.BaseOptions(model_asset_path=MP_MODEL_PATH)
    options      = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,  # ← FIX: Was 1, now 2!
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("❌ Webcam failed")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\n✅ ISL Interpreter running!")
    print("   → Show BOTH hands for best results")
    print("   → Press Q to quit\n")

    word_buffer     = ""
    is_fullscreen   = True
    smoother        = PredictionSmoother(SMOOTH_FRAMES)
    hold_start      = None
    last_prediction = "---"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        h, w      = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results  = detector.detect(mp_image)

        left_coords  = None
        right_coords = None
        hands_detected = 0

        # FIX: Extract BOTH hands
        if results.hand_landmarks and results.handedness:
            hands_detected = len(results.hand_landmarks)

            for hand_lms, handedness in zip(results.hand_landmarks, results.handedness):
                hand_label = handedness[0].category_name
                coords, landmarks_px = extract_landmarks(hand_lms, w, h)

                if hand_label == "Left":
                    left_coords = coords
                    draw_landmarks(frame, landmarks_px, 0)  # Cyan
                elif hand_label == "Right":
                    right_coords = coords
                    draw_landmarks(frame, landmarks_px, 1)  # Magenta

        else:
            smoother.clear()
            hold_start      = None
            last_prediction = "---"

        # FIX: Predict with BOTH hands (126 features)
        if left_coords or right_coords:
            raw_label, raw_conf = predict_sign_two_hands(isl_model, left_coords, right_coords)
            smoother.update(raw_label)

        smooth_prediction, smooth_confidence = smoother.get_smooth()

        # Hold timer
        hold_progress = 0.0

        if (left_coords or right_coords) and smooth_prediction != "---":
            if smooth_prediction == last_prediction:
                if hold_start is None:
                    hold_start = time.time()
                elapsed       = time.time() - hold_start
                hold_progress = elapsed / HOLD_TIME

                if elapsed >= HOLD_TIME and smooth_confidence >= MIN_CONFIDENCE:
                    word_buffer    += smooth_prediction
                    print(f"✍️  Added '{smooth_prediction}' → {word_buffer}")
                    hold_start = None
                    smoother.clear()
            else:
                last_prediction = smooth_prediction
                hold_start      = None
        else:
            hold_start      = None
            last_prediction = smooth_prediction

        # Draw UI
        frame = draw_panel(
            frame,
            prediction    = smooth_prediction,
            confidence    = smooth_confidence,
            word_buffer   = word_buffer,
            hands_detected = hands_detected,
            hold_progress = hold_progress,
        )

        cv2.imshow(WINDOW_NAME, frame)

        # Keys
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q')):
            break

        elif key in (ord('f'), ord('F')):
            is_fullscreen = not is_fullscreen
            prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, prop)

        elif key == 8:     # Backspace
            word_buffer = word_buffer[:-1]

        elif key == 13:    # Enter
            word_buffer = ""

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()