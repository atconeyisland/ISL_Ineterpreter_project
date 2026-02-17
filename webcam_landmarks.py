import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# ─────────────────────────────────────────────
#  ISL Interpreter — Person C
#  Day 3: MediaPipe Hand Landmarks (v0.10.32)
#  Run: python webcam_landmarks.py
# ─────────────────────────────────────────────

# ── CONFIG ───────────────────────────────────
WINDOW_NAME  = "ISL Interpreter"
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
PANEL_WIDTH  = 300
CAM_INDEX    = 0
MODEL_PATH   = "hand_landmarker.task"  # downloaded automatically

# UI Colors (BGR)
COLOR_BG         = (30, 30, 30)
COLOR_ACCENT     = (255, 180, 0)
COLOR_WHITE      = (255, 255, 255)
COLOR_GREEN      = (80, 200, 80)
COLOR_GRAY       = (160, 160, 160)
COLOR_BOX        = (50, 50, 50)
COLOR_LANDMARK   = (0, 255, 255)
COLOR_CONNECTION = (255, 255, 255)
COLOR_RED        = (60, 60, 220)

# Two hand colors — cyan for hand 1, magenta for hand 2
HAND_COLORS = [
    (0, 255, 255),    # Hand 1 — cyan
    (255, 0, 255),    # Hand 2 — magenta
]

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# ── DOWNLOAD MODEL IF NEEDED ─────────────────
def download_model():
    if os.path.exists(MODEL_PATH):
        print("✅ Model already exists, skipping download.")
        return
    print("📥 Downloading hand landmark model (~30MB, one time only)...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("✅ Model downloaded!")


# ── DRAW LANDMARKS ───────────────────────────
def draw_landmarks(frame, landmarks_px, hand_index=0):
    """
    Draws landmarks for one hand.
    hand_index=0 → cyan (first hand)
    hand_index=1 → magenta (second hand)
    """
    dot_color = HAND_COLORS[hand_index % len(HAND_COLORS)]
    for (a, b) in HAND_CONNECTIONS:
        if a < len(landmarks_px) and b < len(landmarks_px):
            cv2.line(frame, landmarks_px[a], landmarks_px[b], COLOR_CONNECTION, 2)
    for (px, py) in landmarks_px:
        cv2.circle(frame, (px, py), 5, dot_color, -1)
        cv2.circle(frame, (px, py), 5, COLOR_WHITE, 1)


def draw_landmark_indices(frame, landmarks_px, hand_index=0):
    dot_color = HAND_COLORS[hand_index % len(HAND_COLORS)]
    for i, (px, py) in enumerate(landmarks_px):
        cv2.putText(frame, str(i), (px + 6, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, dot_color, 1)


# ── EXTRACT LANDMARKS ────────────────────────
def extract_landmarks(hand_landmarks, frame_w, frame_h):
    coords = []
    points = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])
        px = int(lm.x * frame_w)
        py = int(lm.y * frame_h)
        points.append((px, py))
    return coords, points


# ── DRAW PANEL ───────────────────────────────
def draw_panel(frame, prediction="---", confidence=0.0,
               word_buffer="", hand_detected=False, num_hands=0):
    h, w = frame.shape[:2]
    panel_x = w - PANEL_WIDTH

    cv2.rectangle(frame, (panel_x, 0), (w, h), COLOR_BG, -1)
    cv2.line(frame, (panel_x, 0), (panel_x, h), COLOR_ACCENT, 2)

    cv2.putText(frame, "ISL Interpreter", (panel_x + 10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_ACCENT, 2)
    cv2.line(frame, (panel_x + 10, 45), (w - 10, 45), COLOR_GRAY, 1)

    status_color = COLOR_GREEN if hand_detected else COLOR_RED
    status_text  = f"Hands: {num_hands} DETECTED" if hand_detected else "Hands: NOT FOUND"
    cv2.putText(frame, status_text, (panel_x + 10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

    cv2.putText(frame, "Sign Detected:", (panel_x + 10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)
    cv2.putText(frame, str(prediction), (panel_x + 10, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, COLOR_GREEN, 3)

    cv2.putText(frame, "Confidence:", (panel_x + 10, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)
    bar_x, bar_y, bar_w, bar_h = panel_x + 10, 195, PANEL_WIDTH - 20, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_BOX, -1)
    fill = int(bar_w * confidence)
    bar_color = COLOR_GREEN if confidence > 0.75 else (0, 165, 255)
    if fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill, bar_y + bar_h), bar_color, -1)
    cv2.putText(frame, f"{int(confidence * 100)}%",
                (bar_x + bar_w + 4, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

    cv2.line(frame, (panel_x + 10, 230), (w - 10, 230), COLOR_GRAY, 1)
    cv2.putText(frame, "Word Buffer:", (panel_x + 10, 255),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)
    cv2.rectangle(frame, (panel_x + 10, 265), (w - 10, 315), COLOR_BOX, -1)
    display_word = word_buffer if word_buffer else "_"
    cv2.putText(frame, display_word, (panel_x + 18, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)

    cv2.line(frame, (panel_x + 10, 330), (w - 10, 330), COLOR_GRAY, 1)
    cv2.putText(frame, "Controls:", (panel_x + 10, 355),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ACCENT, 1)
    controls = [
        ("SPACE", "Speak word (TTS)"),
        ("BKSP",  "Delete last letter"),
        ("ENTER", "Clear buffer"),
        ("I",     "Toggle indices"),
        ("Q",     "Quit"),
    ]
    for i, (key, action) in enumerate(controls):
        y = 378 + i * 28
        cv2.putText(frame, f"[{key}]", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ACCENT, 1)
        cv2.putText(frame, action, (panel_x + 75, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_GRAY, 1)

    h_frame = frame.shape[0]
    cv2.circle(frame, (panel_x + 20, h_frame - 20), 7,
               COLOR_GREEN if hand_detected else COLOR_RED, -1)
    cv2.putText(frame, "LIVE", (panel_x + 33, h_frame - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GREEN, 1)

    return frame


# ── MAIN ─────────────────────────────────────
def main():
    # Step 1 — download model file if not present
    download_model()

    # Step 2 — setup MediaPipe new-style detector
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,                        # ← supports two-handed ISL signs
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Step 3 — setup webcam
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("❌ Could not open webcam. Try changing CAM_INDEX to 1.")
        return

    # Open window fullscreen automatically
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("✅ Webcam started!")
    print("   → Show your hand to see landmarks")
    print("   → Press I to toggle index numbers")
    print("   → Press Q to quit")

    word_buffer     = ""
    show_indices    = False
    hand_detected   = False
    landmark_coords = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # ── MediaPipe detection ───────────────
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results   = detector.detect(mp_image)

        hand_detected   = False
        landmark_coords = []

        if results.hand_landmarks:
            hand_detected = True

            # Loop through each detected hand (up to 2)
            for hand_index, hand_lms in enumerate(results.hand_landmarks):
                coords, landmarks_px = extract_landmarks(hand_lms, w, h)

                # Combine coords from both hands into one flat list
                landmark_coords.extend(coords)

                # Draw with different color per hand
                draw_landmarks(frame, landmarks_px, hand_index)

                if show_indices:
                    draw_landmark_indices(frame, landmarks_px, hand_index)

            # Show hand count on screen
            panel_x = w - PANEL_WIDTH
            hands_text = f"Hands: {len(results.hand_landmarks)}/2"
            cv2.putText(frame, hands_text, (panel_x // 2 - 50, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ACCENT, 2)

        else:
            panel_x = w - PANEL_WIDTH
            cv2.putText(frame, "Show your hand...",
                        (panel_x // 2 - 120, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GRAY, 2)

        # ── TODO (Day 4): plug model here ────
        # if landmark_coords:
        #     prediction = model.predict([landmark_coords])[0]
        #     confidence = model.predict_proba([landmark_coords]).max()
        prediction = "---"
        confidence = 0.0

        num_hands_detected = len(results.hand_landmarks) if hand_detected else 0

        frame = draw_panel(frame, prediction, confidence,
                           word_buffer, hand_detected, num_hands_detected)
        cv2.imshow(WINDOW_NAME, frame)

        # ── Key handling ─────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            print("👋 Quitting.")
            break
        elif key == ord('i') or key == ord('I'):
            show_indices = not show_indices
            print(f"🔢 Indices: {'ON' if show_indices else 'OFF'}")
        elif key == 8:
            word_buffer = word_buffer[:-1]
        elif key == 13:
            word_buffer = ""
            print("🗑️ Buffer cleared.")
        elif key == 32:
            print(f"🔊 [TTS placeholder] Would speak: '{word_buffer}'")

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()