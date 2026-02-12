import cv2

# ─────────────────────────────────────────────
#  ISL Interpreter — Person C
#  Day 1: Basic Webcam Skeleton
#  Run: python webcam_skeleton.py
# ─────────────────────────────────────────────

# ── CONFIG ───────────────────────────────────
WINDOW_NAME   = "ISL Interpreter"
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
PANEL_WIDTH   = 300          # right-side info panel width
CAM_INDEX     = 0            # change to 1 if wrong camera opens

# UI Colors (BGR format for OpenCV)
COLOR_BG        = (30, 30, 30)       # dark background for panel
COLOR_ACCENT    = (255, 180, 0)      # gold accent
COLOR_WHITE     = (255, 255, 255)
COLOR_GREEN     = (80, 200, 80)
COLOR_GRAY      = (160, 160, 160)
COLOR_BOX       = (50, 50, 50)       # panel box background


def draw_panel(frame, prediction="---", confidence=0.0, word_buffer=""):
    """
    Draws the right-side info panel on the frame.
    Shows: current predicted sign, confidence %, word buffer, and controls.
    """
    h, w = frame.shape[:2]
    panel_x = w - PANEL_WIDTH

    # Panel background
    cv2.rectangle(frame, (panel_x, 0), (w, h), COLOR_BG, -1)

    # Divider line
    cv2.line(frame, (panel_x, 0), (panel_x, h), COLOR_ACCENT, 2)

    # ── Title ──
    cv2.putText(frame, "ISL Interpreter", (panel_x + 10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_ACCENT, 2)
    cv2.line(frame, (panel_x + 10, 45), (w - 10, 45), COLOR_GRAY, 1)

    # ── Predicted Sign ──
    cv2.putText(frame, "Sign Detected:", (panel_x + 10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)
    cv2.putText(frame, str(prediction), (panel_x + 10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, COLOR_GREEN, 3)

    # ── Confidence ──
    cv2.putText(frame, "Confidence:", (panel_x + 10, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)

    # Confidence bar background
    bar_x, bar_y, bar_w, bar_h = panel_x + 10, 175, PANEL_WIDTH - 20, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_BOX, -1)
    # Confidence bar fill
    fill = int(bar_w * confidence)
    bar_color = COLOR_GREEN if confidence > 0.75 else (0, 165, 255)
    if fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
    cv2.putText(frame, f"{int(confidence * 100)}%", (bar_x + bar_w + 4, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

    # ── Word Buffer ──
    cv2.line(frame, (panel_x + 10, 215), (w - 10, 215), COLOR_GRAY, 1)
    cv2.putText(frame, "Word Buffer:", (panel_x + 10, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)

    # Word display box
    cv2.rectangle(frame, (panel_x + 10, 250), (w - 10, 300), COLOR_BOX, -1)
    display_word = word_buffer if word_buffer else "_"
    cv2.putText(frame, display_word, (panel_x + 18, 285),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)

    # ── Controls ──
    cv2.line(frame, (panel_x + 10, 315), (w - 10, 315), COLOR_GRAY, 1)
    cv2.putText(frame, "Controls:", (panel_x + 10, 340),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ACCENT, 1)

    controls = [
        ("SPACE", "Speak word (TTS)"),
        ("BKSP",  "Delete last letter"),
        ("ENTER", "Clear buffer"),
        ("Q",     "Quit"),
    ]
    for i, (key, action) in enumerate(controls):
        y = 365 + i * 28
        cv2.putText(frame, f"[{key}]", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ACCENT, 1)
        cv2.putText(frame, action, (panel_x + 75, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_GRAY, 1)

    # ── Status dot ──
    cv2.circle(frame, (panel_x + 20, h - 20), 7, COLOR_GREEN, -1)
    cv2.putText(frame, "LIVE", (panel_x + 33, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GREEN, 1)

    return frame


def draw_hand_roi(frame):
    """
    Draws a guide box showing the user where to place their hand.
    This will be replaced by MediaPipe landmarks in Day 3.
    """
    h, w = frame.shape[:2]
    panel_x = w - PANEL_WIDTH

    # Guide box — centered in the camera area
    box_size = 350
    cx, cy = panel_x // 2, h // 2
    x1, y1 = cx - box_size // 2, cy - box_size // 2
    x2, y2 = cx + box_size // 2, cy + box_size // 2

    # Dashed-style corners
    corner_len = 30
    thickness  = 2

    for (sx, sy, dx, dy) in [
        (x1, y1,  1,  1), (x2, y1, -1,  1),
        (x1, y2,  1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (sx, sy), (sx + dx * corner_len, sy), COLOR_ACCENT, thickness)
        cv2.line(frame, (sx, sy), (sx, sy + dy * corner_len), COLOR_ACCENT, thickness)

    cv2.putText(frame, "Place hand here", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_ACCENT, 1)

    return frame


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("❌ Could not open webcam. Try changing CAM_INDEX to 1 in the config.")
        return

    print("✅ Webcam started. Press Q to quit.")

    word_buffer = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        # Flip horizontally so it feels like a mirror
        frame = cv2.flip(frame, 1)

        # TODO (Day 3): Replace these placeholders with real MediaPipe + model output
        prediction  = "---"
        confidence  = 0.0

        # Draw UI
        frame = draw_hand_roi(frame)
        frame = draw_panel(frame, prediction, confidence, word_buffer)

        cv2.imshow(WINDOW_NAME, frame)

        # ── Key handling ──
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            print("👋 Quitting.")
            break

        elif key == 8:          # Backspace — delete last letter
            word_buffer = word_buffer[:-1]

        elif key == 13:         # Enter — clear buffer
            word_buffer = ""
            print("🗑️  Buffer cleared.")

        elif key == 32:         # Spacebar — TTS will go here (Day 8)
            print(f"🔊 [TTS placeholder] Would speak: '{word_buffer}'")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()