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

WINDOW_NAME    = "Signify"
FRAME_WIDTH    = 1280
FRAME_HEIGHT   = 720
CAM_INDEX      = 0
MP_MODEL_PATH  = "hand_landmarker.task"
ISL_MODEL_PATH = "isl_model.pkl"
BG_IMAGE_PATH  = "notebook_bg.png"

SMOOTH_FRAMES  = 5
HOLD_TIME      = 1.0
MIN_CONFIDENCE = 0.4

# Minimalist monochrome palette
COLOR_BG         = (255, 255, 255)  # Pure white
COLOR_BLACK      = (20, 20, 20)     # Almost black
COLOR_GRAY       = (120, 120, 120)  # Mid gray
COLOR_LIGHT_GRAY = (200, 200, 200)  # Light gray
COLOR_ACCENT     = (40, 40, 40)     # Dark accent
COLOR_SHADOW     = (230, 230, 230)  # Very light shadow

HAND_COLORS = [(80, 80, 80), (140, 140, 140)]  # Grayscale hands

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
        print("Model not found at:", model_path)
        return None
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("ISL model loaded")
    return model

def download_mp_model():
    if os.path.exists(MP_MODEL_PATH):
        return
    print("Downloading MediaPipe model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, MP_MODEL_PATH)
    print("Downloaded")

class PredictionSmoother:
    def __init__(self, window_size=SMOOTH_FRAMES):
        self.history = deque(maxlen=window_size)

    def update(self, prediction):
        self.history.append(prediction)

    def get_smooth(self):
        if not self.history:
            return "---", 0.0
        counts = Counter(self.history)
        best = counts.most_common(1)[0]
        label = best[0]
        conf = best[1] / len(self.history)
        return label, conf

    def clear(self):
        self.history.clear()

def load_notebook_background():
    """Load the notebook background and convert to clean white"""
    if os.path.exists(BG_IMAGE_PATH):
        bg = cv2.imread(BG_IMAGE_PATH)
        if bg is not None:
            bg = cv2.resize(bg, (FRAME_WIDTH, FRAME_HEIGHT))
            # Convert to clean white
            bg[:] = COLOR_BG
            return bg
    
    # Fallback: pure white
    bg = np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8) * 255
    return bg

def draw_text_compressed(canvas, text, x, y, size=1.0, color=None, thickness=2, italic=True):
    """Draw text with compressed spacing for that cool aesthetic"""
    if color is None:
        color = COLOR_BLACK
    
    # Use HERSHEY_COMPLEX for cleaner look, add italic flag
    font = cv2.FONT_HERSHEY_TRIPLEX if not italic else cv2.FONT_ITALIC
    
    # Get text dimensions
    (text_width, text_height), baseline = cv2.getTextSize(text, font, size, thickness)
    
    # Draw each character with reduced spacing
    current_x = x
    char_spacing = -0.15  # Negative for compression (adjust between -0.2 to 0)
    
    for char in text:
        cv2.putText(canvas, char, (int(current_x), y), font, size, color, thickness, cv2.LINE_AA)
        char_width = cv2.getTextSize(char, font, size, thickness)[0][0]
        current_x += char_width * (1 + char_spacing)

def draw_minimalist_frame(canvas, video_frame, x, y, width, height):
    """Draw video feed with minimal, clean border"""
    # Subtle shadow
    shadow_offset = 6
    cv2.rectangle(canvas, 
                 (x + shadow_offset, y + shadow_offset), 
                 (x + width + shadow_offset, y + height + shadow_offset),
                 COLOR_SHADOW, -1)
    
    # Clean white frame with thin black border
    cv2.rectangle(canvas, (x, y), (x + width, y + height), COLOR_BG, -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), COLOR_BLACK, 2)
    
    # Calculate target size for video area
    target_w = width - 16
    target_h = height - 16
    
    # Get video dimensions and crop to fit
    video_h, video_w = video_frame.shape[:2]
    video_aspect = video_w / video_h
    target_aspect = target_w / target_h
    
    if video_aspect > target_aspect:
        new_w = int(video_h * target_aspect)
        crop_x = (video_w - new_w) // 2
        video_cropped = video_frame[:, crop_x:crop_x + new_w]
    else:
        new_h = int(video_w / target_aspect)
        crop_y = (video_h - new_h) // 2
        video_cropped = video_frame[crop_y:crop_y + new_h, :]
    
    video_resized = cv2.resize(video_cropped, (target_w, target_h))
    canvas[y + 8:y + 8 + target_h, x + 8:x + 8 + target_w] = video_resized

def draw_info_card(canvas, x, y, width, height, label, value, show_bar=False, bar_value=0.0):
    """Draw minimal info card"""
    # Shadow
    cv2.rectangle(canvas, (x + 4, y + 4), (x + width + 4, y + height + 4),
                 COLOR_SHADOW, -1)
    
    # Card background
    cv2.rectangle(canvas, (x, y), (x + width, y + height), COLOR_BG, -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), COLOR_BLACK, 2)
    
    # Label - compressed italic
    draw_text_compressed(canvas, label.upper(), x + 15, y + 28, 0.5, COLOR_GRAY, 1)
    
    # Value - larger, bold
    cv2.putText(canvas, str(value), (x + 15, y + 65),
               cv2.FONT_HERSHEY_SIMPLEX, 1.6, COLOR_BLACK, 3, cv2.LINE_AA)
    
    # Optional progress bar for confidence/hold
    if show_bar:
        bar_y = y + height - 20
        bar_w = width - 30
        
        # Bar background
        cv2.rectangle(canvas, (x + 15, bar_y), (x + 15 + bar_w, bar_y + 6),
                     COLOR_LIGHT_GRAY, -1)
        
        # Bar fill
        fill_w = int(bar_w * min(bar_value, 1.0))
        if fill_w > 0:
            cv2.rectangle(canvas, (x + 15, bar_y), (x + 15 + fill_w, bar_y + 6),
                         COLOR_BLACK, -1)

def draw_minimalist_badge(canvas, x, y, size, value, label):
    """Draw minimal circular badge"""
    # Shadow
    cv2.circle(canvas, (x + 3, y + 3), size, COLOR_SHADOW, -1)
    
    # Circle
    cv2.circle(canvas, (x, y), size, COLOR_BG, -1)
    cv2.circle(canvas, (x, y), size, COLOR_BLACK, 2)
    
    # Value
    text_size = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    text_x = x - text_size[0] // 2
    text_y = y + text_size[1] // 2
    cv2.putText(canvas, value, (text_x, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_BLACK, 2, cv2.LINE_AA)
    
    # Label below
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    cv2.putText(canvas, label, (x - label_size[0] // 2, y + size + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1, cv2.LINE_AA)

def draw_accent_line(canvas, x1, y1, x2, y2):
    """Draw minimal accent line"""
    cv2.line(canvas, (x1, y1), (x2, y2), COLOR_LIGHT_GRAY, 1)

def draw_notebook_interface(canvas, video_with_hands, prediction, confidence, 
                           hands_detected, hold_progress):
    """Minimalist interface"""
    h, w = canvas.shape[:2]
    
    # Pure white background
    canvas[:] = COLOR_BG
    
    # Load background (will be white anyway)
    bg = load_notebook_background()
    canvas[:] = bg
    
    # Top left: Brand
    draw_text_compressed(canvas, "SIGNIFY", 60, 70, 2.0, COLOR_BLACK, 3, italic=True)
    
    # Subtitle - even more compressed
    draw_text_compressed(canvas, "ISL  INTERPRETER", 60, 105, 0.6, COLOR_GRAY, 1, italic=True)
    
    # Thin accent line under header
    cv2.line(canvas, (60, 120), (350, 120), COLOR_BLACK, 2)
    
    # Main video frame - larger, centered left
    video_x = 60
    video_y = 160
    video_w = 580
    video_h = 435
    draw_minimalist_frame(canvas, video_with_hands, video_x, video_y, video_w, video_h)
    
    # Caption under video
    draw_text_compressed(canvas, "LIVE", video_x + 10, video_y + video_h + 35, 
                        0.7, COLOR_GRAY, 1, italic=True)
    
    # Right side - info cards stacked vertically
    card_x = 700
    card_width = 220
    card_height = 110
    
    # Current sign card
    draw_info_card(canvas, card_x, 160, card_width, card_height, 
                  "Current Sign", prediction)
    
    # Confidence card with bar
    conf_percentage = f"{int(confidence * 100)}%"
    draw_info_card(canvas, card_x, 290, card_width, card_height,
                  "Confidence", conf_percentage, 
                  show_bar=True, bar_value=confidence)
    
    # Hands detected badge
    badge_x = card_x + 110
    badge_y = 480
    draw_minimalist_badge(canvas, badge_x, badge_y, 45, 
                         f"{hands_detected}/2", "HANDS")
    
    # Hold progress (only show when active)
    if hold_progress > 0:
        hold_y = 560
        cv2.rectangle(canvas, (card_x + 3, hold_y + 3), 
                     (card_x + card_width + 3, hold_y + 30 + 3),
                     COLOR_SHADOW, -1)
        cv2.rectangle(canvas, (card_x, hold_y), 
                     (card_x + card_width, hold_y + 30),
                     COLOR_BG, -1)
        cv2.rectangle(canvas, (card_x, hold_y), 
                     (card_x + card_width, hold_y + 30),
                     COLOR_BLACK, 2)
        
        # Progress bar
        bar_w = card_width - 20
        fill_w = int(bar_w * hold_progress)
        
        cv2.rectangle(canvas, (card_x + 10, hold_y + 8),
                     (card_x + 10 + bar_w, hold_y + 22),
                     COLOR_LIGHT_GRAY, -1)
        
        if fill_w > 0:
            cv2.rectangle(canvas, (card_x + 10, hold_y + 8),
                         (card_x + 10 + fill_w, hold_y + 22),
                         COLOR_BLACK, -1)
    
    # Bottom right - minimal controls
    ctrl_y = 630
    draw_text_compressed(canvas, "Q  QUIT", card_x, ctrl_y, 0.45, COLOR_GRAY, 1, italic=True)
    draw_text_compressed(canvas, "F  FULLSCREEN", card_x, ctrl_y + 25, 0.45, COLOR_GRAY, 1, italic=True)
    
    # Decorative elements - minimal geometric shapes
    # Small squares as design accents
    accent_size = 8
    cv2.rectangle(canvas, (980, 200), (980 + accent_size, 200 + accent_size), 
                 COLOR_BLACK, -1)
    cv2.rectangle(canvas, (970, 350), (970 + accent_size, 350 + accent_size), 
                 COLOR_LIGHT_GRAY, -1)
    
    # Thin lines as design elements
    cv2.line(canvas, (card_x - 30, 160), (card_x - 30, 600), COLOR_LIGHT_GRAY, 1)

def draw_landmarks(frame, landmarks_px, hand_index=0):
    """Draw hand landmarks - minimal grayscale"""
    dot_color = HAND_COLORS[hand_index % 2]
    
    # Connections - thin lines
    for (a, b) in HAND_CONNECTIONS:
        if a < len(landmarks_px) and b < len(landmarks_px):
            cv2.line(frame, landmarks_px[a], landmarks_px[b], COLOR_LIGHT_GRAY, 2, cv2.LINE_AA)
    
    # Landmarks - small circles
    for (px, py) in landmarks_px:
        cv2.circle(frame, (px, py), 4, dot_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 4, COLOR_BLACK, 1, cv2.LINE_AA)

def extract_landmarks(hand_landmarks, frame_w, frame_h):
    coords = []
    points = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])
        px = int(lm.x * frame_w)
        py = int(lm.y * frame_h)
        points.append((px, py))
    return coords, points

def predict_sign_two_hands(model, left_coords, right_coords):
    if left_coords is None or len(left_coords) < 63:
        left_coords = [0.0] * 63
    if right_coords is None or len(right_coords) < 63:
        right_coords = [0.0] * 63
    
    combined = np.array(left_coords[:63] + right_coords[:63]).reshape(1, -1)
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    
    try:
        prediction = str(model.predict(combined)[0])
        proba = model.predict_proba(combined)[0]
        confidence = float(np.max(proba))
        return prediction, confidence
    except Exception as e:
        return "---", 0.0

def main():
    download_mp_model()
    
    isl_model = load_isl_model(ISL_MODEL_PATH)
    if isl_model is None:
        return
    
    base_options = python.BaseOptions(model_asset_path=MP_MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    detector = vision.HandLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("Webcam failed")
        return
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    print("\nSignify - ISL Interpreter")
    print("Press Q to quit\n")
    
    is_fullscreen = True
    smoother = PredictionSmoother(SMOOTH_FRAMES)
    hold_start = None
    last_prediction = "---"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)
        
        left_coords = None
        right_coords = None
        hands_detected = 0
        
        # Extract hands
        if results.hand_landmarks and results.handedness:
            hands_detected = len(results.hand_landmarks)
            
            for hand_lms, handedness in zip(results.hand_landmarks, results.handedness):
                hand_label = handedness[0].category_name
                coords, landmarks_px = extract_landmarks(hand_lms, w, h)
                
                if hand_label == "Left":
                    left_coords = coords
                    draw_landmarks(frame, landmarks_px, 0)
                elif hand_label == "Right":
                    right_coords = coords
                    draw_landmarks(frame, landmarks_px, 1)
        else:
            smoother.clear()
            hold_start = None
            last_prediction = "---"
        
        # Predict
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
                elapsed = time.time() - hold_start
                hold_progress = elapsed / HOLD_TIME
                
                if elapsed >= HOLD_TIME and smooth_confidence >= MIN_CONFIDENCE:
                    print(f"Detected: {smooth_prediction}")
                    hold_start = None
                    smoother.clear()
            else:
                last_prediction = smooth_prediction
                hold_start = None
        else:
            hold_start = None
            last_prediction = smooth_prediction
        
        # Create canvas and draw interface
        canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        draw_notebook_interface(canvas, frame, smooth_prediction, smooth_confidence,
                               hands_detected, hold_progress)
        
        cv2.imshow(WINDOW_NAME, canvas)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key in (ord('q'), ord('Q')):
            break
        elif key == ord('f') or key == ord('F'):
            is_fullscreen = not is_fullscreen
            prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, prop)
    
    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()