# ISL Interpreter

A real-time **Indian Sign Language (ISL) interpreter** built with Python. Uses your webcam to detect hand signs and converts them into text and speech — live.

> Built as a 2-week prototype by a team of 3. Currently supports static signs (A–Z, 0–9).

---

## What It Does

- Detects your hand in real time using your webcam
- Recognizes ISL hand signs (A–Z and 0–9)
- Displays the predicted sign + confidence score on screen
- Accumulates letters into words as you sign
- Speaks the word aloud via text-to-speech

---

## Tech Stack

| Component | Library |
|---|---|
| Hand Detection | [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands) |
| Sign Classification | scikit-learn (Random Forest / SVM) |
| Webcam & UI | OpenCV |
| Text-to-Speech | pyttsx3 |
| Language | Python 3.9+ |

---

## Project Structure

```
ISL_Interpreter/
│
├── data/
│   └── landmarks.csv          # collected hand landmark dataset
│
├── model/
│   └── isl_model.pkl          # trained classifier
│
├── scripts/
│   ├── collect_data.py        # records hand landmarks to CSV (Person A)
│   ├── train_model.py         # trains the classifier (Person B)
│   └── evaluate_model.py      # confusion matrix + accuracy report (Person B)
│
├── app/
│   ├── webcam_skeleton.py     # Day 1 — basic webcam UI (Person C)
│   ├── webcam_landmarks.py    # Day 3 — MediaPipe landmarks integrated (Person C)
│   └── main.py                # Final app — full interpreter (Person C)
│
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/atconeyisland/ISL_Ineterpreter_project.git
cd ISL_Ineterpreter_project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install mediapipe opencv-python scikit-learn pyttsx3
```

### 3. Verify installation
```bash
python -c "import cv2, mediapipe, sklearn, pyttsx3; print('All good!')"
```

---

## How to Run

### Run the live interpreter (once model is trained)
```bash
python app/main.py
```

### Run the webcam skeleton (Day 1 — no model needed)
```bash
python app/webcam_skeleton.py
```

### Run with MediaPipe landmarks (Day 3 — no model needed)
```bash
python app/webcam_landmarks.py
```

---

## Controls

| Key | Action |
|---|---|
| `SPACE` | Speak the current word (TTS) |
| `BACKSPACE` | Delete last letter from buffer |
| `ENTER` | Clear the word buffer |
| `I` | Toggle landmark index numbers |
| `Q` | Quit the app |

---

## Dataset

The model is trained on hand landmark data collected by the team using MediaPipe Hands. Each sample consists of **63 values** — the x, y, z coordinates of all 21 hand landmarks.

- **36 classes** — A to Z and 0 to 9
- **~100 samples per class** per team member
- **~10,800 total samples**

To collect your own data:
```bash
python scripts/collect_data.py
```
Follow the on-screen instructions — press the label key to save a sample for that sign.

---

## How the Model Works

```
Webcam frame
    ↓
MediaPipe Hands detects 21 landmarks
    ↓
Extract x, y, z → flat list of 63 numbers
    ↓
scikit-learn classifier predicts the sign
    ↓
Display prediction + confidence on screen
```

To train the model yourself:
```bash
python scripts/train_model.py
```

---

## Current Accuracy

| Metric | Value |
|---|---|
| Overall accuracy | TBD (target: >88%) |
| Classes | 36 (A–Z, 0–9) |
| Model type | Random Forest / SVM |

> Accuracy report will be updated after final training run.

---

## Roadmap

- [x] Static sign recognition (A–Z, 0–9)
- [x] Live webcam feed with UI overlay
- [x] Text-to-speech output
- [ ] Dynamic gesture recognition (words/phrases) — LSTM model
- [ ] Sentence formation from individual signs
- [ ] Bidirectional interpreter (text/speech → ISL animation)

---

## Team

| Person | Role |
|---|---|
| Md Zakiur Rahman | Data pipeline — collection, CSV management, dataset QA |
| Anvi Trivedi | Model — training, evaluation, accuracy tuning |
| Prachi Bhowal | App — webcam UI, MediaPipe integration, TTS |

---

## Known Limitations

- Currently supports **static signs only** — no dynamic/motion-based gestures yet
- Works best in **good lighting** with a plain background
- Trained on a small team dataset — accuracy may vary across different hand sizes and skin tones
- Single hand only

---

## Contributing

This is a student prototype project. Feel free to fork, improve the dataset, or swap in a better model. Pull requests welcome!

---

## License

MIT License — free to use and modify.
