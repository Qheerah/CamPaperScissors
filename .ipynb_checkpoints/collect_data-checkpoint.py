import cv2
import csv
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

CSV_FILE = "landmarks_dataset.csv"
MODEL_PATH = "hand_landmarker.task"

# Check for required model file
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Missing '{MODEL_PATH}'. Download it via terminal:\n"
        "curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )

# Initialize CSV file with headers if missing
if not os.path.exists(CSV_FILE):
    headers = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + ["label"]
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

# Initialize Modern MediaPipe HandLandmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

def extract_normalized_landmarks(hand_landmarks):
    """
    Extracts 21 (x, y) coordinates and translates them 
    relative to the wrist (landmark 0).
    """
    coords = np.array([[lm.x, lm.y] for lm in hand_landmarks])
    wrist = coords[0]
    normalized = coords - wrist
    return list(normalized[:, 0]) + list(normalized[:, 1])

cap = cv2.VideoCapture(0)
sample_counts = {"Rock": 0, "Paper": 0, "Scissors": 0, "None": 0}

print("\n--- CONTROLS ---")
print("Press 'r' -> Record Rock sample")
print("Press 'p' -> Record Paper sample")
print("Press 's' -> Record Scissors sample")
print("Press 'n' -> Record None/Idle sample")
print("Press 'q' -> Quit\n")

# Connections between key hand joints for visualization
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Convert BGR to RGB and prepare MediaPipe Image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Process frame
    detection_result = detector.detect(mp_image)

    current_features = None
    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]
        current_features = extract_normalized_landmarks(hand_landmarks)

        # Draw hand skeletal points & connections
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for p1, p2 in HAND_CONNECTIONS:
            cv2.line(frame, pts[p1], pts[p2], (0, 255, 0), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)

    # UI Overlay
    status_text = "Hand Detected" if current_features else "No Hand Detected"
    color = (0, 255, 0) if current_features else (0, 0, 255)
    cv2.putText(frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    counts_str = f"R:{sample_counts['Rock']} | P:{sample_counts['Paper']} | S:{sample_counts['Scissors']} | N:{sample_counts['None']}"
    cv2.putText(frame, f"Samples: {counts_str}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Data Collector (MediaPipe 1.0+)", frame)
    key = cv2.waitKey(1) & 0xFF

    label = None
    if key == ord('r'):
        label = "Rock"
    elif key == ord('p'):
        label = "Paper"
    elif key == ord('s'):
        label = "Scissors"
    elif key == ord('n'):
        label = "None"
    elif key == ord('q'):
        break

    if label:
        if current_features:
            with open(CSV_FILE, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(current_features + [label])
            sample_counts[label] += 1
            print(f"Recorded sample #{sample_counts[label]} for [{label}]")
        else:
            print(f"Cannot save '{label}': No hand detected!")

cap.release()
cv2.destroyAllWindows()