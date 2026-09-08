import cv2
import time
import random
import os
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
PKL_PATH = "rps_model.pkl"

# Verify files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(PKL_PATH):
    raise FileNotFoundError("Missing required model files! Ensure 'hand_landmarker.task' and 'rps_model.pkl' are present.")

# Load Scikit-Learn Model
classifier = joblib.load(PKL_PATH)

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
    """Translates 21 landmark coordinates relative to the wrist."""
    coords = np.array([[lm.x, lm.y] for lm in hand_landmarks])
    wrist = coords[0]
    normalized = coords - wrist
    return list(normalized[:, 0]) + list(normalized[:, 1])

def get_winner(player, computer):
    if player == computer:
        return "Tie"
    rules = {"Rock": "Scissors", "Paper": "Rock", "Scissors": "Paper"}
    if rules.get(player) == computer:
        return "Player"
    return "Computer"

# Connections for visualization
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

cap = cv2.VideoCapture(0)

player_score = 0
computer_score = 0
state = "READY"  # READY, COUNTDOWN, RESULT
countdown_start = 0
player_move = "None"
computer_move = "None"
result_text = ""

print("\n--- GAME CONTROLS ---")
print("Press 'SPACE' -> Start 3-second round countdown")
print("Press 'q'     -> Quit game\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    current_prediction = "None"
    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]
        features = extract_normalized_landmarks(hand_landmarks)
        
        # Predict gesture
        current_prediction = classifier.predict([features])[0]

        # Draw hand joints
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for p1, p2 in HAND_CONNECTIONS:
            cv2.line(frame, pts[p1], pts[p2], (0, 255, 0), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)

    # State Machine Logic
    if state == "COUNTDOWN":
        elapsed = time.time() - countdown_start
        countdown_num = 3 - int(elapsed)
        
        if countdown_num > 0:
            cv2.putText(frame, str(countdown_num), (280, 250), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8)
        else:
            # Capture moves at T=0
            player_move = current_prediction
            if player_move in ["Rock", "Paper", "Scissors"]:
                computer_move = random.choice(["Rock", "Paper", "Scissors"])
                winner = get_winner(player_move, computer_move)
                
                if winner == "Player":
                    player_score += 1
                    result_text = "You Win!"
                elif winner == "Computer":
                    computer_score += 1
                    result_text = "Computer Wins!"
                else:
                    result_text = "Tie Game!"
            else:
                result_text = "No valid move detected!"
                player_move = "None"
                computer_move = "None"
            
            state = "RESULT"

    # Screen UI
    cv2.putText(frame, f"Player: {player_score} | CPU: {computer_score}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Detected: {current_prediction}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if state == "READY":
        cv2.putText(frame, "Press SPACE to Play", (180, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    elif state == "RESULT":
        cv2.putText(frame, f"You: {player_move} | CPU: {computer_move}", (140, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, result_text, (180, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if "Win" in result_text else (0, 0, 255), 3)
        cv2.putText(frame, "Press SPACE for Next Round", (120, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Rock Paper Scissors ML", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' ') and state in ["READY", "RESULT"]:
        state = "COUNTDOWN"
        countdown_start = time.time()
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()