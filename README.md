# ✊🖐️✌️ CamPaperScissors: Real-Time Rock Paper Scissors

An interactive, real-time Rock, Paper, Scissors game powered by computer vision and machine learning. Using **OpenCV** and **MediaPipe Hand Landmarker**, the application tracks 21 3D hand keypoints from a live webcam feed and predicts player gestures to play against an AI opponent.

---

## 🌟 Key Features

* **Real-Time Gesture Tracking:** Extracts 21 3D hand coordinates on every frame with low latency using MediaPipe.
* **ML Gesture Classification:** Uses a trained Machine Learning model (Random forest) (`rps_model.pkl`) to accurately detect Rock, Paper, or Scissors hand signs. A random forest classifier was used because it is capable of performing exceptionally on structured tabular data without requiring feature scaling or normalization. It also mitigates overfitting due to noisy live cam training samples by building a collection of decision trees and averaging their predictions

* **Interactive Gameplay:** Play against a computer opponent directly in your terminal/webcam stream.

* **Custom Data Collection:** Includes scripts to record custom hand landmark datasets for fine-tuning or retraining model weights.

---

## 📁 Repository Structure

```text
├── collect_data.py    # Script to capture hand landmarks via webcam and save to dataset
├── train_model.py     # Script to train and export the machine learning model
├── play_game.py       # Main application loop for playing against the computer
├── requirements.txt   # Required Python dependencies
├── .gitignore         # Excludes model artifacts, datasets, and checkpoint files
└── README.md          # Project documentation