import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
dataset_path = "landmarks_dataset.csv"
try:
    data = pd.read_csv(dataset_path)
    print(f"Loaded raw dataset with {len(data)} rows.")
except FileNotFoundError:
    print(f"Error: Could not find '{dataset_path}'. Run collect_data.py first!")
    exit()

# 2. Clean missing values (NaNs)
data = data.dropna()
print(f"Dataset size after removing NaNs: {len(data)} rows.")

if len(data) < 10:
    print("Not enough samples to train! Run collect_data.py to collect more samples.")
    exit()

# 3. Separate features (42 landmark values) and labels
X = data.drop("label", axis=1)
y = data["label"]

# 4. Train/Test Split (80% training, 20% evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train the Random Forest Classifier
print("Training Random Forest model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 6. Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- MODEL RESULTS ---")
print(f"Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

# 7. Save the trained model to disk
model_path = "rps_model.pkl"
joblib.dump(clf, model_path)
print(f"Model saved successfully as '{model_path}'!")