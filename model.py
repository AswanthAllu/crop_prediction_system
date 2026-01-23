import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

filename = 'crop_recommandation_dataset.csv'

if not os.path.exists(filename):
    print(f"Error: The file '{filename}' was not found in this folder.")
    print("Please check the spelling or make sure it is in the same folder as this script.")
    exit()

print(f"Loading dataset: {filename}...")
df = pd.read_csv(filename)

if 'label' in df.columns:
    target_col = 'label'
elif 'class' in df.columns:
    target_col = 'class'
else:
    print("Error: Could not find a 'label' or 'class' column in your CSV.")
    print("Your columns are:", df.columns.tolist())
    exit()

required_features = ['temperature', 'humidity', 'ph', 'rainfall']

missing_cols = [col for col in required_features if col not in df.columns]
if missing_cols:
    print(f"Error: Your dataset is missing these required columns: {missing_cols}")
    print("Your dataset columns are:", df.columns.tolist())
    exit()

print(f"Training on features: {required_features}")
X = df[required_features]
y = df[target_col]

print("Training Random Forest Model...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Model Trained Successfully!")
print(f"Accuracy: {accuracy * 100:.2f}%")

output_file = 'crop_model.pkl'
joblib.dump(model, output_file)
print(f"Model saved as '{output_file}'")
print("You can now run 'python app.py'")
