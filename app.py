from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

model = None
# Corrected model path for better compatibility
model_path = 'crop_model.pkl' 
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: Model file '{model_path}' not found. Predictions will not work.")

latest_data = {
    "temperature": 0.0,
    "humidity": 0.0,
    "ph": 0.0,
    "rainfall": 0.0,
    "prediction": "N/A - Waiting for data"
}

VALID_RANGES = {
    "temperature": {"min": 0.0, "max": 60.0},
    "humidity": {"min": 10.0, "max": 100.0},
    "ph": {"min": 3.0, "max": 9.5},
    "rainfall": {"min": 0.0, "max": 500.0}
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_data', methods=['POST'])
def update_data():
    global latest_data
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No JSON data received"}), 400

    print(f"Received data from sensor: {data}")

    required_keys = ['temperature', 'humidity', 'ph', 'rainfall']
    if not all(key in data for key in required_keys):
        return jsonify({"status": "error", "message": "Missing data fields"}), 400

    try:
        temp = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])

        # Simple validation
        if not (VALID_RANGES["temperature"]["min"] <= temp <= VALID_RANGES["temperature"]["max"]):
            raise ValueError("Invalid temperature reading")
        if not (VALID_RANGES["humidity"]["min"] <= humidity <= VALID_RANGES["humidity"]["max"]):
             raise ValueError("Invalid humidity reading")

        predicted_crop = "N/A - Model not loaded"
        if model:
            features = np.array([[temp, humidity, ph, rainfall]])
            prediction_result = model.predict(features)
            predicted_crop = prediction_result[0].capitalize()
        
        latest_data = {
            "temperature": temp,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
            "prediction": predicted_crop
        }
        return jsonify({"status": "success", "message": "Data updated"})

    except (ValueError, TypeError) as e:
        return jsonify({"status": "error", "message": f"Invalid data type: {e}"}), 400
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500

@app.route('/get_data')
def get_data():
    return jsonify(latest_data)

if __name__ == '__main__':
    # Render uses a production web server like Gunicorn, so this part is mainly for local testing
    app.run(host='0.0.0.0', port=5001, debug=True)
