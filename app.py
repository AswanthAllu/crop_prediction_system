from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import json

app = Flask(__name__)

# --- Load ML Model and the new Multi-Lingual Crop Database ---
model = None
crop_database = {}

try:
    with open('crop_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    # Load the new, detailed, multi-lingual database file, ensuring UTF-8 encoding
    with open('crop_database.json', 'r', encoding='utf-8') as db_file:
        crop_database = json.load(db_file)
    print("Multi-lingual crop database loaded successfully.")
except Exception as e:
    print(f"Error loading crop database: {e}")

# --- Global variable for latest data ---
latest_data = {
    "temperature": 0, "humidity": 0, "ph": 0, "rainfall": 0,
    "prediction": "Waiting for data..."
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_data', methods=['POST'])
def update_data():
    global latest_data
    data = request.get_json()
    
    try:
        temp = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])

        predicted_crop = "N/A"
        if model:
            features = np.array([[temp, humidity, ph, rainfall]])
            prediction_result = model.predict(features)
            predicted_crop = prediction_result[0]
        else:
            predicted_crop = "Model not loaded"

        latest_data = {
            "temperature": temp,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
            "prediction": predicted_crop.capitalize()
        }
        return jsonify({"status": "success", "prediction": predicted_crop.capitalize()})
    except Exception as e:
        print(f"Error processing data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_data')
def get_data():
    return jsonify(latest_data)

# --- This endpoint now serves the full multi-lingual data for a crop ---
@app.route('/get_crop_info/<crop_name>')
def get_crop_info(crop_name):
    crop_name_lower = crop_name.lower()
    # Fetch info from the loaded database, provide a default if not found
    crop_info = crop_database.get(crop_name_lower, {
        "en": {
            "description": "No detailed information available for this crop.",
            "fertilizers": [{"stage": "N/A", "recommendation": "N/A"}],
            "pesticides": [{"problem": "N/A", "chemical": "N/A", "dosage_per_acre": "N/A", "application": "N/A"}],
            "image_url": "https://i.imgur.com/gimmyS2.png" # A placeholder image
        }
    })
    return jsonify(crop_info)

if __name__ == '__main__':
    # For production, debug should be False. 
    # Render will set the host and port.
    app.run(host='0.0.0.0', port=5001, debug=False)