from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
# Konfigurasi CORS lebih ketat
CORS(app, resources={
    r"/predict_crop": {
        "origins": ["http://localhost:5173","https://coba-coba-deploy.netlify.app/"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Load model
model = joblib.load('models/model.joblib')
scaler = joblib.load('models/scaler.joblib')
label_encoder = joblib.load('models/label_encoder.joblib')

@app.route('/predict_crop', methods=['POST', 'OPTIONS'])
def predict_crop():
    if request.method == 'OPTIONS':
        # Response preflight khusus
        response = jsonify({"message": "Preflight successful"})
        response.headers.add("Access-Control-Allow-Origin", "http://localhost:5173")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST")
        return response, 200  # Pastikan status code 200

    # Handle POST request
    try:
        data_input = request.json
        array = pd.DataFrame([{
            'N': data_input['N'],
            'P': data_input['P'],
            'K': data_input['K'],
            'temperature': data_input['temperature'],
            'humidity': data_input['humidity'],
            'ph': data_input['ph'],
            'rainfall': data_input['rainfall']
        }])

        scaled_data = scaler.transform(array)
        prediction = model.predict(scaled_data)
        prediction_proba = model.predict_proba(scaled_data)
        confidence = np.max(prediction_proba)

        crop_response = {
            'prediction': int(prediction[0]),
            'confidence': float(confidence),
            'labelEncoder': label_encoder.inverse_transform([int(prediction[0])])[0]
        }

        response = jsonify(crop_response)
        response.headers.add("Access-Control-Allow-Origin", "http://localhost:5173")
        return response

    except Exception as e:
        error_response = jsonify({"error": str(e)})
        error_response.headers.add("Access-Control-Allow-Origin", "http://localhost:5173")
        return error_response, 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
