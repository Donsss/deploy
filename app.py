from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__)

# Daftar origin yang diizinkan
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "https://coba-coba-deploy.netlify.app"
]

# Konfigurasi CORS
CORS(app, resources={
    r"/predict_crop": {
        "origins": ALLOWED_ORIGINS,
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
    # Handle preflight request
    if request.method == 'OPTIONS':
        response = jsonify({"message": "Preflight successful"})
        response.headers.add("Access-Control-Allow-Origin", request.headers.get('Origin'))
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST")
        return response

    # Handle POST request
    try:
        # Validasi origin
        origin = request.headers.get('Origin')
        if origin not in ALLOWED_ORIGINS:
            return jsonify({"error": "Origin not allowed"}), 403

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
        response.headers.add("Access-Control-Allow-Origin", origin)
        return response

    except Exception as e:
        error_response = jsonify({"error": str(e)})
        error_response.headers.add("Access-Control-Allow-Origin", request.headers.get('Origin'))
        return error_response, 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
