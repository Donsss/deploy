from flask import Flask, request, jsonify
import joblib
import json
import pandas as pd
import numpy as np
from io import BytesIO
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

model = joblib.load('models/model.joblib')
scaler = joblib.load('models/scaler.joblib')
label_encoder = joblib.load('models/label_encoder.joblib')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
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

    print('Response:', crop_response)
    return jsonify(crop_response)

if __name__ == '__main__':
    app.run(debug=True)
