from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from transformer import AreaPerBedroomTransformer

# Load trained pipeline
try:
    model = joblib.load("best_model.pkl")
except FileNotFoundError:
    raise FileNotFoundError("Model file 'best_model.pkl' not found.")

FEATURE_COLUMNS = ['city', 'area', 'category',
                   'purpose', 'bathroom', 'bedroom']

app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Validate required fields
        required = ['city', 'area', 'category', 'purpose']
        if not all(key in data for key in required):
            return jsonify({"error": f"Missing required fields: {required}"}), 400
        if data['area'] <= 0:
            return jsonify({"error": "Area must be positive"}), 400

        # Convert to DataFrame
        df = pd.DataFrame([{
            'city': data['city'],
            'area': float(data['area']),
            'category': data['category'],
            'purpose': data['purpose'],
            'bathroom': float(data.get('bathroom', 0)),
            'bedroom': float(data.get('bedroom', 0))
        }], columns=FEATURE_COLUMNS)

        # Predict and clip
        log_pred = model.predict(df)
        max_price = 1e9  # Adjust based on dataset max
        log_pred = np.clip(log_pred, np.log1p(100), np.log1p(max_price))
        price_pred = np.expm1(log_pred)

        return jsonify({
            "predicted_price": round(float(price_pred[0]), 2),
            "currency": "NPR",
            "model_version": "2025-08-22",
            "notes": "Price in Nepali Rupees; area in sqft"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(port=5000, debug=True)
