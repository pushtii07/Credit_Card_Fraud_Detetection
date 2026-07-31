from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
model = joblib.load("fraud_model(1).pkl")
scaler = joblib.load("scaler(3).pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Receive JSON data
    data = request.get_json()

    # ---------------- INPUT VALIDATION ----------------

    if data is None:
        return jsonify({
            "error": "No input received."
        }), 400

    if len(data) != 30:
        return jsonify({
            "error": "Please enter all 30 values."
        }), 400

    try:
        features = np.array(data, dtype=float).reshape(1, -1)
    except ValueError:
        return jsonify({
            "error": "Please enter valid numeric values."
        }), 400

    # --------------------------------------------------

    # Scale Time and Amount
    time_amount = features[:, [0, 29]]
    time_amount = scaler.transform(time_amount)
    features[:, [0, 29]] = time_amount

    # Prediction
    prediction = model.predict(features)

    if prediction[0] == 1:
        return jsonify({
            "prediction": "🚨 Fraud Transaction"
        })
    else:
        return jsonify({
            "prediction": "✅ Normal Transaction"
        })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
