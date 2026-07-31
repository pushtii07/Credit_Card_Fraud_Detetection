from flask import Flask,request, jsonify
import pickle
import joblib
import numpy as np 

app=Flask(__name__)
model=joblib.load("fraud_model(1).pkl")
scaler = joblib.load("scaler(3).pkl")

@app.route("/")
def home():
    return "Credit Card Fraud Detection API is Running!"
@app.route("/predict", methods=["POST"])


@app.route("/predict", methods=["POST"])
def predict():

    # Receive JSON data
    data = request.get_json()

    # Convert to NumPy array
    features = np.array(data).reshape(1, -1)

    # Scale only Time and Amount
    time_amount = features[:, [0, 29]]
    time_amount = scaler.transform(time_amount)

    features[:, [0, 29]] = time_amount

    # Predict
    prediction = model.predict(features)

    # Return response
    if prediction[0] == 1:
        return jsonify({
            "prediction": "Fraud Transaction"
        })
    else:
        return jsonify({
            "prediction": "Normal Transaction"
        })

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
