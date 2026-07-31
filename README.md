# Credit Card Fraud Detection API

## Project Overview
This project uses a Random Forest Classifier to detect fraudulent credit card transactions. The model is deployed using Flask and exposes a REST API for predictions.

## Technologies Used
- Python
- Flask
- Scikit-learn
- NumPy
- Pandas
- Joblib

## Model
- Algorithm: Random Forest Classifier
- Features: 30 (Time, V1–V28, Amount)

## API Endpoint

POST /predict

Input:
A JSON array containing 30 feature values.

Output:

```json
{
    "prediction": "Normal Transaction"
}
```

or

```json
{
    "prediction": "Fraud Transaction"
}
```