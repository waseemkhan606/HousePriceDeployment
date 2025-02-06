import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI()

# Get absolute path for the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
MODEL_PATH = os.path.join(BASE_DIR, "../models/house_price_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "../models/scaler.pkl")

# Debugging: Check if files exist before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")

# Load the trained model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Define input format
class HouseFeatures(BaseModel):
    Bedroom: float
    Space: float
    Room: float
    Lot: float
    Tax: float
    Bathroom: float
    Garage: float
    Condition: float

@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}

@app.post("/predict/")
def predict_price(features: HouseFeatures):
    """API endpoint for predicting house prices"""
    input_data = pd.DataFrame([features.dict()])

    # Scale input data using the saved scaler
    scaled_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_data)[0]

    return {"Predicted Price": round(prediction, 2)}
