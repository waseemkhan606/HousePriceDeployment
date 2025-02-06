from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI()

# Load the trained model and scaler
model = joblib.load("../models/house_price_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

# Define the expected input data format
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
