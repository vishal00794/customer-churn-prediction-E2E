# src/api/app.py

# FastAPI app to serve the trained customer churn model
# Accepts JSON input with customer features and returns prediction

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import os

# Initialize FastAPI
app = FastAPI(title="Customer Churn Prediction API")

# Root endpoint to show API is running
@app.get("/")
def read_root():
    return {"message": "Customer Churn Prediction API is running. Use /docs to test the endpoints."}

# Load preprocessing pipeline and trained model
pipeline_path = os.path.join("artifacts", "pipeline.joblib")
model_path = os.path.join("artifacts", "logistic_model.joblib")

preprocessor = joblib.load(pipeline_path)
model = joblib.load(model_path)

# Define expected input features for one customer
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Prediction endpoint
@app.post("/predict")
def predict_churn(customer: CustomerData):
    # Convert input to DataFrame
    customer_df = pd.DataFrame([customer.dict()])
    
    # Transform features using preprocessing pipeline
    X_processed = preprocessor.transform(customer_df)
    
    # Make prediction
    pred = model.predict(X_processed)[0]
    prob = model.predict_proba(X_processed)[0][1]  # probability of churn
    
    # Return results
    return {
        "churn_prediction": int(pred),
        "churn_probability": float(prob)
    }
