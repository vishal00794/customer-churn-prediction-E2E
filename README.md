
# Customer Churn Prediction - End-to-End ML Project

This project demonstrates an end-to-end machine learning workflow, including data preprocessing, model training, API deployment, and Dockerization.

## Project Overview
- Dataset: Telco Customer Churn
- Goal: Predict if a customer will churn (Yes/No)
- Model: Logistic Regression (baseline)
- API: FastAPI to serve predictions
- Deployment: Docker container

## Project Structure
customer-churn-e2e/
├── data/ # Raw dataset
├── src/
│ ├── data/ # Preprocessing scripts
│ ├── models/ # Training scripts
│ └── api/ # FastAPI app
├── artifacts/ # Preprocessed data, trained model, pipeline
├── Dockerfile
├── requirements.txt
└── README.md


## How to Run Locally

### 1. Preprocessing
```bash
python src/data/preprocess.py
```

### 1. Preprocessing
```
python src/models/train.py
```
### 2. Train Model
```
python src/models/train.py
```

### 3. Run FastAPI
```
uvicorn src.api.app:app --reload
```

### 3. Run FastAPI
```
API access for test: http://127.0.0.1:8000/docs
```
sample to test : 
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}

### 4. Run Docker Container
```
docker build -t customer-churn-api .
docker run -p 8000:8000 customer-churn-api
```
