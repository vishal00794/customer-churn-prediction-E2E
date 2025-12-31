
# src/data/preprocess.py

# This script preprocesses the Telco Customer Churn dataset
# It handles numeric conversion, splits train/test, builds a pipeline, 
# transforms data, and saves both processed data and the pipeline.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# File paths
raw_data_path = os.path.join("data", "raw", "Telco-Customer-Churn.csv")
pipeline_path = os.path.join("artifacts", "pipeline.joblib")
train_data_path = os.path.join("artifacts", "train_data.npz")
test_data_path = os.path.join("artifacts", "test_data.npz")

# Load raw data
data = pd.read_csv(raw_data_path)

# Convert target column 'Churn' to numeric: Yes -> 1, No -> 0
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

# Convert TotalCharges to numeric (empty strings become NaN)
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

# Separate features and target
X = data.drop(columns=["Churn", "customerID"])  # Drop ID column
y = data["Churn"]

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Split dataset into training and test sets
# Stratify to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Build preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine pipelines into a ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# Fit preprocessor on training data only
preprocessor.fit(X_train)

# Transform both train and test data
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save processed data using npz (handles object arrays safely)
np.savez(train_data_path, X=X_train_processed, y=y_train.to_numpy())
np.savez(test_data_path, X=X_test_processed, y=y_test.to_numpy())

# Save preprocessing pipeline for future inference
joblib.dump(preprocessor, pipeline_path)

print("Preprocessing completed. Processed data and pipeline saved in artifacts folder.")
