# src/models/train.py

# This script trains a baseline Logistic Regression model for customer churn
# It evaluates the model and saves it for future use (e.g., API deployment)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import joblib
import os

# Paths to processed data
train_data_path = os.path.join("artifacts", "train_data.npz")
test_data_path = os.path.join("artifacts", "test_data.npz")
model_path = os.path.join("artifacts", "logistic_model.joblib")

# Load processed training data
train_data = np.load(train_data_path)
X_train = train_data["X"]
y_train = train_data["y"]

# Load processed test data
test_data = np.load(test_data_path)
X_test = test_data["X"]
y_test = test_data["y"]

# Initialize Logistic Regression model
# solver 'liblinear' is suitable for small datasets
model = LogisticRegression(solver='liblinear', random_state=42)

# Fit model on training data
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # probability of class 1

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print evaluation results
print("Model Evaluation Metrics")
print("Accuracy:", accuracy)
print("ROC AUC Score:", roc_auc)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)

# Save trained model for future use (API deployment, inference)
joblib.dump(model, model_path)
print("Trained Logistic Regression model saved at", model_path)
