# Login_Risk_Model
A machine learning model to predict login risk based on user behavior using RandomForestClassifier.

# Login Risk Prediction Using Machine Learning

This project predicts whether a user login attempt is safe or risky using a machine learning model (RandomForestClassifier). It is trained on behavioral data such as login time, number of attempts, device used, and location.

## Features Used
- Username (encoded)
- Time of Day (Morning, Afternoon, Evening, Night)
- Login Location (encoded)
- Number of Login Attempts
- Device Type (encoded)

## Model Used
- RandomForestClassifier from scikit-learn
- Trained with both real and AI-generated synthetic data
- Evaluated with accuracy, confusion matrix, and feature importance

## Technologies
- Python
- Google Colab
- Pandas
- scikit-learn
- Joblib (for model saving)

## Files Included
- `LoginRiskProject.ipynb` – Complete notebook with data prep, training, and testing
- `login_risk_model.pkl` – Saved trained model
- `synthetic_data.csv` – AI-generated dataset
- `README.md` – Project description

## How to Run

1. Clone or download the repository
2. Load the model and test with a custom input:

```python
import joblib

# Load trained model
model = joblib.load('login_risk_model.pkl')

# Example login attempt: [Username, TimeOfDay, Location, NumAttempts, Device]
sample = [[2, 3, 1, 4, 1]]

# Predict
prediction = model.predict(sample)

print("Prediction:", "Risky" if prediction[0] == 1 else "Safe")

