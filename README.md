# Login_Risk_Model

A machine learning model that intelligently predicts whether a login attempt is **safe** or **risky**, based on user behavior patterns. This solution uses a RandomForestClassifier trained on structured and synthetic data, and is ideal for detecting unusual or suspicious login activity.

---

## Overview

The project builds a predictive model using user behavior data such as login time, number of attempts, device type, and login location. It applies classification techniques to label login attempts as either "safe" or "risky."

The model is trained, tested, and evaluated in **Google Colab**, and saved as a `.pkl` file for reuse in real-time applications.

---

## Features Used in the Model

| Feature         | Description                                        |
|----------------|----------------------------------------------------|
| Username        | Encoded user identity (converted to numeric)       |
| Time of Day     | Time category (Morning, Afternoon, Evening, Night) |
| Location        | Encoded location of login                          |
| NumAttempts     | Number of consecutive login attempts               |
| Device          | Encoded device type used to log in                 |

---

## Machine Learning Approach

- **Type**: Supervised Learning – Binary Classification
- **Model Used**: `RandomForestClassifier` from scikit-learn
- **Reason**: Robustness, interpretability, and ability to handle categorical + numeric data

The model was trained using both original data and **AI-generated synthetic samples** created via prompt-based generation to simulate real-world login scenarios.

---

## Tools and Technologies

| Tool/Library     | Purpose                                              |
|------------------|------------------------------------------------------|
| Python           | Core programming language                            |
| Pandas           | Data analysis and preparation                        |
| Scikit-learn     | Model training, evaluation, and preprocessing        |
| Joblib           | Saving and loading the trained model (`.pkl` file)   |
| Google Colab     | Cloud-based notebook environment for development     |

---

## Files Included in This Repository

| File                     | Description                                      |
|--------------------------|--------------------------------------------------|
| `LoginRiskProject.ipynb` | Complete Colab notebook with full implementation |
| `login_risk_model.pkl`   | Saved trained model for prediction               |
| `synthetic_data.csv`     | AI-generated dataset used for training           |
| `README.md`              | Project overview and usage instructions          |

---

## How to Use the Trained Model

You can use the `.pkl` model file to make predictions on new login attempts:

```python
import joblib

# Load the pre-trained model
model = joblib.load('login_risk_model.pkl')

# Provide a test input [Username, TimeOfDay, Location, NumAttempts, Device]
sample = [[2, 3, 1, 4, 1]]

# Make prediction
prediction = model.predict(sample)

print("Prediction:", "Risky" if prediction[0] == 1 else "Safe")
