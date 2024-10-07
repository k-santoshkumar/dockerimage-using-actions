from fastapi import FastAPI #type:ignore
from pydantic import BaseModel
import joblib #type:ignore
import pandas as pd
import os

app = FastAPI()

# Load artifacts
model = joblib.load('artifacts/models/bestmodel.pkl')
scaler = joblib.load('artifacts/preprocessors/scaler.joblib')
label_encoder = joblib.load('artifacts/preprocessors/label_encoder.joblib')

class CustomerData(BaseModel):
    # Define your input data schema here
    # For example:
    age: int
    salary: float
    # Add all necessary fields

@app.post("/predict")
def predict_churn(data: CustomerData):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Preprocess input data
    # Apply label encoding and scaling
    for column in input_df.select_dtypes(include=['object']).columns:
        input_df[column] = label_encoder.transform(input_df[column])

    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Return result
    result = "Churn" if prediction[0] == 1 else "No Churn"
    return {"prediction": result}
