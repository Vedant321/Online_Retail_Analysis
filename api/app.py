# import os
# import joblib
# import numpy as np
# from fastapi import FastAPI
# from pydantic import BaseModel

# # Get the path relative to this script
# model_path = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'final_model_pipeline.pkl')
# model_path = os.path.abspath(model_path)  # optional: resolves full path

# print("Loading model from:", model_path)
# model = joblib.load(model_path)


# class CustomerFeatures(BaseModel):
#     Recency: float
#     Frequency: float
#     Monetary: float
#     AvgUnitPrice: float
#     MonetaryValue: float
#     AvgBasketValue: float

# app = FastAPI()

# @app.post("/predict")
# def predict_high_value_customer(data: CustomerFeatures):
#     # Convert input to array for the pipeline
#     features = np.array([[data.Recency, data.Frequency, data.Monetary,
#                           data.AvgUnitPrice, data.MonetaryValue, data.AvgBasketValue]])
#     # Predict probability
#     prob = model.predict_proba(features)[0][1]
#     return {"HighValueCustomerProbability": prob}

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "final_model_pipeline.pkl"
)

model = joblib.load(MODEL_PATH)
app = FastAPI(title="High Value Customer Prediction API")


FEATURE_ORDER = [
    "Recency",
    "Frequency",
    "Monetary",
    "AvgUnitPrice",
    "AvgBasketValue"
]


class CustomerFeatures(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float
    AvgUnitPrice: float
    AvgBasketValue: float



@app.post("/predict")
def predict_high_value_customer(features: CustomerFeatures):

    input_df = pd.DataFrame([[
            features.Recency,
            features.Frequency,
            features.Monetary,
            features.AvgUnitPrice,
            features.AvgBasketValue
        ]], columns=FEATURE_ORDER)

    probability = model.predict_proba(input_df)[0][1]

    return {
        "high_value_probability": float(probability)
    }


@app.get("/")
def root():
    return {"status": "Online Retail API running"}

@app.get("/health")
def health():
    return {"status": "healthy"}
