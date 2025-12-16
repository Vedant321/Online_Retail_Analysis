import streamlit as st
import requests
import pandas as pd
from typing import List, Dict


REQUIRED_COLUMNS = [
    "Recency",
    "Frequency",
    "Monetary",
    "AvgUnitPrice",
    "AvgBasketValue",
]


class HighValueCustomerApp:
    def __init__(self, api_url: str):
        self.api_url = api_url

    def call_api(self, payload: Dict) -> float:
        response = requests.post(self.api_url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "high_value_probability" not in data:
            raise KeyError(f"Unexpected Response:{data}")
        
        return data["high_value_probability"]

    def predict(self, payloads: List[Dict]) -> List[float]:
        """
        Common prediction logic for both single and batch inference.
        """
        return [self.call_api(payload) for payload in payloads]

    def render_single_ui(self):
        st.subheader("🔹 Single Customer Prediction")

        values = {
            "Recency": st.number_input("Recency (days)", min_value=0.0, value=10.0),
            "Frequency": st.number_input("Frequency", min_value=0.0, value=5.0),
            "Monetary": st.number_input("Monetary", min_value=0.0, value=200.0),
            "AvgUnitPrice": st.number_input("Average Unit Price", min_value=0.0, value=20.0),
            "AvgBasketValue": st.number_input("Average Basket Value", min_value=0.0, value=50.0),
        }

        if st.button("Predict Single Customer"):
            try:
                prob = self.predict([values])[0]
                label = "High Value Customer" if prob >= 0.5 else "Low Value Customer"

                st.success(f"Probability: **{prob:.2f}**")
                st.markdown(f"### ✅ Prediction: **{label}**")

            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")

    def render_batch_ui(self):
        st.subheader("📁 Batch Prediction via CSV Upload")

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())

                if not all(col in df.columns for col in REQUIRED_COLUMNS):
                    st.error(
                        f"CSV must contain columns: {', '.join(REQUIRED_COLUMNS)}"
                    )
                    return

                if st.button("Run Batch Prediction"):
                    with st.spinner("Running predictions..."):
                        payloads = df[REQUIRED_COLUMNS].to_dict(orient="records")
                        probs = self.predict(payloads)

                        df["HighValueCustomerProbability"] = probs
                        df["PredictedLabel"] = [
                            "High Value" if p >= 0.5 else "Low Value" for p in probs
                        ]

                    st.subheader("Prediction Results")
                    st.dataframe(df)

                    st.download_button(
                        label="Download Predictions CSV",
                        data=df.to_csv(index=False),
                        file_name="customer_predictions.csv",
                        mime="text/csv",
                    )

            except Exception as e:
                st.error(f"Error processing file: {e}")

    def render_ui(self):
        st.set_page_config(
            page_title="High Value Customer Predictor",
            layout="wide"
        )

        st.title("🛍️ High Value Customer Prediction System")
        st.write(
            "This application supports both **single-customer prediction** "
            "and **batch prediction via CSV upload**, using a "
            "cloud-deployed FastAPI inference service."
        )

        tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

        with tab1:
            self.render_single_ui()

        with tab2:
            self.render_batch_ui()


if __name__ == "__main__":
    API_URL = "https://online-retail-api-latest1.onrender.com/predict"
    app = HighValueCustomerApp(API_URL)
    app.render_ui()
