import streamlit as st
import pandas as pd
import joblib
import numpy as np

MODEL_PATH = "model_pipeline.joblib"

@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH)

st.title("Loan Default Prediction App")
st.markdown("This app predicts whether a loan will **default (1)** or **not default (0)** based on applicant features.")

# Load trained model
data = load_artifacts()
pipeline = data["pipeline"]
le_target = data.get("label_encoder_target", None)
feature_columns = data.get("feature_columns", [])
target_column = data.get("target_column", "loan_default")

# --- Option 1: Predict from CSV ---
st.header("Option 1 — Predict from CSV")
uploaded = st.file_uploader(
    "Upload a CSV with the same feature columns (without the target column).",
    type=["csv"]
)

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("Uploaded data preview:")
    st.dataframe(df.head())

    X = df[feature_columns]
    preds = pipeline.predict(X)

    if le_target is not None:
        try:
            preds = le_target.inverse_transform(preds)
        except Exception:
            pass

    df["prediction"] = preds
    st.write("Predictions:")
    st.dataframe(df.head())

# --- Option 2: Single sample prediction ---
st.header("Option 2 — Single sample prediction")
st.markdown("Fill in the values below. Leave blank numeric fields to use imputation.")

sample = {}
for col in feature_columns:
    val = st.text_input(f"Enter value for {col}", key=col)
    try:
        sample[col] = float(val) if val != "" else np.nan
    except:
        sample[col] = val if val != "" else None

if st.button("Predict single sample"):
    df_sample = pd.DataFrame([sample])
    try:
        pred = pipeline.predict(df_sample)
        if le_target is not None:
            try:
                pred = le_target.inverse_transform(pred)
            except Exception:
                pass
        st.success(f"Predicted Loan Default: {pred[0]} (1 = Default, 0 = No Default)")
    except Exception as e:
        st.error("Prediction failed: " + str(e))
