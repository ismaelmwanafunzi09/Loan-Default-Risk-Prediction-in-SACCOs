import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Paths
DATA_PATH = r"C:\Users\User1\Desktop\project iness ok\sacco_loan_dataset.csv"
MODEL_PATH = r"C:\Users\User1\Desktop\project iness ok\model_pipeline.joblib"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Target column
target_col = "loan_default"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found. Available: {df.columns.tolist()}")

# Features & target
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode target if categorical
le_target = None
if y.dtype == "object":
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Separate numeric & categorical columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Pipeline with RandomForest
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Train model
pipeline.fit(X_train, y_train)

# Save model bundle
model_bundle = {
    "pipeline": pipeline,
    "label_encoder_target": le_target,
    "feature_columns": X.columns.tolist(),
    "target_column": target_col
}
joblib.dump(model_bundle, MODEL_PATH)

print(f"✅ Model trained and saved at: {MODEL_PATH}")
