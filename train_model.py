
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "hotel.csv"
OUTPUT_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)


def train_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "hotel(2).csv not found in the repository root."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    target = "booking_status"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    df = df.dropna(subset=[target])
    X = df.drop(columns=[target])
    y = df[target].astype(str).str.strip()

    # Remove identifier columns that should not be predictive features.
    id_columns = [
        col for col in X.columns
        if col.lower() in ("booking_id", "id", "user_id")
    ]
    X = X.drop(columns=id_columns)

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    labels = sorted(y.unique())

    metrics = {
        "dataset": DATA_PATH.name,
        "records": int(len(df)),
        "features": int(X.shape[1]),
        "target": target,
        "classes": labels,
        "accuracy": float(accuracy),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=labels
        ).tolist()
    }

    joblib.dump(model, OUTPUT_DIR / "hotel_booking_model.pkl")

    with open(OUTPUT_DIR / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print(json.dumps(metrics, indent=4))
    print("Model training completed successfully.")


if __name__ == "__main__":
    train_model()
