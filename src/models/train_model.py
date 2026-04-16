import os
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest


def train_isolation_forest(X: pd.DataFrame, contamination: float = 0.05):
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X)

    return model


def train_model(X: pd.DataFrame, model_type: str = "isolation_forest"):
    if model_type == "isolation_forest":
        return train_isolation_forest(X)

    else:
        raise ValueError(f"Model type '{model_type}' not supported")


def save_model(model, path: str = "outputs/models/model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f" Model saved at: {path}")


def train_and_save(X: pd.DataFrame, model_type: str = "isolation_forest"):
    print(f"Training {model_type}...")

    model = train_model(X, model_type=model_type)

    save_model(model)

    print(" Training completed!")

    return model