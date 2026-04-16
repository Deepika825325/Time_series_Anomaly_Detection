import numpy as np
import pandas as pd


def predict_isolation_forest(model, X: pd.DataFrame):
    
    preds = model.predict(X)

    preds = np.where(preds == 1, 0, 1)

    return preds


def add_predictions_to_df(df: pd.DataFrame, preds, column_name="predicted_anomaly"):
    df[column_name] = preds
    return df


def predict_pipeline(model, X: pd.DataFrame, df: pd.DataFrame):
    preds = predict_isolation_forest(model, X)

    df = add_predictions_to_df(df, preds)

    return df, preds