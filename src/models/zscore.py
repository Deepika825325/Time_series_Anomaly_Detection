import numpy as np
import pandas as pd


def zscore_anomaly(df: pd.DataFrame, feature: str, threshold: float = 3):
    mean = df[feature].mean()
    std = df[feature].std()

    z_scores = (df[feature] - mean) / std

    anomalies = np.where(np.abs(z_scores) > threshold, 1, 0)

    return anomalies