import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
  
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values(by='timestamp')
    df = df.ffill().bfill()
    return df


def scale_features(df: pd.DataFrame, features: list):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    return df, scaler


def get_feature_columns() -> list:
    return [
        'temperature',
        'vibration',
        'pressure',
        'rotation_speed',
        'power_consumption'
    ]