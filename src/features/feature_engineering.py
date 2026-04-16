import pandas as pd


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    return df


def create_rolling_features(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    df['temp_rolling_mean'] = df['temperature'].rolling(window=window).mean()
    df['vibration_rolling_mean'] = df['vibration'].rolling(window=window).mean()

    df = df.bfill()

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df['temp_vibration_ratio'] = df['temperature'] / (df['vibration'] + 1)
    df['power_per_speed'] = df['power_consumption'] / (df['rotation_speed'] + 1)
    return df


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = create_time_features(df)
    df = create_rolling_features(df)
    df = create_interaction_features(df)
    return df


def get_engineered_features() -> list:
    return [
        'temperature',
        'vibration',
        'pressure',
        'rotation_speed',
        'power_consumption',
        'temp_rolling_mean',
        'vibration_rolling_mean',
        'temp_vibration_ratio',
        'power_per_speed'
    ]