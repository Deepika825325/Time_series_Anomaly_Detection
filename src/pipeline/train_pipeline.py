import pandas as pd
import numpy as np
import os
import json

from src.utils.config import load_config
from src.utils.logger import get_logger

from src.data.preprocess import preprocess, scale_features
from src.features.feature_engineering import create_all_features, get_engineered_features

from src.models.train_model import train_and_save
from src.models.predict import predict_isolation_forest
from src.models.evaluate import evaluate_and_print
from src.models.zscore import zscore_anomaly

from src.models.lstm_autoencoder import (
    train_lstm_autoencoder,
    detect_anomalies
)


def run_training_pipeline():

    config = load_config()
    logger = get_logger()

    logger.info(" Starting training pipeline...")

    data_path = config['data']['path']
    df = pd.read_csv(data_path)
    logger.info(f" Data loaded: {df.shape}")

    df = preprocess(df)
    logger.info(" Preprocessing done")

    df = create_all_features(df)
    logger.info(" Feature engineering done")

    features = get_engineered_features()
    X = df[features]

    X, scaler = scale_features(X, features)
    logger.info(" Scaling done")

    X_np = X.values
    
    logger.info(" Training Isolation Forest...")
    model = train_and_save(X)

    y_pred_if = predict_isolation_forest(model, X)
    y_true = df['anomaly']

    logger.info(" Evaluating Isolation Forest...")
    results_if = evaluate_and_print(y_true, y_pred_if)


    logger.info(" Evaluating Z-score...")
    y_pred_z = zscore_anomaly(df, feature='temperature')
    results_z = evaluate_and_print(y_true, y_pred_z)

    logger.info(" Training LSTM Autoencoder...")

    epochs = config['lstm']['epochs']
    batch_size = config['lstm']['batch_size']

    lstm_model, X_seq = train_lstm_autoencoder(
        X_np,
        epochs=epochs,
        batch_size=batch_size
    )

    y_pred_lstm, errors = detect_anomalies(lstm_model, X_seq)

    # Align labels
    y_true_lstm = df['anomaly'].iloc[len(df) - len(y_pred_lstm):]

    logger.info(" Evaluating LSTM Autoencoder...")
    results_lstm = evaluate_and_print(y_true_lstm, y_pred_lstm)

    df['if_anomaly'] = y_pred_if
    df['zscore_anomaly'] = y_pred_z

    lstm_full = np.zeros(len(df))
    lstm_full[-len(y_pred_lstm):] = y_pred_lstm
    df['lstm_anomaly'] = lstm_full

    metrics_path = config['output']['metrics_path']
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    results = {
        "isolation_forest": results_if,
        "zscore": results_z,
        "lstm": results_lstm
    }

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f" Metrics saved to {metrics_path}")

    final_data_path = "outputs/final_data.csv"
    os.makedirs("outputs", exist_ok=True)

    df.to_csv(final_data_path, index=False)

    logger.info(f" Final data saved to {final_data_path}")

    logger.info("Pipeline completed successfully!")

    return {
        "isolation_forest": results_if,
        "zscore": results_z,
        "lstm": results_lstm,
        "data": df
    }