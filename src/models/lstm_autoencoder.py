import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam

def create_sequences(X, window_size=30):
    sequences = []
    for i in range(len(X) - window_size):
        sequences.append(X[i:i + window_size])
    return np.array(sequences)

def build_lstm_autoencoder(input_shape):
    inputs = Input(shape=input_shape)

    encoded = LSTM(64, activation='relu')(inputs)
    decoded = RepeatVector(input_shape[0])(encoded)
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
    outputs = TimeDistributed(Dense(input_shape[1]))(decoded)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    return model

def train_lstm_autoencoder(X, epochs=5, batch_size=32):
    X_seq = create_sequences(X)

    model = build_lstm_autoencoder((X_seq.shape[1], X_seq.shape[2]))

    model.fit(
        X_seq, X_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    return model, X_seq

def detect_anomalies(model, X_seq, threshold=None):
    reconstructions = model.predict(X_seq)

    mse = np.mean(np.power(X_seq - reconstructions, 2), axis=(1, 2))

    if threshold is None:
        threshold = np.percentile(mse, 95)

    anomalies = (mse > threshold).astype(int)

    return anomalies, mse