# Signal Guard: Time_series_Anomaly_Detection
Time series anomaly detection identifies unusual patterns or deviations in sequential data, enabling early detection of faults in systems like industrial sensors to prevent failures and reduce downtime.
A production-grade machine learning system for detecting anomalies in industrial sensor data using statistical, machine learning, and deep learning approaches with an interactive Streamlit dashboard for visualization.

# 📌 Overview

SignalGuard is designed to monitor time-series sensor data (temperature, vibration, pressure, etc.) and detect anomalies that may indicate equipment failures or abnormal system behavior.

The system integrates:

📊 Statistical methods (Z-score)
🤖 Machine Learning (Isolation Forest)
🧠 Deep Learning (LSTM Autoencoder)

It provides a complete ML pipeline + dashboard, making it suitable for real-world monitoring systems.

# 🎯 Key Features

✔ End-to-end ML pipeline
✔ Multi-model anomaly detection
✔ Feature engineering for time-series data
✔ Model evaluation (Accuracy, Precision, Recall, F1)
✔ Config-driven architecture
✔ Logging system
✔ Automated testing with pytest
✔ Interactive dashboard (Streamlit)
✔ Production-ready structure

# 🧠 Models Used
1. 📊 Z-score (Statistical)
Detects anomalies based on deviation from mean
Simple and fast baseline model
2. 🌲 Isolation Forest (Machine Learning)
Detects anomalies using tree-based isolation
Best performance in this project
3. 🔁 LSTM Autoencoder (Deep Learning)
Learns temporal patterns in time-series
Detects anomalies via reconstruction error
# 📊 Model Performance
Model	Accuracy	Precision	Recall	F1 Score
Isolation Forest	~0.98	~0.64	~0.95	~0.77
Z-score	~0.97	~0.94	~0.40	~0.56
LSTM Autoencoder	~0.96	~0.47	~0.69	~0.56
🏗️ Project Structure
Time_series_Anomaly_Detection/
│
├── app/
│   └── streamlit_app.py        # Dashboard UI
│
├── config/
│   └── config.yaml             # Config file
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│   ├── models/
│   ├── metrics/
│   ├── plots/
│   └── final_data.csv
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── pipeline/
│   └── utils/
│
├── tests/
│   └── test_pipeline.py
│
├── main.py
├── requirements.txt
└── README.md
# ⚙️ Installation
1. Clone repository
git clone https://github.com/your-username/signalguard.git
cd signalguard
2. Create virtual environment
python -m venv venv
venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
    Run the Project
🔹 Train models & generate outputs
python main.py

This will:

Train models
Save metrics
Generate predictions
Create outputs/final_data.csv
🔹 Run Dashboard
streamlit run app/streamlit_app.py

👉 Open browser:

http://localhost:8501
📊 Dashboard Features
📈 Time-series anomaly visualization
🔍 Model selection (IF, Z-score, LSTM)
📊 Performance comparison
🚨 Real-time anomaly highlighting
⚙️ System status panel
🧪 Testing

Run pipeline test:

python -m pytest tests/

✔ Ensures:

Pipeline runs successfully
Outputs are valid
Models are working
# ⚙️ Configuration

All configurations are stored in:

config/config.yaml

Example:

data:
  path: data/raw/sensor_dataset.csv

lstm:
  epochs: 5
  batch_size: 32

output:
  metrics_path: outputs/metrics/results.json

This project demonstrates:

End-to-end ML pipeline design
Time-series analysis
Model comparison (Stat + ML + DL)
Production-ready architecture
Dashboard development

# 🔮 Future Improvements
Real-time streaming data
Alert system (email/SMS)
Model retraining pipeline
API deployment (FastAPI)
Docker containerization

# 👨‍💻 Author
Deepika Kumari