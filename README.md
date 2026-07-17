<div align="center">

# 🛰️ SignalGuard — Time Series Anomaly Detection

**A production-grade ML system for detecting anomalies in industrial sensor data**
*Statistical • Machine Learning • Deep Learning — unified in one pipeline with an interactive dashboard*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/Deepika825325/Time_series_Anomaly_Detection?style=social)](https://github.com/Deepika825325/Time_series_Anomaly_Detection)

[Overview](#-overview) • [Demo](#-demo) • [Features](#-key-features) • [Models](#-models-used) • [Installation](#️-installation) • [Usage](#️-usage) • [Results](#-model-performance) • [Roadmap](#-future-improvements) • [Author](#-author--contact)

</div>

---

## 📌 Overview

**SignalGuard** monitors time-series sensor data — temperature, vibration, pressure, and similar industrial signals — and flags anomalies that may indicate equipment faults or abnormal system behavior before they cause failures or downtime.

Rather than relying on a single detection method, the project combines three complementary approaches into one end-to-end pipeline:

- 📊 **Statistical** — Z-score deviation analysis
- 🤖 **Machine Learning** — Isolation Forest
- 🧠 **Deep Learning** — LSTM Autoencoder (reconstruction-error based)

The pipeline is config-driven, logged, tested, and paired with an interactive **Streamlit dashboard** for real-time visualization and model comparison — making it a practical reference for real-world condition-monitoring systems.

---

## 🎥 Demo

A full video walkthrough of the dashboard is available in [`demo.mp4`](https://github.com/Deepika825325/Time_series_Anomaly_Detection/blob/main/demo.mp4) in this repository.

<div align="center">

**Live metrics at a glance**
![SignalGuard AI Dashboard header showing 2160 anomalies detected with accuracy, precision, recall, and F1 metrics](assets/dashboard_header.png)

**Real-time anomaly detection on sensor data**
![Time-series chart with Isolation Forest highlighting anomalies in temperature sensor data](assets/anomaly_chart.png)

**Model comparison — F1 score across all three approaches**
![Bar chart comparing F1 scores of Isolation Forest, Z-score, and LSTM Autoencoder](assets/model_comparison.png)

**Stats, live alerts, and system health**
![Stats panel showing total anomalies and records, live alert feed, and system CPU/memory status](assets/stats_alerts_system.png)

</div>

> 💡 *Tip: GitHub renders `.mp4` files inline — open the link above to watch the full walkthrough in the browser.*

---

## 📈 Exploratory Data Analysis

Before modeling, the raw sensor data was profiled to understand distributions, correlations, and baseline anomaly patterns across all five channels (temperature, vibration, pressure, rotation speed, power consumption).

<div align="center">

**All sensor channels over time**
![Line plots of temperature, vibration, pressure, rotation speed, and power consumption over the full data window](assets/sensor_data_trends.png)

**Feature correlation matrix**
![Heatmap showing correlation coefficients between the five sensor features](assets/correlation_heatmap.png)

**Feature distributions**
![Histograms showing the distribution of each of the five sensor features](assets/feature_distributions.png)

**Detected anomalies in the temperature signal**
![Temperature time series with detected anomaly points highlighted in yellow](assets/anomaly_temperature.png)

</div>

**Key observations:**
- Temperature and vibration show the strongest correlation (**r ≈ 0.61**) among all feature pairs — the two most physically coupled signals in a rotating industrial system
- Most features are approximately normally distributed around a stable operating band, with anomalies appearing as sharp, short-lived spikes rather than gradual drift
- Anomaly spikes are visually distinct and infrequent, confirming the dataset is well-suited for both statistical outlier methods and reconstruction-based deep learning approaches

---

## 🎯 Key Features

| | |
|---|---|
| ✅ **End-to-end ML pipeline** | Ingestion → feature engineering → training → evaluation → serving |
| ✅ **Multi-model anomaly detection** | Compare statistical, ML, and DL models side-by-side |
| ✅ **Time-series feature engineering** | Rolling stats, lag features, and signal transforms |
| ✅ **Rigorous evaluation** | Accuracy, Precision, Recall, F1 across all models |
| ✅ **Config-driven architecture** | All parameters controlled via `config.yaml` — no hardcoding |
| ✅ **Structured logging** | Full pipeline traceability |
| ✅ **Automated testing** | `pytest` suite validates pipeline integrity |
| ✅ **Interactive dashboard** | Real-time anomaly visualization built with Streamlit |
| ✅ **Production-ready structure** | Modular `src/` layout ready to extend or deploy |

---

## 🧠 Models Used

| Model | Type | How it Works |
|---|---|---|
| **Z-score** | Statistical | Flags points deviating significantly from the rolling mean — fast, interpretable baseline |
| **Isolation Forest** | Machine Learning | Isolates anomalies via random tree partitioning — best overall performer in this project |
| **LSTM Autoencoder** | Deep Learning | Learns normal temporal patterns and flags high reconstruction error as anomalous |

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|:---:|:---:|:---:|:---:|
| **Isolation Forest** | ~0.98 | ~0.64 | ~0.95 | **~0.77** |
| Z-score | ~0.97 | ~0.94 | ~0.40 | ~0.57 |
| LSTM Autoencoder | ~0.96 | ~0.47 | ~0.69 | ~0.74 |

> **Isolation Forest** delivers the best overall F1 score, with **LSTM Autoencoder** close behind — making it a strong choice when temporal pattern context matters. **Z-score** remains a fast, interpretable baseline with the highest precision of the three.

---

## 🏗️ Project Structure

```
Time_series_Anomaly_Detection/
│
├── app/
│   └── streamlit_app.py       # Dashboard UI
│
├── config/
│   └── config.yaml            # Central configuration
│
├── data/
│   ├── raw/                   # Raw sensor data
│   └── processed/             # Processed / feature-engineered data
│
├── outputs/
│   ├── models/                # Trained model artifacts
│   ├── metrics/                # Evaluation results (JSON)
│   ├── plots/                  # Generated visualizations
│   └── final_data.csv          # Final scored dataset
│
├── src/
│   ├── data/                   # Data loading & preprocessing
│   ├── features/               # Feature engineering
│   ├── models/                 # Model definitions (Z-score, IF, LSTM)
│   ├── pipeline/                # Pipeline orchestration
│   └── utils/                   # Shared utilities
│
├── tests/
│   └── test_pipeline.py         # Pipeline unit tests
│
├── main.py                      # Entry point — runs full training pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/Deepika825325/Time_series_Anomaly_Detection.git
cd Time_series_Anomaly_Detection
```

**2. Create a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 Train models & generate outputs
```bash
python main.py
```
This will:
- Train all three models (Z-score, Isolation Forest, LSTM Autoencoder)
- Save evaluation metrics to `outputs/metrics/`
- Generate anomaly predictions
- Create `outputs/final_data.csv`

### 🔹 Launch the interactive dashboard
```bash
streamlit run app/streamlit_app.py
```
Then open **[http://localhost:8501](http://localhost:8501)** in your browser.

**Dashboard capabilities:**
- 📈 Time-series anomaly visualization
- 🔍 Model selection (Isolation Forest / Z-score / LSTM)
- 📊 Side-by-side performance comparison
- 🚨 Real-time anomaly highlighting
- ⚙️ System status panel

---

## 🧪 Testing

Validate the pipeline end-to-end:
```bash
python -m pytest tests/
```
This confirms the pipeline runs successfully, outputs are valid, and all models train without errors.

---

## ⚙️ Configuration

All parameters live in `config/config.yaml` — no code changes needed to adjust data paths, hyperparameters, or output locations.

```yaml
data:
  path: data/raw/sensor_dataset.csv

lstm:
  epochs: 5
  batch_size: 32

output:
  metrics_path: outputs/metrics/results.json
```

---

## 🛠️ Tech Stack

`Python` · `Pandas` / `NumPy` · `scikit-learn` · `TensorFlow / Keras` · `Streamlit` · `pytest` · `YAML`

---

## 🔮 Future Improvements

- 🔴 Real-time streaming data ingestion
- 📩 Alert system (email / SMS notifications)
- 🔁 Automated model retraining pipeline
- 🌐 API deployment via FastAPI
- 🐳 Docker containerization for one-command deployment

---

## 👩‍💻 Author & Contact

**Deepika Kumari**
R&D Intern, TiHAN — IIT Hyderabad

[![GitHub](https://img.shields.io/badge/GitHub-Deepika825325-181717?style=flat&logo=github&logoColor=white)](https://github.com/Deepika825325)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-deepikakri-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/deepikakri)

---

<div align="center">

⭐ If you found this project useful, consider giving it a star!

</div>