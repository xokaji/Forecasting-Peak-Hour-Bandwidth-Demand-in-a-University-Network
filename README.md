# 🎓 Forecasting Peak-Hour Bandwidth Demand in a University Network

## 📌 Project Overview
Binary classification ML project that predicts whether a given network flow
occurs during **Peak Hour** or **Non-Peak Hour** in a university network.
Includes a fully interactive web dashboard served through a FastAPI backend.

---

## 📂 Folder Structure
```
university_bandwidth_forecasting/
├── data/
│   ├── raw/                        ← Place downloaded CSV files here
│   └── processed/                  ← Auto-generated cleaned data
├── notebooks/
│   └── 01_bandwidth_peak_forecasting.ipynb   ← Main ML notebook
├── models/                         ← Saved after running notebook
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.json
│   └── model_metadata.json
├── api/
│   ├── main.py                     ← FastAPI backend (REST API + serves UI)
│   └── index.html                  ← Web dashboard UI
├── reports/
│   └── figures/                    ← All plots saved here
├── requirements.txt
└── README.md
```

---

## 📥 Dataset Setup

### Dataset: CICIDS2017 (Canadian Institute for Cybersecurity)
**URL:** https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset

1. Download the CSV files (e.g. `Monday-WorkingHours.pcap_ISCX.csv`)
2. Place them in `data/raw/`
3. Run the notebook

**Why CICIDS2017?**
- Real network flows captured Mon–Fri, 9AM–5PM (perfect for peak/non-peak labelling)
- 78 features including flow duration, bytes/s, packet rates
- Publicly available, widely cited in research
- CSV format — no PCAP tools needed

---

## ⚙️ Setup & Run

```bash
# 1. Clone / navigate to project
cd university_bandwidth_forecasting

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open and run the notebook (trains + saves the model)
code notebooks/01_bandwidth_peak_forecasting.ipynb
# Run all cells top to bottom

# 4. Launch the API + Web UI
uvicorn api.main:app --reload --port 8000

# 5. Open in browser
#   Web Dashboard → http://127.0.0.1:8000/ui
#   REST API Docs  → http://127.0.0.1:8000/docs
```

---

## 🌐 Web Dashboard

The interactive dashboard at **`/ui`** lets you:
- Enter 16 network flow metrics manually or load a **Quick Demo Scenario**
- Click **Analyse & Predict** to classify the flow in real time
- See the prediction result (Peak / Non-Peak), confidence score, and probability breakdown
- View model performance metrics (Accuracy, F1, Precision, Recall) pulled live from the API

The dashboard is fully **mobile responsive** — it adapts from wide desktop (4-column form) down to tablet (2-column) and mobile (1-column) layouts.

---

## 🔌 REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`         | Health check — returns API status + model info |
| `GET`  | `/ui`       | Serves the web dashboard |
| `GET`  | `/model-info` | Returns model metadata (accuracy, F1, etc.) |
| `POST` | `/predict`  | Single-flow prediction |
| `POST` | `/predict-batch` | Batch predictions for multiple flows |
| `GET`  | `/docs`     | Interactive Swagger UI |

---

## 🧠 ML Pipeline

| Step | Description |
|------|-------------|
| Data Loading | Auto-detects CSV files in `data/raw/`, samples 25K rows |
| Data Cleaning | Remove duplicates, handle inf/NaN, fix negatives |
| Feature Engineering | Log-transform skewed features, engineer target label |
| EDA | Distributions, boxplots, port analysis |
| Correlation Heatmap | Pearson correlation matrix |
| Train/Test Split | 80/20 stratified split |
| Feature Scaling | StandardScaler (fit on train only) |
| 5 Models | Logistic Regression, Decision Tree, Random Forest, GBM, XGBoost |
| Evaluation | Accuracy, Precision, Recall, F1 + Confusion Matrix |
| Model Comparison | F1 ranking + heatmap dashboard |
| Feature Importance | Top features from tree-based models |
| Best Model | Highest F1 selected and saved automatically |
| FastAPI | Saved model + scaler deployed via REST API with CORS support |

---

## 🎯 Features Used

| Feature | Reason |
|---------|--------|
| Flow Duration | Long flows = sustained demand |
| Total Fwd/Bwd Packets | Traffic volume |
| Total Length Fwd/Bwd Packets | Byte volume = bandwidth proxy |
| Flow Bytes/s | Throughput rate |
| Flow Packets/s | Congestion signal |
| Flow IAT Mean/Std | Burstiness pattern |
| Fwd/Bwd Packet Length Mean | Application type inference |
| Fwd IAT Total/Mean | Upload pattern |
| Active Mean / Idle Mean | Session pattern |
| Destination Port | Application identification |

---

## 📊 Expected Results
- Best model: **XGBoost** or **Random Forest** (typically >95% F1 on CICIDS2017)
- Baseline: Logistic Regression (~75–80% F1)

---

## 📱 Browser Compatibility
The web dashboard is tested and works on:
- Chrome / Edge / Firefox (desktop & mobile)
- Safari (iOS & macOS)
- Any screen size from 320px wide upwards
