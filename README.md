# 🎓 Forecasting Peak-Hour Bandwidth Demand in a University Network

## 📌 Project Overview
Binary classification ML project to predict whether a given network flow
occurs during **Peak Hour** or **Non-Peak Hour** in a university network.

---

## 📂 Folder Structure
```
university_bandwidth_forecasting/
├── data/
│   ├── raw/                        ← Place downloaded CSV here
│   └── processed/                  ← Auto-generated cleaned data
├── notebooks/
│   └── bandwidth_peak_forecasting.ipynb   ← Main ML notebook
├── models/                         ← Saved after running notebook
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.json
│   └── model_metadata.json
├── api/
│   └── main.py                     ← FastAPI deployment
├── reports/
│   └── figures/                    ← All plots saved here
├── requirements.txt
└── README.md
```

---

## 📥 Dataset Setup

### Dataset: CICIDS2017 (Canadian Institute for Cybersecurity)
**URL:** https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset

1. Download `Monday-WorkingHours.pcap_ISCX.csv`
2. Place it in `data/raw/`
3. Run the notebook

**Why CICIDS2017?**
- Real network flows captured Mon–Fri, 9AM–5PM (perfect for peak/non-peak)
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

# 3. Open notebook
code notebooks/bandwidth_peak_forecasting.ipynb

# 4. Run all cells top to bottom

# 5. Launch API (after notebook completes)
uvicorn api.main:app --reload
# Docs at: http://127.0.0.1:8000/docs
```

---

## 🧠 ML Pipeline

| Step | Description |
|------|-------------|
| Data Loading | Auto-detects CSV in data/raw/, samples 25K rows |
| Data Cleaning | Remove duplicates, handle inf/NaN, fix negatives |
| Feature Engineering | Log-transform skewed features, engineer target |
| EDA | Distributions, boxplots, port analysis |
| Correlation Heatmap | Pearson correlation matrix |
| Train/Test Split | 80/20 stratified split |
| Feature Scaling | StandardScaler (fit on train only) |
| 5 Models | LR, Decision Tree, RF, GBM, XGBoost |
| Evaluation | Accuracy, Precision, Recall, F1 + Confusion Matrix |
| Model Comparison | F1 ranking + heatmap dashboard |
| Feature Importance | Top features from tree-based models |
| Best Model | Highest F1 selected automatically |
| FastAPI | Saved model + scaler deployed via REST API |

---

## 🎯 Features Used

| Feature | Reason |
|---------|--------|
| Flow Duration | Long flows = sustained demand |
| Total Fwd/Bwd Packets | Traffic volume |
| Total Length Fwd/Bwd Packets | Byte volume = BW proxy |
| Flow Bytes/s | Throughput rate |
| Flow Packets/s | Congestion signal |
| Flow IAT Mean/Std | Burstiness pattern |
| Fwd/Bwd Packet Length Mean | Application type |
| Fwd IAT Total/Mean | Upload pattern |
| Active Mean / Idle Mean | Session pattern |
| Destination Port | App identification |

---

## 📊 Expected Results
- Best model: **XGBoost** or **Random Forest** (typically >92% F1)
- Baseline: Logistic Regression (~75-80% F1)
