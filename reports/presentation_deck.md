# Presentation Deck: Forecasting Peak-Hour Bandwidth Demand in a University Network

## Slide 1 - Introduction (Problem Context)
**Title:** Why Peak-Hour Forecasting Matters in University Networks  

**Content:**
- University networks face time-varying traffic demand from lectures, LMS usage, cloud tools, and streaming.
- Peak periods cause congestion, packet delays, and poor user experience if not managed proactively.
- Traditional reactive monitoring identifies issues after degradation has already occurred.
- A predictive ML approach enables earlier intervention and better bandwidth planning.

**Presenter line:**  
"The core challenge is not just detecting congestion, but predicting demand conditions early enough to act."

---

## Slide 2 - Introduction (Project Objective and Scope)
**Title:** Project Aim and Deliverables  

**Content:**
- Aim: Classify network flows into **Peak Hour** vs **Non-Peak Hour** states.
- Scope: End-to-end ML lifecycle from data ingestion to deployable API + dashboard.
- Core tasks:
- Data preparation and feature engineering.
- Benchmarking multiple classifiers.
- Selecting the best model using F1-driven evaluation.
- Deploying inference via FastAPI endpoints.

**Presenter line:**  
"This is not just a notebook experiment; it is a complete, deployable ML system."

---

## Slide 3 - Dataset (Source and Coverage)
**Title:** Dataset Overview - CICIDS2017  

**Content:**
- Dataset used: CICIDS2017 flow-level network telemetry.
- Total rows across selected files: **2,830,743**.
- Proportional sampling target: **25,000** rows.
- Final clean dataset: **24,133** rows.
- Source coverage: 8 weekday working-hours traffic files.

**Suggested visual:**  
Use a simple table or infographics panel with the four key numbers above.

---

## Slide 4 - Dataset (Feature Space and Class Profile)
**Title:** Features and Target Distribution  

**Content:**
- Key features include:
- Throughput metrics (`Flow Bytes/s`, `Flow Packets/s`).
- Temporal dynamics (`Flow IAT Mean/Std`, `Fwd IAT Total/Mean`).
- Packet-size behavior (`Fwd/Bwd Packet Length Mean`).
- Service indicators (`Destination Port`).
- Class distribution after preprocessing:
- Non-Peak: **14,313 (59.3%)**
- Peak: **9,820 (40.7%)**

**Suggested visual:**  
Insert [01_class_distribution.png](/e:/KDU%20INTAKE%2040/Semester%207/ML/university_bandwidth_forecasting/reports/figures/01_class_distribution.png).

---

## Slide 5 - Methodology (Data Pipeline)
**Title:** Data Preprocessing Pipeline  

**Content:**
- Duplicate rows removed: **849**.
- Infinite values replaced with NaN: **24**.
- NaN rows removed: **14**.
- Negative-value filtering applied to physically non-negative metrics.
- Log transforms (`log1p`) added for skewed variables.
- Stratified split:
- Train: **19,306** (80%)
- Test: **4,827** (20%)

**Presenter line:**  
"Preprocessing quality was critical to achieve stable, high-performing models."

---

## Slide 6 - Methodology (Modeling Strategy)
**Title:** Model Training and Selection Strategy  

**Content:**
- Benchmarked models:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- Evaluation metrics:
- Accuracy, Precision, Recall, F1
- Confusion matrix-based error analysis
- Best model selected by highest F1 and exported for deployment.

**Suggested visual:**  
Insert [08_model_comparison.png](/e:/KDU%20INTAKE%2040/Semester%207/ML/university_bandwidth_forecasting/reports/figures/08_model_comparison.png).

---

## Slide 7 - Key Results (Performance Summary)
**Title:** Benchmark Results and Best Model  

**Content:**
- Best model: **XGBoost**
- Accuracy: **0.9940**
- Precision: **0.9899**
- Recall: **0.9954**
- F1 Score: **0.9926**
- Tree-based ensembles consistently outperformed linear baseline.

**Suggested visual:**  
Insert [06_metrics_comparison.png](/e:/KDU%20INTAKE%2040/Semester%207/ML/university_bandwidth_forecasting/reports/figures/06_metrics_comparison.png).

---

## Slide 8 - Key Results (Interpretation and Operational Value)
**Title:** What the Results Mean for Network Operations  

**Content:**
- Low misclassification indicates reliable detection of demand state transitions.
- Most influential predictors:
- `Flow IAT Mean`
- `Bwd Packet Length Mean`
- `Flow Bytes/s`
- `Flow Packets/s`
- Operational impact:
- Better proactive bandwidth allocation.
- Improved QoS planning during high-demand intervals.
- Practical usage already enabled via FastAPI + dashboard interface.

**Suggested visuals:**
- [07_confusion_matrices.png](/e:/KDU%20INTAKE%2040/Semester%207/ML/university_bandwidth_forecasting/reports/figures/07_confusion_matrices.png)
- [09_feature_importance.png](/e:/KDU%20INTAKE%2040/Semester%207/ML/university_bandwidth_forecasting/reports/figures/09_feature_importance.png)

---

## Slide 9 - Future Work (Methodological Enhancements)
**Title:** Future Work - Modeling Improvements  

**Content:**
- Move fully to **timestamp-based peak-hour labeling** for stronger academic validity.
- Add k-fold and temporal cross-validation for robust generalization checks.
- Evaluate calibration quality for reliable probability interpretation.
- Explore sequence-aware architectures:
- LSTM/GRU
- Temporal CNN
- Transformer-based models

**Presenter line:**  
"The next phase is to increase temporal robustness, not just static test-set accuracy."

---

## Slide 10 - Future Work (Deployment and Expansion)
**Title:** Future Work - Productization and Scale  

**Content:**
- Add multilingual dashboard support (English/Sinhala/Tamil).
- Expand dataset across semesters, exam windows, and special events.
- Introduce drift monitoring with periodic retraining.
- Extend to multi-campus traffic for stronger external validity.
- Build model governance artifacts:
- Model card
- Retraining policy
- Alert threshold playbook

**Closing line:**  
"This project is a strong production-ready baseline and a clear foundation for a campus-scale forecasting platform."

---

## Optional Backup Slide (If Needed in Q&A)
**Title:** Current Limitations  

**Content:**
- Single hold-out split currently used for headline metrics.
- External validation across unseen institutional settings is pending.
- Forecast horizon and latency-aware evaluation can be expanded.

