"""
FastAPI — University Bandwidth Peak Hour Prediction
=====================================================
Run: uvicorn api.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""

import os
import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# ── Load artifacts ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, 'models')

try:
    model    = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
    scaler   = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    with open(os.path.join(MODEL_DIR, 'model_metadata.json')) as f:
        metadata = json.load(f)
    with open(os.path.join(MODEL_DIR, 'feature_names.json')) as f:
        feature_names = json.load(f)
    print(f"✅ Model loaded: {metadata['model_name']} (F1={metadata['f1_score']})")
except Exception as e:
    print(f"❌ Model load error: {e}")
    model = scaler = metadata = feature_names = None

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "🎓 University Bandwidth Peak Hour Predictor",
    description = "Predicts whether current network traffic is in Peak Hour or Non-Peak Hour",
    version     = "1.0.0"
)

# ── Request schema ────────────────────────────────────────────────────────────
class NetworkFlowInput(BaseModel):
    Flow_Duration              : float = Field(..., example=100000.0,  description="Flow duration in microseconds")
    Total_Fwd_Packets          : float = Field(..., example=15.0,      description="Total forward packets")
    Total_Backward_Packets     : float = Field(..., example=10.0,      description="Total backward packets")
    Total_Length_Fwd_Packets   : float = Field(..., example=5000.0,    description="Total bytes in forward packets")
    Total_Length_Bwd_Packets   : float = Field(..., example=3000.0,    description="Total bytes in backward packets")
    Flow_Bytes_per_s           : float = Field(..., example=80000.0,   description="Flow bytes per second")
    Flow_Packets_per_s         : float = Field(..., example=250.0,     description="Flow packets per second")
    Flow_IAT_Mean              : float = Field(..., example=5000.0,    description="Flow inter-arrival time mean (μs)")
    Flow_IAT_Std               : float = Field(..., example=2000.0,    description="Flow IAT standard deviation")
    Fwd_Packet_Length_Mean     : float = Field(..., example=333.0,     description="Forward packet length mean (bytes)")
    Bwd_Packet_Length_Mean     : float = Field(..., example=300.0,     description="Backward packet length mean (bytes)")
    Fwd_IAT_Total              : float = Field(..., example=50000.0,   description="Forward IAT total")
    Fwd_IAT_Mean               : float = Field(..., example=3500.0,    description="Forward IAT mean")
    Active_Mean                : float = Field(..., example=0.0,       description="Active mean")
    Idle_Mean                  : float = Field(..., example=0.0,       description="Idle mean")
    Destination_Port           : float = Field(..., example=443.0,     description="Destination port number")

# ── Response schema ───────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction        : int
    label             : str
    confidence        : float
    prob_non_peak     : float
    prob_peak         : float
    model_used        : str

# ── Helper: map input to feature array ───────────────────────────────────────
FIELD_MAP = {
    'Flow Duration'                  : 'Flow_Duration',
    'Total Fwd Packets'              : 'Total_Fwd_Packets',
    'Total Backward Packets'         : 'Total_Backward_Packets',
    'Total Length of Fwd Packets'    : 'Total_Length_Fwd_Packets',
    'Total Length of Bwd Packets'    : 'Total_Length_Bwd_Packets',
    'Flow Bytes/s'                   : 'Flow_Bytes_per_s',
    'Flow Packets/s'                 : 'Flow_Packets_per_s',
    'Flow IAT Mean'                  : 'Flow_IAT_Mean',
    'Flow IAT Std'                   : 'Flow_IAT_Std',
    'Fwd Packet Length Mean'         : 'Fwd_Packet_Length_Mean',
    'Bwd Packet Length Mean'         : 'Bwd_Packet_Length_Mean',
    'Fwd IAT Total'                  : 'Fwd_IAT_Total',
    'Fwd IAT Mean'                   : 'Fwd_IAT_Mean',
    'Active Mean'                    : 'Active_Mean',
    'Idle Mean'                      : 'Idle_Mean',
    'Destination Port'               : 'Destination_Port',
}

def build_feature_array(data: NetworkFlowInput, feat_names: list) -> np.ndarray:
    """Map input fields to ordered feature array, including log features."""
    input_dict = data.dict()
    row = []
    for feat in feat_names:
        if feat.endswith('_log'):
            # Reconstruct log feature
            orig_feat = feat[:-4]  # strip '_log'
            api_key   = FIELD_MAP.get(orig_feat)
            val       = input_dict.get(api_key, 0.0)
            row.append(float(np.log1p(max(0.0, val))))
        else:
            api_key = FIELD_MAP.get(feat)
            row.append(float(input_dict.get(api_key, 0.0)) if api_key else 0.0)
    return np.array(row).reshape(1, -1)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status"     : "✅ API is running",
        "model"      : metadata.get('model_name') if metadata else "Not loaded",
        "f1_score"   : metadata.get('f1_score') if metadata else None,
        "docs"       : "/docs"
    }

@app.get("/model-info", tags=["Info"])
def model_info():
    if not metadata:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return metadata

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: NetworkFlowInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run the notebook first.")
    
    try:
        features = build_feature_array(data, feature_names)
        
        # Apply scaling if needed
        if metadata.get('scaling_required'):
            features = scaler.transform(features)
        
        pred  = int(model.predict(features)[0])
        proba = model.predict_proba(features)[0]
        
        return PredictionResponse(
            prediction    = pred,
            label         = "🔴 Peak Hour" if pred == 1 else "🟢 Non-Peak Hour",
            confidence    = round(float(max(proba)) * 100, 2),
            prob_non_peak = round(float(proba[0]), 4),
            prob_peak     = round(float(proba[1]), 4),
            model_used    = metadata.get('model_name', 'Unknown')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-batch", tags=["Prediction"])
def predict_batch(data_list: list[NetworkFlowInput]):
    """Batch predictions for multiple flows."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for data in data_list:
        features = build_feature_array(data, feature_names)
        if metadata.get('scaling_required'):
            features = scaler.transform(features)
        pred  = int(model.predict(features)[0])
        proba = model.predict_proba(features)[0]
        results.append({
            "prediction"    : pred,
            "label"         : "Peak Hour" if pred == 1 else "Non-Peak Hour",
            "confidence"    : round(float(max(proba)) * 100, 2)
        })
    return {"total": len(results), "predictions": results}
