from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = FastAPI(title="Non-Invasive Glucose Monitoring API")

PREPROCESSED_CSV = r"C:\Users\henvitha\Desktop\my_glucose_project\data\preprocessed.csv"
MODEL_FOLDER = r"C:\Users\henvitha\Desktop\my_glucose_project\models"
SCALER_FILE = r"C:\Users\henvitha\Desktop\my_glucose_project\models\scaler.pkl"

# Global models (loaded once)
df = None
scaler = None
model = None

class PredictionInput(BaseModel):
    age: float = 30
    glucose_level_mgdl: float = 100
    diastolic: float = 80
    systolic: float = 120
    heart_rate: float = 75
    body_temp: float = 98.6
    spo2: float = 98
    sweating: int = 0
    shivering: int = 0

@app.get("/")
def root():
    return {"status": "Non-Invasive Glucose Monitoring API ✅", "docs": "/docs"}

@app.post("/preprocess")
def preprocess(background_tasks: BackgroundTasks):
    background_tasks.add_task(lambda: print("Run: python preprocess.py"))
    return {"status": "Run python preprocess.py first"}

@app.get("/preview")
def preview(n: int = 5):
    global df
    try:
        if df is None:
            df = pd.read_csv(PREPROCESSED_CSV)
        return df.head(n).to_dict(orient="records")
    except:
        return {"error": "Run python preprocess.py first"}

@app.post("/train")
def train(background_tasks: BackgroundTasks):
    background_tasks.add_task(lambda: print("Run: python train.py"))
    return {"status": "Run python train.py first"}

@app.post("/predict")
def predict(data: PredictionInput):
    global scaler, model
    
    try:
        # Load scaler and model if not loaded
        if scaler is None:
            scaler = joblib.load(SCALER_FILE)
        if model is None:
            model = load_model(os.path.join(MODEL_FOLDER, "ffnn.h5"))
        
        # Prepare features in EXACT training order
        features = np.array([[
            data.age, data.glucose_level_mgdl, data.diastolic, data.systolic,
            data.heart_rate, data.body_temp, data.spo2, data.sweating, data.shivering
        ]])
        
        # 👈 CRITICAL: Scale input exactly like training data
        features_scaled = scaler.transform(features)
        
        # Predict (binary: 0=Normal, 1=Diabetic)
        prob = model.predict(features_scaled, verbose=0)[0][0]
        pred_class = 1 if prob >= 0.5 else 0
        
        labels = {0: "Normal", 1: "Diabetic"}
        confidence = float(max(prob, 1-prob))
        
        return {
            "prediction": labels[pred_class],
            "confidence": confidence,
            "probability_diabetic": float(prob),
            "probability_normal": float(1-prob)
        }
        
    except Exception as e:
        return {"error": str(e), "message": "Ensure models/scaler.pkl exists"}

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": model is not None}
