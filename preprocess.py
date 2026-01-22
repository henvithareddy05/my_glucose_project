import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import joblib  # 👈 ADDED

EXCEL_PATH = r"C:\Users\henvitha\Desktop\my_glucose_project\glucose_dataset.xlsx"
PREPROCESSED_CSV = r"C:\Users\henvitha\Desktop\my_glucose_project\data\preprocessed.csv"
SCALER_FILE = r"C:\Users\henvitha\Desktop\my_glucose_project\models\scaler.pkl"
os.makedirs(os.path.dirname(PREPROCESSED_CSV), exist_ok=True)
os.makedirs(os.path.dirname(SCALER_FILE), exist_ok=True)

def preprocess_excel():
    # Read RAW Excel data
    df_raw = pd.read_excel(EXCEL_PATH, header=None)
    print("Raw shape:", df_raw.shape)
    
    # Data starts from row 2 (skip title + header)
    data_start_row = 2
    data = df_raw.iloc[data_start_row:].reset_index(drop=True)
    
    # Create DataFrame with CORRECT column extraction
    df = pd.DataFrame({
        'age': pd.to_numeric(data.iloc[:, 0], errors='coerce'),
        'blood_glucose': pd.to_numeric(data.iloc[:, 1], errors='coerce'),
        'diastolic': pd.to_numeric(data.iloc[:, 2], errors='coerce'),
        'systolic': pd.to_numeric(data.iloc[:, 3], errors='coerce'),
        'heart_rate': pd.to_numeric(data.iloc[:, 4], errors='coerce'),
        'body_temp': pd.to_numeric(data.iloc[:, 5], errors='coerce'),
        'spo2': pd.to_numeric(data.iloc[:, 6], errors='coerce'),
        'sweating_raw': data.iloc[:, 7].astype(str),
        'shivering_raw': data.iloc[:, 8].astype(str),
        'diabetic_raw': data.iloc[:, 9].astype(str)
    })
    
    # Convert categorical columns PROPERLY
    df['sweating'] = df['sweating_raw'].str.upper().str.strip().map({'Y': 1, 'N': 0}).fillna(0).astype(int)
    df['shivering'] = df['shivering_raw'].str.upper().str.strip().map({'Y': 1, 'N': 0}).fillna(0).astype(int)
    df['diabetic'] = df['diabetic_raw'].str.upper().str.strip().map({'D': 1, 'N': 0}).fillna(0).astype(int)
    
    # Drop raw columns
    df.drop(['sweating_raw', 'shivering_raw', 'diabetic_raw'], axis=1, inplace=True)
    
    print("✅ Data extracted! Shape:", df.shape)
    
    # Glucose conversion (mmol/L to mg/dL)
    df['glucose_level_mgdl'] = df['blood_glucose'].apply(lambda x: x*18 if pd.notna(x) and x<30 else x)
    
    # Fix age outliers
    median_age = df['age'].median()
    df['age'] = df['age'].apply(lambda x: median_age if pd.isna(x) or x<5 or x>100 else x)
    
    # PERFECT label creation (matches paper exactly)
    def label_glucose(g):
        if pd.isna(g): return np.nan
        if g < 140: return 0    # Normal
        elif 140 <= g <= 199: return 1  # Prediabetic
        else: return 2         # Diabetic
    
    df['label'] = df['glucose_level_mgdl'].apply(label_glucose)
    
    # STRICT data cleaning
    before = len(df)
    required_cols = ['label', 'age', 'glucose_level_mgdl', 'diastolic', 'systolic', 'heart_rate', 'body_temp', 'spo2']
    df.dropna(subset=required_cols, inplace=True)
    print(f"✅ Cleaned: {before} → {len(df)} rows")
    
    # Scale ALL features 👈 SCALER SAVED HERE
    all_features = ['age', 'glucose_level_mgdl', 'diastolic', 'systolic', 
                   'heart_rate', 'body_temp', 'spo2', 'sweating', 'shivering']
    scale_cols = [col for col in all_features if col in df.columns]
    
    scaler = MinMaxScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    
    # 👈 SAVE SCALER FOR API PREDICTION
    joblib.dump(scaler, SCALER_FILE)
    print(f"✅ Scaler saved: {SCALER_FILE}")
    
    # FINAL SAVE
    df.to_csv(PREPROCESSED_CSV, index=False)
    
    print("🎉 PREPROCESSING 100% COMPLETE!")
    print("Final shape:", df.shape)
    print("Class distribution:")
    print(df['label'].value_counts().sort_index())
    print(f"📁 Data: {PREPROCESSED_CSV}")
    
    return df

if __name__ == "__main__":
    preprocess_excel()
