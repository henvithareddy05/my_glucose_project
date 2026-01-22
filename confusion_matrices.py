# save as confusion_matrices.py
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv(r"C:\Users\henvitha\Desktop\my_glucose_project\data\preprocessed.csv")
X = df[['age', 'glucose_level_mgdl', 'diastolic', 'systolic', 'heart_rate', 'body_temp', 'spo2', 'sweating', 'shivering']].fillna(0)
y = np.where(df['label'].fillna(0) == 0, 0, 1)  # Binary

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load YOUR trained models
rf = joblib.load(r"C:\Users\henvitha\Desktop\my_glucose_project\models\rf.pkl")
xg = joblib.load(r"C:\Users\henvitha\Desktop\my_glucose_project\models\xg.pkl")
svm = joblib.load(r"C:\Users\henvitha\Desktop\my_glucose_project\models\svm.pkl")

rf_pred = rf.predict(X_test)
xg_pred = xg.predict(X_test)
svm_pred = svm.predict(X_test)

print("✅ YOUR Confusion Matrices:")
print("RF:\n", confusion_matrix(y_test, rf_pred))
print("XGBoost:\n", confusion_matrix(y_test, xg_pred))
print("SVM:\n", confusion_matrix(y_test, svm_pred))
