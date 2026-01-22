import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

PREPROCESSED_CSV = r"C:\Users\henvitha\Desktop\my_glucose_project\data\preprocessed.csv"
MODEL_FOLDER = r"C:\Users\henvitha\Desktop\my_glucose_project\models"
RESULTS_CSV = r"C:\Users\henvitha\Desktop\my_glucose_project\model_performance.csv"

os.makedirs(MODEL_FOLDER, exist_ok=True)

# Load data
df = pd.read_csv(PREPROCESSED_CSV)
print("Dataset loaded:", df.shape)

feature_cols = ['age', 'glucose_level_mgdl', 'diastolic', 'systolic', 'heart_rate', 'body_temp', 'spo2', 'sweating', 'shivering']
feature_cols = [col for col in feature_cols if col in df.columns]
print("Features:", feature_cols)

X = df[feature_cols].fillna(0).values  # Fill NaNs
y_original = df['label'].fillna(0).values

# Map [0,2] → [0,1] for binary classification
y = np.where(y_original == 0, 0, 1).astype(int)  # 0=Normal, 1=Diabetic

print("Labels: Normal vs Diabetic")
print("Normal count:", np.sum(y == 0))
print("Diabetic count:", np.sum(y == 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

results = []

# 1. Random Forest ⭐ PERFECT
print("\n🌳 Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
joblib.dump(rf, os.path.join(MODEL_FOLDER, "rf.pkl"))
results.append({"Model": "Random Forest", "Accuracy": rf_acc})
print(f"✅ RF: {rf_acc:.4f}")

# 2. XGBoost ⭐ FIXED (Binary classification)
print("\n⚡ XGBoost...")
xg = xgb.XGBClassifier(
    objective='binary:logistic',  # BINARY classification
    random_state=42, 
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)
xg.fit(X_train, y_train)
xg_pred = xg.predict(X_test)
xg_acc = accuracy_score(y_test, xg_pred)
joblib.dump(xg, os.path.join(MODEL_FOLDER, "xg.pkl"))
results.append({"Model": "XGBoost", "Accuracy": xg_acc})
print(f"✅ XGBoost: {xg_acc:.4f}")

# 3. SVM ⭐ PERFECT
print("\n🎯 SVM...")
svm = SVC(kernel='rbf', probability=True, C=10, gamma='scale', random_state=42)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
joblib.dump(svm, os.path.join(MODEL_FOLDER, "svm.pkl"))
results.append({"Model": "SVM", "Accuracy": svm_acc})
print(f"✅ SVM: {svm_acc:.4f}")

# 4. Neural Network ⭐ PERFECT
print("\n🧠 Neural Network...")
model = Sequential([
    Dense(100, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(50, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # BINARY output
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0, validation_split=0.1)
nn_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
nn_acc = accuracy_score(y_test, nn_pred)
model.save(os.path.join(MODEL_FOLDER, "ffnn.h5"))
results.append({"Model": "Neural Network", "Accuracy": nn_acc})
print(f"✅ Neural Network: {nn_acc:.4f}")

# Save results (TABLE 1)
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
results_df.to_csv(RESULTS_CSV, index=False)

print("\n📊 FINAL RESULTS (TABLE 1):")
print(results_df.round(4))
print(f"\n🎉 ✅ PROJECT 100% COMPLETE!")
print(f"Models saved: {MODEL_FOLDER}")
print(f"Results saved: {RESULTS_CSV}")
