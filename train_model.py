import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


DATA_FILE = 'creditcard.csv'
MODEL_FILE = 'fraud_model.pkl'
SCALER_FILE = 'scaler.pkl'

def train_and_save_model():

    try:
        data = pd.read_csv(DATA_FILE)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Please place the dataset file in the current directory.")
        return

    
    print("Starting data preprocessing...")
    
    
    scaler = RobustScaler()
    data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

    X = data.drop(['Time', 'Class'], axis=1) 
    y = data['Class']

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Applying SMOTE to balance training data...")
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"Original Training Class 1 count: {y_train.sum()}")
    print(f"Resampled Training Class 1 count: {y_train_res.sum()}")

    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train_res, y_train_res)
    print("Model training complete.")

    y_pred = model.predict(X_test)
    
    print("\n--- Model Evaluation on Test Set ---")
    print(f"Total Test Transactions: {len(X_test)}")
    print("Confusion Matrix:")
    
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report (Focus on Recall/F1-score for Class 1 - Fraud):")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"\nTrained model saved as '{MODEL_FILE}'.")
    print(f"Scaler saved as '{SCALER_FILE}'.")

if __name__ == "__main__":
    train_and_save_model()