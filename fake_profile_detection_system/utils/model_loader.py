import joblib
from tensorflow.keras.models import load_model
import os

def load_sklearn_model(model_path):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

def load_ann_model(model_path):
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

def predict_sklearn(model, X):
    return model.predict_proba(X)[:, 1]

def predict_ann(model, X):
    y_pred_prob = model.predict(X).ravel()
    return y_pred_prob
