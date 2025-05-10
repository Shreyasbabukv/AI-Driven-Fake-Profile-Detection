import numpy as np
import joblib
import warnings
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, precision_recall_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), '../../models/fake_profile_model.pkl')

def train_random_forest(X_train, y_train, config):
    params = config['model']['random_forest']
    model = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=config['training']['random_state'])
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, config):
    params = config['model']['xgboost']
    model = XGBClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], learning_rate=params['learning_rate'], use_label_encoder=False, eval_metric='logloss', random_state=config['training']['random_state'])
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train, config):
    params = config['model']['lightgbm']
    model = LGBMClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], learning_rate=params['learning_rate'], random_state=config['training']['random_state'])
    model.fit(X_train, y_train)
    return model

def train_ann(X_train, y_train, config):
    params = config['model']['ann']
    model = Sequential()
    model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
    return model

def train_and_evaluate_models(df, features, labels, config):
    X = features.values if hasattr(features, 'values') else features
    y = labels.values if hasattr(labels, 'values') else labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['training']['test_size'], random_state=config['training']['random_state'], stratify=y)

    rf_model = train_random_forest(X_train, y_train, config)
    xgb_model = train_xgboost(X_train, y_train, config)
    lgbm_model = train_lightgbm(X_train, y_train, config)
    ann_model = train_ann(X_train, y_train, config)

    joblib.dump(xgb_model, MODEL_SAVE_PATH)

    models = {
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
        "LightGBM": lgbm_model,
        "ANN": ann_model
    }

    results = {}
    for name, model in models.items():
        if name == "ANN":
            y_pred_prob = model.predict(X_test).ravel()
            y_pred = (y_pred_prob > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)[:,1]

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)

        results[name] = {
            "accuracy": acc,
            "confusion_matrix": cm,
            "roc_curve": (fpr, tpr),
            "precision_recall_curve": (precision, recall)
        }

    return results
