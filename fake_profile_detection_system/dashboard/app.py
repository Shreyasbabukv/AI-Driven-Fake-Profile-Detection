import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import yaml
import time
import shap
import joblib
import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model_loader import load_sklearn_model, load_ann_model, predict_sklearn, predict_ann
from utils.logger import setup_logger

logger = setup_logger()

# Load config
with open(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml'), "r") as f:
    config = yaml.safe_load(f)
    # Adjust config to include model_paths for compatibility
    config['model_paths'] = {
        'random_forest': 'models/random_forest.pkl',
        'xgboost': 'models/xgboost.pkl',
        'lightgbm': 'models/lightgbm.pkl',
        'ann': 'models/ann.h5'
    }

st.set_page_config(page_title="Fake Profile Detection Dashboard", layout="wide", initial_sidebar_state="expanded")

# Dark theme styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stButton>button {
        background-color: #1f2937;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def plot_accuracy_vs_epochs(epochs, accuracies):
    fig, ax = plt.subplots()
    ax.plot(epochs, accuracies, marker='o')
    ax.set_title("Accuracy vs Epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    st.pyplot(fig)

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def plot_roc_curve(fpr, tpr, key_suffix=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
    fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    # Use unique key by appending id of fig to key_suffix
    st.plotly_chart(fig, key=f"roc_curve_{key_suffix}_{id(fig)}")

def plot_precision_recall_curve(precision, recall, key_suffix=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall Curve'))
    fig.update_layout(title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision')
    # Use unique key by appending id of fig to key_suffix
    st.plotly_chart(fig, key=f"precision_recall_curve_{key_suffix}_{id(fig)}")

def plot_model_comparison(models, accuracies, key_suffix=""):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=models, y=accuracies))
    fig.update_layout(title='Model Comparison', xaxis_title='Model', yaxis_title='Accuracy')
    # Use a unique key by adding a suffix or unique identifier
    unique_key = f"model_comparison_{key_suffix}_{id(fig)}" if key_suffix else f"model_comparison_{id(fig)}"
    st.plotly_chart(fig, key=unique_key)

def risk_score_gauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Score"},
        gauge = {'axis': {'range': [0, 100]},
                 'bar': {'color': "red"},
                 'steps' : [
                     {'range': [0, 50], 'color': "green"},
                     {'range': [50, 75], 'color': "yellow"},
                     {'range': [75, 100], 'color': "red"}]}))
    # Add unique key to avoid duplicate element id error
    st.plotly_chart(fig, key=f"risk_score_gauge_{id(fig)}")

def main():
    st.title("AI-Driven Fake Profile Detection Dashboard")

    # Input fields for account features
    st.subheader("Enter Account Features for Analysis")
    num_friends = st.number_input("Number of Friends", min_value=0, max_value=100000, value=100, key=f"num_friends_{time.time()}")
    num_posts = st.number_input("Number of Posts", min_value=0, max_value=100000, value=50, key=f"num_posts_{time.time()}")
    has_profile_picture = st.selectbox("Has Profile Picture", options=["Yes", "No"], key=f"has_profile_picture_{time.time()}")
    bio_length = st.number_input("Bio Length (characters)", min_value=0, max_value=1000, value=100, key=f"bio_length_{time.time()}")
    account_age_days = st.number_input("Account Age (days)", min_value=0, max_value=10000, value=365, key=f"account_age_days_{time.time()}")
    follower_count = st.number_input("Number of Followers", min_value=0, max_value=100000, value=150, key=f"follower_count_{time.time()}")
    following_count = st.number_input("Number Following", min_value=0, max_value=100000, value=100, key=f"following_count_{time.time()}")
    avg_post_frequency = st.number_input("Average Posts per Day", min_value=0.0, max_value=100.0, value=0.5, key=f"avg_post_frequency_{time.time()}")
    has_verified_badge = st.selectbox("Has Verified Badge", options=["Yes", "No"], key=f"has_verified_badge_{time.time()}")

    if st.button("Check Account", key=f"check_account_button_{time.time()}"):
        try:
            # Preprocess inputs into feature vector
            features = {
                "num_friends": num_friends,
                "num_posts": num_posts,
                "has_profile_picture": 1 if has_profile_picture == "Yes" else 0,
                "bio_length": bio_length,
                "account_age_days": account_age_days,
                "follower_count": follower_count,
                "following_count": following_count,
                "avg_post_frequency": avg_post_frequency,
                "has_verified_badge": 1 if has_verified_badge == "Yes" else 0
            }

            # Convert features to DataFrame for preprocessing
            import pandas as pd
            input_df = pd.DataFrame([features])

            # Preprocess input data
            from data.preprocessing import preprocess_data as preprocess_data_func
            # Add required columns with default empty strings to input_df to avoid KeyError
            for col in ['username', 'bio', 'post_content']:
                if col not in input_df.columns:
                    input_df[col] = ''
            processed_input, _, _ = preprocess_data_func(input_df)

            # Load models
            rf_model = load_sklearn_model(config['model_paths']['random_forest'])
            xgb_model = load_sklearn_model(config['model_paths']['xgboost'])
            lgbm_model = load_sklearn_model(config['model_paths']['lightgbm'])
            ann_model = load_ann_model(config['model_paths']['ann'])

            # Predict probabilities
            rf_pred = predict_sklearn(rf_model, processed_input)[0]
            xgb_pred = predict_sklearn(xgb_model, processed_input)[0]
            lgbm_pred = predict_sklearn(lgbm_model, processed_input)[0]
            ann_pred = predict_ann(ann_model, processed_input)[0]

            # Aggregate predictions (simple average)
            risk_score = np.mean([rf_pred, xgb_pred, lgbm_pred, ann_pred]) * 100

            st.subheader("Analysis Results")
            st.write(f"Risk Score: {risk_score:.2f}%")

            # Explainability with SHAP for Random Forest
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(processed_input)

            st.header("Explainability Section")
            st.write("Feature importance and SHAP values for the Random Forest model:")
            shap.summary_plot(shap_values, processed_input, show=False)
            st.pyplot(bbox_inches='tight')
            plt.clf()

            # Profile Insights Popup
            st.header("Profile Insights Popup")
            st.write("Detailed profile insights will be displayed here on selection.")
            st.write(f"Input Features: {features}")
            st.write(f"Predictions: RF={rf_pred:.2f}, XGB={xgb_pred:.2f}, LGBM={lgbm_pred:.2f}, ANN={ann_pred:.2f}")

def display_risk_assessment(risk_score, shap_values, feature_names):
    st.header("Risk Assessment Report")
    if risk_score > 70:
        st.error("This account is flagged as potentially fake due to high risk score.")
        st.write("Reasons:")
    else:
        st.success("This account is likely genuine based on the analysis.")
        st.write("Reasons:")

    importances = np.abs(shap_values).mean(axis=0)
    top_indices = importances.argsort()[-3:][::-1]
    for idx in top_indices:
        st.write(f"- {feature_names[idx]}: impact score {importances[idx]:.4f}")

# In main function, replace the previous risk explanation block with a call to this function
...
            # Explanation for flagged account with reasons
+            display_risk_assessment(risk_score, shap_values, processed_input.columns)

        except Exception as e:
            st.error("An error occurred during analysis. Please try again.")
            logger.error(f"Error during account analysis: {e}", exc_info=True)

    # Placeholder data for demonstration
    models = ["Random Forest"]
    accuracies = [0.85]
    epochs = list(range(1, 11))
    acc_epochs = [0.7 + 0.02*i for i in range(10)]

    plot_accuracy_vs_epochs(epochs, acc_epochs)
    # Only plot accuracy curve and risk score gauge once as requested
    risk_score_gauge(0)

if __name__ == "__main__":
    main()

    # Placeholder data for demonstration
    models = ["Random Forest", "XGBoost", "LightGBM", "ANN"]
    accuracies = [0.85, 0.88, 0.87, 0.83]
    epochs = list(range(1, 11))
    acc_epochs = [0.7 + 0.02*i for i in range(10)]
    cm = [[50, 10], [5, 35]]
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr)
    precision = np.linspace(1, 0, 100)
    recall = np.linspace(0, 1, 100)

    plot_accuracy_vs_epochs(epochs, acc_epochs)
    plot_confusion_matrix(cm)
    plot_roc_curve(fpr, tpr)
    plot_precision_recall_curve(precision, recall)
    plot_model_comparison(models, accuracies)
    risk_score_gauge(0)

if __name__ == "__main__":
    main()
