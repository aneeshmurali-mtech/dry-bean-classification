import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report
)

# ---------------------------------------------------------
# CUSTOM CSS FOR VIBRANT UI
# ---------------------------------------------------------
st.markdown("""
<style>
/* Background gradient */
.main {
    background: linear-gradient(135deg, #1e1e2f, #2e2e42);
}

/* Title */
h1 {
    color: #FF6F61 !important;
    text-shadow: 2px 2px 4px #00000050;
}

/* Subheaders */
h2, h3 {
    color: #FFD369 !important;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    border: 2px solid #FF6F61;
    border-radius: 10px;
}

/* Buttons */
.stButton>button {
    background-color: #FF6F61;
    color: white;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    border: none;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #FF9671;
}

/* Selectbox */
.stSelectbox>div>div {
    background-color: #2E2E42 !important;
    color: white !important;
    border-radius: 8px;
}

/* File uploader */
.stFileUploader {
    background-color: #2E2E42;
    padding: 1rem;
    border-radius: 10px;
    border: 2px dashed #FF6F61;
}

/* Metric boxes */
.metric-card {
    padding: 15px;
    border-radius: 10px;
    background: linear-gradient(135deg, #FF6F61, #FF9671);
    color: white;
    text-align: center;
    font-weight: bold;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HEADER BANNER
# ---------------------------------------------------------
st.markdown("""
<div style="padding: 15px; border-radius: 10px;
background: linear-gradient(90deg, #FF6F61, #FF9671, #FFC75F);
color: black; text-align: center; font-size: 22px; font-weight: bold;">
Dry Bean Multi-Class Classifier — ML Dashboard
</div>
""", unsafe_allow_html=True)

st.title("Dry Bean Multi-Class Classifier")

# ---------------------------------------------------------
# Load Models
# ---------------------------------------------------------
model_paths = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

models = {}
for name, path in model_paths.items():
    with open(path, "rb") as f:
        models[name] = pickle.load(f)

# Load Label Encoder
with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load Scaler (for LR & KNN)
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------------------------------------------------------
# File Upload
# ---------------------------------------------------------
st.subheader("Upload Test Dataset (CSV Only)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    # Fix common column name issues
    test_df.columns = test_df.columns.str.strip()
    rename_map = {
        "AspectRation": "AspectRatio",
        "roundness": "Roundness"
    }
    test_df = test_df.rename(columns=rename_map)

    st.write("### Preview of uploaded data:")
    st.dataframe(test_df.head())

    if "Class" not in test_df.columns:
        st.error("Uploaded CSV must contain 'Class' column.")
    else:
        X_test = test_df.drop("Class", axis=1)
        y_test = label_encoder.transform(test_df["Class"])

        # ---------------------------------------------------------
        # Model Selection
        # ---------------------------------------------------------
        st.subheader("Select Model")
        selected_model_name = st.selectbox("Choose a model", list(models.keys()))
        model = models[selected_model_name]

        # ---------------------------------------------------------
        # Apply scaling for LR & KNN
        # ---------------------------------------------------------
        if selected_model_name in ["Logistic Regression", "KNN"]:
            X_test_input = scaler.transform(X_test)
        else:
            X_test_input = X_test

        # ---------------------------------------------------------
        # Predictions
        # ---------------------------------------------------------
        y_pred = model.predict(X_test_input)

        # ---------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------
        st.subheader("Evaluation Metrics")

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")
        mcc = matthews_corrcoef(y_test, y_pred)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_input)
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
        else:
            auc = np.nan

        # Metric Cards
        st.markdown(f"<div class='metric-card'>Accuracy: {accuracy:.4f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'>AUC: {auc:.4f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'>Precision: {precision:.4f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'>Recall: {recall:.4f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'>F1 Score: {f1:.4f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'>MCC: {mcc:.4f}</div>", unsafe_allow_html=True)

        # ---------------------------------------------------------
        # Confusion Matrix
        # ---------------------------------------------------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="magma", ax=ax)
        st.pyplot(fig)

        # ---------------------------------------------------------
        # Classification Report
        # ---------------------------------------------------------
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        st.text(report)