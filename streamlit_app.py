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

st.title("Dry Bean Multi-Class Classifier")

# -----------------------------
# Load all saved models
# -----------------------------
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

# -----------------------------
# Dataset Upload
# -----------------------------
st.subheader("Upload Test Dataset (CSV Only)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(test_df.head())

    if "Class" not in test_df.columns:
        st.error("Uploaded CSV must contain 'Class' column.")
    else:
        X_test = test_df.drop("Class", axis=1)
        y_test = test_df["Class"]

        # -----------------------------
        # Model Selection
        # -----------------------------
        st.subheader("Select Model")
        selected_model_name = st.selectbox("Choose a model", list(models.keys()))
        model = models[selected_model_name]

        # -----------------------------
        # Predictions
        # -----------------------------
        y_pred = model.predict(X_test)

        # -----------------------------
        # Evaluation Metrics
        # -----------------------------
        st.subheader("Evaluation Metrics")

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")
        mcc = matthews_corrcoef(y_test, y_pred)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
        else:
            auc = np.nan

        st.write(f"**Accuracy:** {accuracy:.4f}")
        st.write(f"**AUC:** {auc:.4f}")
        st.write(f"**Precision:** {precision:.4f}")
        st.write(f"**Recall:** {recall:.4f}")
        st.write(f"**F1 Score:** {f1:.4f}")
        st.write(f"**MCC:** {mcc:.4f}")

        # -----------------------------
        # Confusion Matrix
        # -----------------------------
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        # -----------------------------
        # Classification Report
        # -----------------------------
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred)
        st.text(report)