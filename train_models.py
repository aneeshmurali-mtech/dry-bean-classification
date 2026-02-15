import pandas as pd
import numpy as np
import pickle
import os

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

# -----------------------------------------
# STEP 1: FETCH DRY BEAN DATASET FROM UCI
# -----------------------------------------
dry_bean = fetch_ucirepo(id=602)

X = dry_bean.data.features
y = dry_bean.data.targets["Class"]   # target column

print("Dataset loaded successfully!")
print("Shape:", X.shape)
print("Classes:", y.unique())

# Encode string class labels to integers for XGBoost (and consistency)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Label classes (encoded):", list(enumerate(le.classes_)))

# -----------------------------------------
# TRAIN-TEST SPLIT
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)

# Scaling for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------
# METRIC FUNCTION
# -----------------------------------------
def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)

    # AUC for multi-class
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    else:
        auc = np.nan

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"\n===== {name} =====")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"MCC:       {mcc:.4f}")

    return model

# -----------------------------------------
# CREATE MODEL DIRECTORY
# -----------------------------------------
os.makedirs("model", exist_ok=True)

def save_model(model, filename):
    with open(f"model/{filename}", "wb") as f:
        pickle.dump(model, f)

# -----------------------------------------
# MODEL 1: LOGISTIC REGRESSION
# -----------------------------------------
lr = LogisticRegression(max_iter=3000, multi_class="multinomial")
lr.fit(X_train_scaled, y_train)
evaluate(lr, X_test_scaled, y_test, "Logistic Regression")
save_model(lr, "logistic_regression.pkl")

# -----------------------------------------
# MODEL 2: DECISION TREE
# -----------------------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
evaluate(dt, X_test, y_test, "Decision Tree")
save_model(dt, "decision_tree.pkl")

# -----------------------------------------
# MODEL 3: KNN
# -----------------------------------------
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_scaled, y_train)
evaluate(knn, X_test_scaled, y_test, "KNN")
save_model(knn, "knn.pkl")

# -----------------------------------------
# MODEL 4: GAUSSIAN NAIVE BAYES
# -----------------------------------------
nb = GaussianNB()
nb.fit(X_train, y_train)
evaluate(nb, X_test, y_test, "Naive Bayes")
save_model(nb, "naive_bayes.pkl")

# -----------------------------------------
# MODEL 5: RANDOM FOREST
# -----------------------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
evaluate(rf, X_test, y_test, "Random Forest")
save_model(rf, "random_forest.pkl")

# -----------------------------------------
# MODEL 6: XGBOOST
# -----------------------------------------
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=len(le.classes_),
    eval_metric="mlogloss"
)
xgb_model.fit(X_train, y_train)
evaluate(xgb_model, X_test, y_test, "XGBoost")
save_model(xgb_model, "xgboost.pkl")

# Save label encoder for inference mapping
save_model(le, "label_encoder.pkl")

print("\nAll models trained and saved successfully!")