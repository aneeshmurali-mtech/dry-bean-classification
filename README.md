# Dry Bean Multi-Class Classification Project

## a. Problem Statement
The goal of this project is to build and evaluate multiple machine learning classification models on a **multi-class dataset**.  
The task is to predict the **type of dry bean** based on 16 morphological features extracted from images.  
This project includes:
- Fetching dataset directly from UCI
- Training 6 ML models
- Computing 6 evaluation metrics for each model
- Saving trained models
- Deploying a Streamlit app for inference
- Preparing a GitHub repository with the required structure

---

## b. Dataset Description

**Dataset:** Dry Bean Dataset  
**Source:** UCI Machine Learning Repository  
**Instances:** 13,611  
**Features:** 16 numeric features  
**Target:** 7 bean varieties (multi-class)

### Target Classes
- SEKER  
- BARBUNYA  
- BOMBAY  
- CALI  
- DERMASON  
- HOROZ  
- SIRA  

### Feature Summary
All features are numeric and represent shape, geometry, and texture of beans, such as:
- Area  
- Perimeter  
- MajorAxisLength  
- MinorAxisLength  
- ConvexArea  
- Eccentricity  
- Solidity  
- Extent  
- Roundness  
- Compactness  
- ShapeFactor1–4  

---

## c. Models Used & Evaluation Metrics 

The following machine learning models were implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Gaussian Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

Each model was evaluated using the following metrics:
- Accuracy  
- AUC Score (macro)  
- Precision (macro)  
- Recall (macro)  
- F1 Score (macro)  
- Matthews Correlation Coefficient (MCC)

### Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.9236 | 0.9948 | 0.9379 | 0.9350 | 0.9362 | 0.9077 |
| Decision Tree | 0.8927 | 0.9447 | 0.9100 | 0.9084 | 0.9091 | 0.8703 |
| KNN | 0.9186 | 0.9870 | 0.9339 | 0.9296 | 0.9315 | 0.9016 |
| Naive Bayes |  |  |  |  |  |  |
| Random Forest |  |  |  |  |  |  |
| XGBoost |  |  |  |  |  |  |

*******TODO =========*(Fill these values after running `train_models.py`.)*