# Overview
---
## 1. Data Preprocessing
This notebook focuses exclusively on preparing the FinTech dataset for
downstream machine learning tasks.

- The raw dataset is loaded and inspected for structure, feature types, and
  class imbalance.
- PCA-transformed features (`V1–V28`) are treated as anonymized numerical
  embeddings and are not re-scaled.
- Raw numerical features (e.g., transaction amount) are standardized to ensure
  numerical stability.
- The target variable is extracted and validated separately to avoid leakage.
- Class distribution is analyzed to quantify imbalance prior to modeling.

The output of this preprocessing step is a clean, machine-learning-ready
dataset that preserves privacy constraints while maintaining statistical
integrity.


## 2. SVMs for credit card fraud detection 
This project implements a fraud detection pipeline on a PCA-anonymized FinTech
dataset using Support Vector Machines (SVMs), with careful handling of extreme
class imbalance and leakage-free evaluation.

---

### Data Splitting (Stratified)

The dataset is split into training, validation, and test sets using a two-stage
**stratified split**. Stratification preserves the original fraud-to-non-fraud
ratio in all subsets, ensuring that fraud cases are present everywhere without
altering class imbalance.

---

### Handling Class Imbalance

Class imbalance is addressed **only in the training set** using:

- **SMOTE** to generate synthetic fraud samples and increase learning signal  
- **`class_weight='balanced'`** to penalize misclassification of fraud more
  heavily during optimization  

SMOTE changes the data distribution, while class weighting changes the loss
function, making them complementary.

---

### Model and Evaluation

Linear and kernel-based SVMs are trained and evaluated on untouched test data.
Performance is measured using **precision, recall, F1-score, and PR-AUC**, which
are more informative than accuracy for rare-event detection.

---

### Summary

Stratified splitting ensures fair evaluation, SMOTE and class weighting address
imbalance during training, and SVMs with nonlinear kernels capture complex fraud
patterns effectively.
