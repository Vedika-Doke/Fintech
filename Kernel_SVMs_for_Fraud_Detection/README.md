# Detailed Analysis
## **Overview**

Detecting fraudulent credit card transactions means identifying the tiny fraction of transactions that are unauthorized (typically 0.1%–1% of transactions) among vast number of legitimate ones. This is a high-stakes classification task: missed fraud (false negatives) produces direct financial loss; excessive false alarms (false positives) damage customer experience.

This task is a machine learning challenge due to several factors:

* **Extreme Class Imbalance**: Fraudulent transactions are exceptionally rare, often being less than 0.1% of total transaction volume. This small fraction makes it difficult for standard algorithms to learn the defining features of fraud.
* **High Cost of Errors**:
    * **False Negatives** (missed fraud) result in direct financial losses and loss of customer trust.
    * **False Positives** (legitimate transactions flagged as fraud) create significant customer frustration.
* **Adversarial and Dynamic Nature**: Fraudsters continuously adapt their methods, requiring detection models to be robust and evolve over time.

**Support Vector Machines (SVMs)** offer a solution uniquely suited to these challenges. By moving beyond the limitations of simple linear models like the Perceptron, SVMs provide:
1.  **Optimal Margin Maximization**: For better generalization on unseen data.
2.  **Weighted Penalties**: To counteract class imbalance by focusing on the minority (fraud) class.
3.  **The Kernel Trick**: To model complex, non-linear relationships in transaction data, which are characteristic of complex fraud patterns.

Effective implementation of SVMs can save financial institutions a lot of money annually, secure customer accounts, and ensure the integrity of the digital payments system.

## Project Dependencies and Library Justification

This project implements a machine learning pipeline for credit card fraud
detection using Support Vector Machines (SVMs) on a privacy-preserving,
PCA-anonymized FinTech dataset. Due to the highly imbalanced nature of the data
and the need for reliable evaluation, each library imported in this project
serves a specific and well-motivated role.

---

### Overview of the Pipeline
The imported libraries collectively support the following stages:
1. Data splitting and leakage prevention  
2. Feature scaling and numerical stability  
3. Baseline and advanced classification models  
4. Handling extreme class imbalance  
5. Evaluation using metrics appropriate for rare-event detection

---

## Two-Stage Stratified Data Splitting Strategy

This project uses a **two-stage stratified data splitting strategy** to create
separate training, validation, and test datasets. This design ensures reliable
model evaluation and prevents data leakage, which is especially critical in
highly imbalanced FinTech fraud detection problems.

### Why Stratification Is Critical
Fraud detection datasets are extremely imbalanced. Stratified splitting ensures:
- Minority-class samples appear in all subsets
- Performance metrics remain meaningful
- Models are not trained or evaluated on skewed data

## Handling Class Imbalance with SMOTE

Fraud detection datasets are characterized by extreme class imbalance, where
fraudulent transactions represent only a very small fraction of the total data.
Training a model on such data without correction leads to biased decision
boundaries that favor the majority class.

To address this issue, this project applies **SMOTE (Synthetic Minority
Over-sampling Technique)** to the **training data only**.

---

### Purpose of SMOTE

SMOTE increases the effective representation of the minority (fraud) class during training, enabling the model to learn a meaningful decision boundary
without duplicating samples or introducing label noise.

The validation and test datasets remain untouched to ensure honest and
unbiased evaluation.

---

### How SMOTE Works

For each minority-class sample, SMOTE performs the following steps:

1. Identify the *k* nearest neighbors belonging to the minority class
2. Randomly select one of these neighbors
3. Generate a synthetic sample by interpolating between the two points

Mathematically, a synthetic point is created as:

$$x_{\text{new}} = x_i + \lambda (x_{nn} - x_i), \quad \lambda \in (0,1)$$

This process generates new samples that lie within the existing minority-class
feature space rather than duplicating original observations.

---

### Application in the Pipeline

SMOTE is applied **after stratified data splitting** and **only to the training
set**

##  Use of class_weight='balanced'
`class_weight='balanced'` is used to bias the optimization process toward
correctly classifying fraud cases, while SMOTE ensures adequate representation
of fraud in the training data. Together, they enable effective learning under
severe class imbalance without contaminating validation or test sets.

## Why High Precision and High Recall Are Preferred

In fraud detection, model performance cannot be evaluated using accuracy alone
due to extreme class imbalance. Instead, precision and recall are critical
metrics that capture different types of failure.

---

#### Precision

Precision measures how many transactions flagged as fraud are actually
fraudulent.

- Low precision leads to a high number of false positives
- False positives cause customer inconvenience, operational overhead, and loss
  of trust

High precision ensures that fraud alerts are reliable and actionable.

---

#### Recall

Recall measures how many actual fraud cases are successfully detected.

- Low recall results in missed fraud cases
- Missed fraud leads to direct financial loss and regulatory risk

High recall ensures that fraudulent activity is not overlooked.

---

#### Why Both Are Necessary

Optimizing only recall causes excessive false alarms, while optimizing only
precision allows fraud to go undetected. A practical fraud detection system
must balance both objectives.

For this reason:
- Precision and recall are jointly optimized
- F1-score summarizes this trade-off
- Precision–Recall AUC (PR-AUC) is used as a primary evaluation metric

---

## Hyperparameter Tuning for SVM Models

Support Vector Machines, particularly with the RBF kernel, are highly sensitive
to hyperparameter choices. In this project, hyperparameter tuning is performed
to control the **bias–variance trade-off** and to ensure robust fraud detection
without overfitting synthetic samples generated by SMOTE.

---
### Regularization Parameter (`C`)

The parameter `C` controls the penalty for misclassification.

- Low `C`:
  - Allows more misclassifications
  - Produces a wider margin
  - Leads to higher bias and lower variance
- High `C`:
  - Penalizes misclassification strongly
  - Produces a narrower margin
  - Increases the risk of overfitting

In fraud detection, overly small `C` values may miss fraud cases (low recall),
while overly large `C` values may overfit minority-class noise.

---

### Kernel Coefficient (`gamma`) — RBF Kernel

The parameter `gamma` controls the locality of influence of individual data
points.

- Low `gamma`:
  - Each point influences a large region
  - Results in a smoother, more global decision boundary
- High `gamma`:
  - Each point influences only a small region
  - Produces highly complex, localized boundaries

Excessively large `gamma` values can cause the model to memorize synthetic SMOTE
samples, reducing generalization performance.

---

#### Evaluation Criterion

Hyperparameter selection is guided by **Precision–Recall AUC (PR-AUC)** rather
than accuracy or ROC-AUC. PR-AUC is more informative for highly imbalanced
datasets, as it directly reflects the model’s ability to rank fraud cases above
legitimate transactions.

Evaluation is performed on untouched validation or test data to ensure unbiased
performance estimates.

---

Hyperparameter tuning controls the flexibility of the SVM decision boundary.
By carefully selecting `C` and `gamma`, the model achieves a balance between
detecting fraudulent transactions and generalizing beyond synthetic training
samples, leading to robust performance on real-world data.

