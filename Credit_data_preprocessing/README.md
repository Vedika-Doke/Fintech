# Detailed Analysis

## FinTech Data Preprocessing with PCA-Anonymized Features
### Dataset Context
The FinTech dataset used for modeling is **privacy-preserving**.  
To protect user identities and sensitive transaction attributes, the original features have been transformed using **Principal Component Analysis (PCA)**.

The released dataset contains:
- Features: `V1` to `V28` (PCA-transformed components)
- Target label: Binary classification (Fraud / Non-Fraud)
---

#### Motivation for PCA in Financial Data

Raw financial transaction data often includes:
- Merchant identifiers
- Location and device metadata
- Time-based behavioral patterns

Such attributes are **personally identifiable or commercially sensitive**.  
PCA anonymizes these features by projecting them into an orthogonal feature space where:

- Original variables cannot be directly reconstructed
- Semantic meaning of features is intentionally removed
- Statistical structure relevant for learning is preserved

---

#### What PCA Does (Mathematical Overview)

Given a standardized data matrix \( X \in \mathbb{R}^{n \times d} \):

1. **Standardization**
2. **Covariance Matrix**
3. **Eigen Decomposition**
4. **Projection**

Where:
- \( V_k \) contains the top-\(k\) eigenvectors
- Each principal component captures maximum remaining variance
- Components are **orthogonal and uncorrelated**

---

#### Interpretation of Features (V1–V28)

- `V1`–`V28` are **principal components**, not raw transaction attributes
- Each component is a **linear combination** of original features
- Components:
  - Are mean-centered and variance-scaled
  - Have no direct financial interpretation
  - Preserve global variance, not class separability

As a result:
- Feature semantics are lost
- Interpretability is limited
- Predictive modeling remains feasible

---

### Preprocessing Implications

#### What Is Already Done
- Features are standardized
- Correlation between components is removed
- Scale normalization is implicit

#### What Must Be Avoided
- Applying PCA again (double PCA destroys structure)
- Domain-based feature engineering
- Dropping components without variance analysis

#### What Is Still Required
- Target vector validation
- Class imbalance analysis
- Train–test separation before any resampling

---

### Impact on Fraud Detection Modeling

#### Class Imbalance
Fraud cases are extremely rare compared to legitimate transactions.

Mitigation strategies:
- Class-weighted loss functions
- SMOTE (distance-based oversampling)
- ROC–AUC–based evaluation

#### Model Choice Justification
- PCA preserves variance, not linear separability
- Nonlinear models (e.g., RBF-SVM, neural networks) perform better
- Euclidean distance remains meaningful due to orthogonality

---

During preprocessing, the focus shifts from feature interpretation to statistical robustness—handling class imbalance, ensuring proper evaluation, and selecting models capable of capturing nonlinear structure in the transformed feature space.

---

## Preprocessing Objectives

The preprocessing pipeline was designed to:
- Prepare PCA-transformed data for downstream ML models
- Preserve privacy guarantees by avoiding inverse transformations
- Handle extreme class imbalance correctly
- Prevent data leakage
- Enable reliable model evaluation

---

## Data Loading and Inspection

- Loaded dataset using Pandas into a structured DataFrame
- Verified:
  - Feature dimensionality
  - Data types
  - Presence of null or malformed entries
- Confirmed absence of missing values in PCA-transformed components

---

## Understanding PCA-Anonymized Features

- `V1`–`V28` are treated as **abstract numerical features**
- No semantic interpretation is assumed
- No domain-driven feature engineering is performed
- Features are already:
  - Mean-centered
  - Variance-normalized
  - Uncorrelated

As a result, additional normalization or PCA is intentionally avoided.

---

## Target Variable Extraction

- Isolated the target vector (`y`) from the feature matrix
- Verified label integrity and binary encoding
- Ensured target labels were not influenced by any future information

This step is critical in FinTech datasets, where incorrect label construction
can introduce subtle data leakage.

---

## Class Distribution Analysis

- Computed class frequencies to quantify imbalance
- Visualized class distribution using bar plots
- Observed extreme skew toward the non-fraud class

This analysis guided downstream decisions regarding:
- Evaluation metrics
- Resampling strategies
- Model choice

---

## Multilingual Data Handling

In real-world FinTech scenarios, transaction details and other features (`x`) may be stored in multiple languages (e.g., English, Hindi, Marathi). A key challenge is to normalize these inputs so that a model can interpret them uniformly, while the target vector `y` remains consistent. For example, a loan default is `1` regardless of the language of the application.

A conceptual preprocessing approach includes these steps:
*   **Language Detection:** Identifying the language of each text entry.
*   **Translation or Multilingual Embeddings:** Converting all text into a single language (e.g., English) or using models that generate language-agnostic numerical embeddings.
*   **Vectorization:** Transforming the normalized text into a numerical format that the model can process.

Throughout this process, `y` is unchanged, as it is already in a uniform, language-agnostic numerical format. By unifying the multilingual input features, it can be ensured that the model’s interpretation of transactions is consistent, while `y` remains an accurate representation of the target.

The dataset in this project can be viewed as the *result* of such a process, where all original features, including any potential text data has been vectorized and anonymized via PCA into a numerical embedding suitable for a machine learning model.

- ### Multilingual Embedding-Based Normalization (Preferred)

Text is passed through a multilingual embedding model trained to map semantically similar sentences across languages to nearby vectors.
Output embeddings are:
- Fixed-dimensional
- Language-agnostic
- Comparable across scripts and grammars

Key property:

Embed
(
“Loan approved”
)
≈
Embed
(
“ऋण स्वीकृत”
)
Embed(“Loan approved”)≈Embed(“ऋण स्वीकृत”)

This removes the need for translation while preserving meaning.

## Implementation
The practical analysis was implemented in Python within the first part of this notebook. The high-level steps were:
*    **Setup:** The relevant libraries were loaded followed by the dataset of credit card fraud which was loaded from a public URL into a `pandas` DataFrame. The dataset is numerical and is a result of PCA Dimensionality reduction to protect user identities and sensitive features(v1-v28).
*    **Extracting the Target Vector:** The target vector was loaded from the `Class` row in the DataFrame to a `numpy` vector `y_vector`
*    **Analysis:** Basic `numpy` operations were done on `y_vector` to calculate the mean of the vector (fraud rate), and the distribution among the two classes was counted.
*    **Visualisation:** The `seaborn` library was used to generate a count plot, visually demonstrating the class imbalance. Logarithmic scale was used on the y-axis for clarity.

## Conclusion
The target vector `y` is a fundamental concept in supervised machine learning. The proper extraction and representation of the target vector are the first steps towards building any predictive model. As demonstrated with the credit card fraud detection use-case, a preliminary analysis of the target vector provides critical insights such as class imbalance, which directly affects the strategy for model development, and hence ensures the creation of robust solutions.

### Citations and References
1. CS 419 Slides, Autumn 2025, Prof. Ganesh Ramakrishnan
2. Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
