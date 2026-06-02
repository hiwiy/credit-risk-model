# Credit Risk Probability Model for Alternative Data
## Project Overview

This project develops an end-to-end credit risk scoring system for Bati Bank's Buy-Now-Pay-Later (BNPL) service in partnership with an eCommerce platform. Since the available transaction dataset does not contain a direct default label, an alternative approach is used to construct a proxy target variable based on customer behavioral patterns derived from Recency, Frequency, and Monetary (RFM) analysis.
 
## Data Source

This project uses the Xente Fraud Detection Dataset provided as part of the Xente Challenge.

Data Sources:

* Xente Challenge Dataset
* Kaggle: Xente Fraud Detection Challenge

The dataset contains transaction-level records collected from customers using various payment channels on the Xente platform. Each record represents a single transaction and includes customer identifiers, transaction amounts, product information, timestamps, and fraud indicators.

### Dataset Setup

1. Download the dataset from the challenge source or Kaggle.
2. Create the following directory structure:

```text
data/
└── raw/
    └── data.csv
```

3. Place the downloaded CSV file inside the `data/raw/` directory.
4. All data files are excluded from version control through `.gitignore`.
5. Processed datasets generated during feature engineering are saved under `data/processed/`.


# The final solution includes:

Exploratory Data Analysis (EDA)
Feature Engineering Pipeline
Proxy Target Variable Construction
Credit Risk Model Development
MLflow Experiment Tracking
FastAPI Model Serving
Docker Containerization
CI/CD Automation using GitHub Actions
Credit Scoring Business Understanding

## 1. Basel II and the Need for Interpretable Models

In practice, Basel II compliance requires governance artifacts that support model oversight throughout the model lifecycle. Examples include:

* Model development documentation
* Feature engineering documentation
* Validation reports
* Model performance monitoring reports
* Model change logs
* Versioned experiment records
* Periodic model review reports

Maintaining these artifacts ensures transparency, reproducibility, and audit readiness.


This requirement directly influences model development in several ways:

### Transparency

Credit decisions affect customers financially and legally. Regulators require banks to explain why a customer was approved or rejected. Interpretable models provide clear reasoning behind predictions.

### Model Validation

Banks must regularly validate model performance and demonstrate that predictions are based on meaningful risk indicators. Models that are easy to understand simplify validation and auditing processes.

### Regulatory Compliance

Basel II emphasizes governance, documentation, and risk management. Every stage of the modeling pipeline must be documented, including:

Data sources
Feature engineering
Target variable definition
Model selection
Performance evaluation
Risk Monitoring

Over time, customer behavior and economic conditions change. Interpretable models allow risk analysts to identify performance degradation and understand which variables contribute to changing risk patterns.

For these reasons, interpretability is not merely a technical preference but a regulatory necessity in credit risk modeling.

# 2. Why a Proxy Variable Is Necessary

The provided transaction dataset contains no direct indicator of loan default or repayment behavior. Since supervised machine learning requires labeled outcomes, a proxy variable must be created to represent customer risk.

### Need for a Proxy Target

Without a target variable, it is impossible to train a predictive classification model. Therefore, customer behavioral patterns must be used to infer risk.

This project uses RFM (Recency, Frequency, Monetary) analysis to identify customer engagement levels:

Recency: How recently a customer transacted
Frequency: How often a customer transacts
Monetary: The value of customer transactions

Customers exhibiting low engagement characteristics are assumed to have a higher probability of future default and are labeled as high-risk.

### Business Risks of Proxy-Based Modeling

While proxy targets enable model development, they introduce several risks:

### Label Noise

The proxy label is not actual default behavior. Some customers labeled high-risk may never default, while some low-risk customers may eventually default.

### Assumption Risk

The model assumes disengaged customers are more likely to default. This relationship may not always hold in practice.

### Bias Introduction

Behavioral patterns may inadvertently reflect demographic or operational biases, leading to unfair predictions.

### Reduced Predictive Accuracy

A model trained on proxy labels can only learn patterns associated with the proxy, not true default behavior.

Therefore, predictions should be interpreted as estimated risk indicators rather than direct measurements of default probability.

# 3. Trade-Offs Between Logistic Regression and Gradient Boosting

Selecting a credit risk model involves balancing interpretability, regulatory requirements, and predictive performance.

## Logistic Regression with WoE
### Advantages
Highly interpretable
Coefficients have clear business meaning
Easy to explain to regulators
Supports traditional credit scorecard development
Stable and relatively simple to maintain
### Disadvantages
Assumes linear relationships
May underperform on complex datasets
Limited ability to capture nonlinear interactions
## Gradient Boosting Models (XGBoost, LightGBM)
### Advantages
High predictive performance
Captures nonlinear relationships
Handles complex feature interactions automatically
Often achieves superior ROC-AUC scores
### Disadvantages
Less transparent
Harder to explain individual predictions
More difficult to validate and audit
Increased governance and monitoring requirements
## Recommended Approach

In a regulated banking environment, a balance between performance and interpretability is essential.

For this project:

Logistic Regression with WoE will serve as the baseline interpretable model.
Gradient Boosting will serve as the high-performance benchmark.
Model selection will consider both predictive performance and regulatory explainability requirements.

The final recommendation will be based on quantitative metrics as well as compliance with Basel II principles.