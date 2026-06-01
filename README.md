# Credit Risk Probability Model for Alternative Data

## Credit Scoring Business Understanding

### 1. Basel II and the need for an interpretable, well-documented model

The Basel II Accord emphasizes sound risk measurement, transparency, and strong governance in credit risk management. For a credit scoring model, this means the model should not only produce accurate predictions, but also be explainable, reproducible, and properly documented.

In this project, interpretability is important because the model may influence lending decisions such as loan approval, credit limits, and repayment terms. A regulated financial institution like Bati Bank must be able to explain why a customer is classified as high risk or low risk. This supports internal model validation, auditability, fairness review, and regulatory reporting.

Therefore, the modeling process should include clear documentation of data sources, feature engineering logic, proxy target construction, model assumptions, performance metrics, and limitations. Simpler models such as Logistic Regression with Weight of Evidence transformation are often useful in regulated contexts because their outputs are easier to explain to business and risk stakeholders.

### 2. Why a proxy variable is necessary and the risks it introduces

The dataset does not contain a direct default label showing whether a customer failed to repay a loan. However, supervised credit risk modeling requires a target variable. Because actual repayment behavior is unavailable, a proxy variable must be created to approximate credit risk.

In this project, the proxy target will be engineered from customer transaction behavior using Recency, Frequency, and Monetary value analysis. Customers who are less engaged, transact rarely, and generate low monetary value may be treated as higher-risk customers, while more active customers may be treated as lower-risk customers.

This approach introduces business risks because the proxy is not the same as actual loan default. A customer with low activity is not necessarily unable or unwilling to repay a loan. Similarly, a highly active customer may still default. Proxy-based prediction can therefore introduce misclassification risk, bias, and unfair lending outcomes. For this reason, the proxy target must be clearly documented as a modeling assumption rather than ground truth.

### 3. Trade-offs between simple interpretable models and high-performance models

In a regulated financial context, there is an important trade-off between interpretability and predictive performance.

A simple model such as Logistic Regression with Weight of Evidence is easier to explain, validate, and document. It provides clear relationships between features and risk, making it suitable for scorecard-style credit risk modeling. However, it may fail to capture complex non-linear relationships in customer behavior.

A high-performance model such as Random Forest, Gradient Boosting, XGBoost, or LightGBM may achieve better predictive accuracy by capturing complex patterns in the data. However, these models are often harder to interpret and may require additional explainability tools. This can make regulatory review, stakeholder communication, and model governance more difficult.

For this project, both interpretable and higher-performance models should be compared. The final model choice should balance predictive performance, explainability, reproducibility, and regulatory defensibility.