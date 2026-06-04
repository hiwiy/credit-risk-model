import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
DATA_PATH = "data/processed/model_dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/best_model.pkl"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)

    if "is_high_risk" not in df.columns:
        raise ValueError("Target column 'is_high_risk' not found.")

    X = df.drop(columns=["is_high_risk"])
    y = df["is_high_risk"]

    if "CustomerId" in X.columns:
        X = X.drop(columns=["CustomerId"])

    return X, y


def build_preprocessor(X):
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def get_models(preprocessor):
    models = {
        "logistic_regression": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=1000,
                            random_state=RANDOM_STATE,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
            "params": {
                "classifier__C": [0.1, 1.0, 10.0],
            },
        },
        "random_forest": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        RandomForestClassifier(
                            random_state=RANDOM_STATE,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [5, 10, None],
            },
        },
        "gradient_boosting": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        GradientBoostingClassifier(
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, 5],
            },
        },
    }

    return models


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    return metrics


def train_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)
    models = get_models(preprocessor)

    mlflow.set_experiment("credit-risk-model")

    best_model = None
    best_model_name = None
    best_roc_auc = -1
    best_metrics = None

    for model_name, config in models.items():
        with mlflow.start_run(run_name=model_name):
            grid_search = GridSearchCV(
                estimator=config["pipeline"],
                param_grid=config["params"],
                cv=3,
                scoring="roc_auc",
                n_jobs=-1,
            )

            grid_search.fit(X_train, y_train)

            model = grid_search.best_estimator_
            metrics = evaluate_model(model, X_test, y_test)

            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name="credit-risk-best-model",
            )

            print(f"\nModel: {model_name}")
            print(f"Best Params: {grid_search.best_params_}")
            print(f"Metrics: {metrics}")

            if metrics["roc_auc"] > best_roc_auc:
                best_roc_auc = metrics["roc_auc"]
                best_model = model
                best_model_name = model_name
                best_metrics = metrics

    joblib.dump(best_model, MODEL_PATH)

    print("\nBest Model")
    print(f"Name: {best_model_name}")
    print(f"ROC-AUC: {best_roc_auc}")
    print(f"Metrics: {best_metrics}")
    print(f"Saved to: {MODEL_PATH}")

    return best_model, best_model_name, best_metrics


if __name__ == "__main__":
    train_models()