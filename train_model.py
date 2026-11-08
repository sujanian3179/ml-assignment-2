"""
Train and evaluate five classifiers on the UCI Bank Marketing dataset.

Outputs
-------
model/<name>.joblib   full sklearn Pipeline (preprocessing + classifier) per model
model/metrics.csv     evaluation metrics of every model on the held-out test split
test_data.csv         the held-out test split (features + target), for the Streamlit app

Run:  python train_model.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "bank.csv"
MODEL_DIR = ROOT / "model"
TARGET = "y"
RANDOM_STATE = 42

# `duration` (length of the last call) is only known *after* the call ends,
# so it leaks the outcome. The UCI documentation recommends dropping it for a
# realistic predictive model.
DROP_COLS = ["duration"]

NUMERIC_COLS = ["age", "balance", "day", "campaign", "pdays", "previous"]
CATEGORICAL_COLS = ["job", "marital", "education", "default", "housing",
                    "loan", "contact", "month", "poutcome"]

MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=20, class_weight="balanced",
        random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(
        kernel="rbf", C=1.0, probability=True, class_weight="balanced",
        random_state=RANDOM_STATE),
}


def file_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_") + ".joblib"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        # dense output so GaussianNB can consume it
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_COLS),
    ])


def evaluate(model, X, y) -> dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH).drop(columns=DROP_COLS)
    X = df.drop(columns=[TARGET])
    y = (df[TARGET] == "yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    MODEL_DIR.mkdir(exist_ok=True)
    rows = []
    for name, clf in MODELS.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, MODEL_DIR / file_name(name))
        rows.append({"Model": name, **evaluate(pipe, X_test, y_test)})
        print(f"trained {name}")

    metrics = pd.DataFrame(rows).set_index("Model").round(4)
    metrics.to_csv(MODEL_DIR / "metrics.csv")
    print("\n", metrics.to_string())

    # Save the held-out split with the original target labels for the app.
    test_df = X_test.copy()
    test_df[TARGET] = y_test.map({1: "yes", 0: "no"})
    test_df.to_csv(ROOT / "test_data.csv", index=False)
    print(f"\nsaved test_data.csv ({len(test_df)} rows)")


if __name__ == "__main__":
    main()
