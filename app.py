"""Streamlit app: evaluate pre-trained Bank Marketing classifiers on uploaded test data."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
SAMPLE_TEST = ROOT / "test_data.csv"
TARGET = "y"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Term Deposit Predictor", page_icon="🏦", layout="wide")


@st.cache_resource
def load_models() -> dict:
    return {name: joblib.load(MODEL_DIR / f) for name, f in MODEL_FILES.items()}


def to_binary(target: pd.Series) -> pd.Series:
    """Accept 'yes'/'no' or 1/0 targets."""
    if pd.api.types.is_numeric_dtype(target):
        return target.astype(int)
    return (target.astype(str).str.strip().str.lower() == "yes").astype(int)


def score(model, X, y) -> tuple[dict, pd.Series]:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob) if y.nunique() > 1 else float("nan"),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return metrics, pd.Series(y_pred, index=X.index)


models = load_models()
expected_cols = list(next(iter(models.values())).feature_names_in_)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox("Choose a model", list(MODEL_FILES))
    uploaded = st.file_uploader("Upload test data (CSV)", type="csv")
    use_sample = st.checkbox("Use bundled sample test data", value=uploaded is None)
    st.download_button("Download sample test_data.csv", SAMPLE_TEST.read_bytes(),
                       file_name="test_data.csv", mime="text/csv")
    st.caption("Models are pre-trained; uploaded data is only used for evaluation.")

# ---------------- Header ----------------
st.title("🏦 Bank Term-Deposit Subscription Predictor")
st.write(
    "Predicts whether a bank client will subscribe to a term deposit "
    "(UCI Bank Marketing dataset). Upload a test CSV with the 15 input features "
    f"and the target column **`{TARGET}`**, pick a model, and inspect its performance."
)

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    df = pd.read_csv(SAMPLE_TEST)
else:
    st.info("⬅️ Upload a CSV or tick *Use bundled sample test data* to begin.")
    st.stop()

missing = [c for c in expected_cols + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing required columns: {', '.join(missing)}")
    st.stop()

X = df[expected_cols]
y = to_binary(df[TARGET])

with st.expander(f"Preview data — {len(df)} rows"):
    st.dataframe(df.head(20), width="stretch")

tab_single, tab_compare = st.tabs([f"📊 {model_name}", "🏆 Compare all models"])

# ---------------- Selected model ----------------
with tab_single:
    metrics, y_pred = score(models[model_name], X, y)

    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.3f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["no", "yes"], yticklabels=["no", "yes"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.subheader("Classification report")
        report = classification_report(y, y_pred, labels=[0, 1],
                                       target_names=["no", "yes"],
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

    result = df.copy()
    result["prediction"] = y_pred.map({1: "yes", 0: "no"})
    st.download_button("Download predictions", result.to_csv(index=False),
                       file_name=f"predictions_{model_name.replace(' ', '_').lower()}.csv",
                       mime="text/csv")

# ---------------- Comparison ----------------
with tab_compare:
    table = pd.DataFrame({name: score(m, X, y)[0] for name, m in models.items()}).T
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#c6efce"),
                 width="stretch")

    metric = st.radio("Metric to plot", list(table.columns), horizontal=True, index=4)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.barplot(x=table.index, y=table[metric], ax=ax, color="#4c72b0")
    ax.set_ylabel(metric)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    st.pyplot(fig)
