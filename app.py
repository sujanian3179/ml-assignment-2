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

ACCENT = "#2f4fb0"
MUTED = "#c9d1e6"

st.markdown(f"""
<style>
  .block-container {{ padding-top: 2rem; max-width: 1150px; }}
  .hero {{
    background: linear-gradient(135deg, #1e2f6e 0%, {ACCENT} 60%, #4a6fd8 100%);
    color: #fff; border-radius: 18px; padding: 1.6rem 2rem 1.4rem;
    margin-bottom: 1.4rem; box-shadow: 0 8px 24px rgba(30,47,110,.18);
  }}
  .hero .chips {{ display: flex; flex-wrap: wrap; gap: .45rem; margin-bottom: .9rem; }}
  .hero .chip {{
    background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25);
    border-radius: 999px; padding: .18rem .75rem; font-size: .78rem;
    letter-spacing: .04em; text-transform: uppercase;
  }}
  .hero h1 {{ color: #fff; font-size: 1.9rem; font-weight: 700; margin: 0 0 .35rem; padding: 0; }}
  .hero p.sub {{ color: rgba(255,255,255,.85); margin: 0 0 1rem; font-size: .98rem; }}
  .hero .meta {{
    display: flex; flex-wrap: wrap; gap: 1.6rem; font-size: .88rem;
    border-top: 1px solid rgba(255,255,255,.2); padding-top: .8rem;
  }}
  .hero .meta span {{ color: rgba(255,255,255,.65); display: block; font-size: .72rem;
    text-transform: uppercase; letter-spacing: .06em; }}
  [data-testid="stMetric"] {{
    background: #f4f6fb; border: 1px solid #e3e8f3; border-radius: 14px;
    padding: .8rem 1rem; border-top: 3px solid {ACCENT};
  }}
  [data-testid="stMetricLabel"] p {{ font-size: .78rem; text-transform: uppercase;
    letter-spacing: .05em; color: #5b6478; }}
  [data-testid="stMetricValue"] {{ font-size: 1.55rem; font-weight: 650; color: #1b2235; }}
  h3 {{ font-size: 1.05rem !important; font-weight: 650 !important; color: #1b2235; }}
  .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
  .footer {{ text-align: center; color: #8a92a6; font-size: .8rem; margin-top: 2.5rem; }}
</style>
""", unsafe_allow_html=True)


def style_axes(ax) -> None:
    sns.despine(ax=ax)
    ax.set_facecolor("none")
    ax.figure.patch.set_alpha(0)
    ax.tick_params(colors="#5b6478")
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_color("#5b6478")


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    model_name = st.selectbox("Model", list(MODEL_FILES), index=4)
    uploaded = st.file_uploader("Upload test data (CSV)", type="csv")
    use_sample = st.toggle("Use bundled sample test data", value=uploaded is None)
    st.download_button("⬇️ Download sample test_data.csv", SAMPLE_TEST.read_bytes(),
                       file_name="test_data.csv", mime="text/csv", width="stretch")
    st.divider()
    st.caption("Models are pre-trained and loaded from disk. Uploaded data is used "
               "only for evaluation, never for retraining.")

# ---------------- Header ----------------
st.markdown("""
<div class="hero">
  <div class="chips">
    <div class="chip">BITS Pilani · WILP</div>
    <div class="chip">M.Tech AIML</div>
    <div class="chip">Machine Learning · Assignment 2</div>
  </div>
  <h1>🏦 Bank Term-Deposit Subscription Predictor</h1>
  <p class="sub">Five classifiers trained on the UCI Bank Marketing dataset. Upload test data,
  pick a model and inspect how well it predicts which clients will subscribe.</p>
  <div class="meta">
    <div><span>Instructor In-Charge (I/C)</span>Prof. Kiruthiga A R</div>
    <div><span>Submitted by</span>Ayush Ranjan Jha · 2026AJ05881</div>
  </div>
</div>
""", unsafe_allow_html=True)

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    df = pd.read_csv(SAMPLE_TEST)
else:
    st.info("⬅️ Upload a CSV or turn on *Use bundled sample test data* to begin.")
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
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        cmap = sns.light_palette(ACCENT, as_cmap=True)
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax,
                    linewidths=3, linecolor="white", annot_kws={"size": 14},
                    xticklabels=["no", "yes"], yticklabels=["no", "yes"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        style_axes(ax)
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
    st.caption("Best value per metric highlighted. Selected model shown in the accent colour below.")
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#dbe4fb"),
                 width="stretch")

    metric = st.segmented_control("Metric to plot", list(table.columns), default="F1") or "F1"
    fig, ax = plt.subplots(figsize=(8, 3.2))
    colors = [ACCENT if m == model_name else MUTED for m in table.index]
    sns.barplot(x=table.index, y=table[metric], ax=ax, hue=table.index,
                palette=colors, legend=False)
    for i, v in enumerate(table[metric]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color="#1b2235")
    ax.set_ylabel(metric)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=12)
    style_axes(ax)
    st.pyplot(fig)

st.markdown('<div class="footer">BITS Pilani WILP · M.Tech AIML · Machine Learning Assignment 2 '
            '· Ayush Ranjan Jha</div>', unsafe_allow_html=True)
