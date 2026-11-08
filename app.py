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

GLASS_CSS = """
<style>
  /* ---------- animated colour backdrop (what the glass blurs) ---------- */
  [data-testid="stApp"] { background: #eef1fb; }
  [data-testid="stApp"]::before {
    content: ""; position: fixed; inset: -15%; z-index: 0; pointer-events: none;
    background:
      radial-gradient(38% 38% at 18% 22%, rgba(99,123,255,.55), transparent 70%),
      radial-gradient(32% 32% at 82% 18%, rgba(167,112,255,.45), transparent 70%),
      radial-gradient(40% 40% at 75% 85%, rgba(56,189,248,.45), transparent 70%),
      radial-gradient(30% 30% at 22% 88%, rgba(244,114,182,.35), transparent 70%);
    filter: blur(30px);
    animation: drift 22s ease-in-out infinite alternate;
  }
  @keyframes drift {
    0%   { transform: translate(0, 0) rotate(0deg) scale(1); }
    50%  { transform: translate(3%, -2%) rotate(6deg) scale(1.06); }
    100% { transform: translate(-3%, 2%) rotate(-5deg) scale(1.02); }
  }
  [data-testid="stAppViewContainer"] { position: relative; z-index: 1; background: transparent; }
  [data-testid="stMain"] { background: transparent; }
  [data-testid="stHeader"] { background: transparent; }
  .block-container { padding-top: 2rem; max-width: 1150px; }

  /* ---------- shared glass surface ---------- */
  /* NOTE: backdrop-filter / transform on an element turn it into the containing
     block for position:fixed children, which breaks Streamlit's fullscreen view
     of charts and tables. Containers that hold such elements therefore get the
     blur on a ::before layer and an opacity-only entrance animation. */
  .hero, [data-testid="stMetric"], [data-testid="stExpander"] details,
  [class*="st-key-panel"], [data-testid="stDataFrame"] {
    background: rgba(255,255,255,.42);
    border: 1px solid rgba(255,255,255,.65);
    box-shadow: 0 8px 32px rgba(31,45,110,.12), inset 0 1px 0 rgba(255,255,255,.7);
  }
  .hero, [data-testid="stMetric"] {
    backdrop-filter: blur(18px) saturate(160%);
    -webkit-backdrop-filter: blur(18px) saturate(160%);
  }
  [data-testid="stExpander"] details, [class*="st-key-panel"] {
    position: relative;
  }
  [data-testid="stExpander"] details::before, [class*="st-key-panel"]::before {
    content: ""; position: absolute; inset: 0; z-index: -1; border-radius: inherit;
    backdrop-filter: blur(18px) saturate(160%);
    -webkit-backdrop-filter: blur(18px) saturate(160%);
    pointer-events: none;
  }

  /* ---------- sidebar ---------- */
  [data-testid="stSidebar"] {
    background: rgba(255,255,255,.38);
    backdrop-filter: blur(22px) saturate(160%);
    -webkit-backdrop-filter: blur(22px) saturate(160%);
    border-right: 1px solid rgba(255,255,255,.6);
  }
  [data-testid="stSidebar"] > div { background: transparent; }

  /* ---------- hero ---------- */
  .hero {
    position: relative; overflow: hidden; border-radius: 22px;
    padding: 1.7rem 2rem 1.4rem; margin-bottom: 1.4rem;
    background: linear-gradient(135deg, rgba(30,47,110,.82), rgba(47,79,176,.72) 55%, rgba(110,90,230,.66));
    background-size: 200% 200%;
    color: #fff; border-color: rgba(255,255,255,.35);
    animation: rise .8s cubic-bezier(.2,.7,.2,1) both, hue 12s ease-in-out infinite alternate;
  }
  .hero::after {   /* glossy sheen sweeping across */
    content: ""; position: absolute; top: 0; left: -60%; width: 45%; height: 100%;
    background: linear-gradient(100deg, transparent, rgba(255,255,255,.28), transparent);
    transform: skewX(-18deg); animation: sheen 6s ease-in-out 1.2s infinite;
  }
  .hero::before {  /* top gloss */
    content: ""; position: absolute; inset: 0 0 55% 0; border-radius: 22px 22px 0 0;
    background: linear-gradient(180deg, rgba(255,255,255,.22), transparent); pointer-events: none;
  }
  @keyframes sheen { 0%, 60% { left: -60%; } 100% { left: 130%; } }
  @keyframes hue   { 0% { background-position: 0% 50%; } 100% { background-position: 100% 50%; } }
  .hero .chips { display: flex; flex-wrap: wrap; gap: .45rem; margin-bottom: .9rem; position: relative; z-index: 1; }
  .hero .chip {
    background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.35);
    backdrop-filter: blur(6px); border-radius: 999px; padding: .2rem .8rem;
    font-size: .76rem; letter-spacing: .05em; text-transform: uppercase;
    animation: rise .6s cubic-bezier(.2,.7,.2,1) both;
  }
  .hero .chip:nth-child(2) { animation-delay: .1s; }
  .hero .chip:nth-child(3) { animation-delay: .2s; }
  .hero h1 { color: #fff; font-size: 1.95rem; font-weight: 700; margin: 0 0 .35rem; padding: 0;
    position: relative; z-index: 1; text-shadow: 0 2px 12px rgba(0,0,0,.18); }
  .hero p.sub { color: rgba(255,255,255,.88); margin: 0 0 1rem; font-size: .98rem; position: relative; z-index: 1; }
  .hero .meta { display: flex; flex-wrap: wrap; gap: 1.8rem; font-size: .9rem; position: relative; z-index: 1;
    border-top: 1px solid rgba(255,255,255,.22); padding-top: .8rem; }
  .hero .meta span { color: rgba(255,255,255,.68); display: block; font-size: .7rem;
    text-transform: uppercase; letter-spacing: .07em; }

  /* ---------- metric cards ---------- */
  [data-testid="stMetric"] {
    border-radius: 16px; padding: .9rem 1rem; position: relative; overflow: hidden;
    transition: transform .25s ease, box-shadow .25s ease;
    animation: rise .6s cubic-bezier(.2,.7,.2,1) both;
  }
  [data-testid="stMetric"]::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #2f4fb0, #7c5cff, #38bdf8);
    background-size: 200% 100%; animation: hue 4s linear infinite alternate;
  }
  [data-testid="stMetric"]:hover { transform: translateY(-4px); box-shadow: 0 14px 36px rgba(31,45,110,.2); }
  [data-testid="stColumn"]:nth-child(1) [data-testid="stMetric"] { animation-delay: .05s; }
  [data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"] { animation-delay: .12s; }
  [data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"] { animation-delay: .19s; }
  [data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"] { animation-delay: .26s; }
  [data-testid="stColumn"]:nth-child(5) [data-testid="stMetric"] { animation-delay: .33s; }
  [data-testid="stColumn"]:nth-child(6) [data-testid="stMetric"] { animation-delay: .40s; }
  [data-testid="stMetricLabel"] p { font-size: .76rem; text-transform: uppercase; letter-spacing: .06em; color: #4d5670; }
  [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700;
    background: linear-gradient(135deg, #1e2f6e, #5b4fd6); -webkit-background-clip: text;
    background-clip: text; color: transparent; }

  /* ---------- panels, tables, tabs, buttons ---------- */
  [class*="st-key-panel"] { border-radius: 18px; padding: 1rem 1.2rem .6rem; animation: fade .7s ease .2s backwards; }
  [data-testid="stDataFrame"] { border-radius: 14px; padding: 4px; }
  [data-testid="stExpander"] details { border-radius: 14px; }
  h3 { font-size: 1.05rem !important; font-weight: 650 !important; color: #1b2235; }
  .stTabs [data-baseweb="tab-list"] { gap: .4rem; background: rgba(255,255,255,.35);
    backdrop-filter: blur(12px); padding: .3rem; border-radius: 14px; border: 1px solid rgba(255,255,255,.6); }
  .stTabs [data-baseweb="tab"] { font-weight: 600; border-radius: 10px; padding: .4rem 1rem; transition: background .2s; }
  .stTabs [aria-selected="true"] { background: rgba(255,255,255,.75); box-shadow: 0 2px 10px rgba(31,45,110,.1); }
  .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
  [data-testid="stTabPanel"] { animation: fade .45s ease backwards; }
  .stButton button, .stDownloadButton button {
    background: rgba(255,255,255,.55); backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,.75); border-radius: 12px;
    transition: transform .2s ease, box-shadow .2s ease, background .2s;
  }
  .stButton button:hover, .stDownloadButton button:hover {
    transform: translateY(-2px); background: rgba(255,255,255,.85);
    box-shadow: 0 8px 20px rgba(31,45,110,.15);
  }

  .footer { text-align: center; color: #5b6478; font-size: .8rem; margin-top: 2.5rem; opacity: .8; }

  @keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
  @keyframes fade { from { opacity: 0; } to { opacity: 1; } }

  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
  }
</style>
"""

st.markdown(GLASS_CSS, unsafe_allow_html=True)


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
    with left, st.container(key="panel_cm"):
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
    with right, st.container(key="panel_report"):
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
