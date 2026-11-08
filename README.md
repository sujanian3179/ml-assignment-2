# Bank Term-Deposit Subscription — ML Classification & Streamlit App

**Live app:** _add Streamlit Cloud link here_
**Repository:** https://github.com/sujanian3179/ml-assignment-2

## a. Problem statement

A Portuguese bank runs telephone marketing campaigns to sell term deposits. Calling every
client is expensive and most say no. The goal is to **predict whether a client will
subscribe to a term deposit (`y` = yes/no)** from their demographic, financial and
campaign-contact attributes, so the bank can prioritise the clients most likely to convert.
This is a **binary classification** problem. Five classifiers are trained, compared on six
evaluation metrics, and served through an interactive Streamlit app.

## b. Dataset description

| Item | Detail |
|---|---|
| Source | [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) (`bank.csv`) |
| Instances | 4,521 |
| Features used | 15 (6 numeric, 9 categorical) |
| Target | `y` — did the client subscribe? (`yes` 521 / `no` 4,000 → **11.5 % positive, imbalanced**) |
| Train / test split | 80 / 20, stratified, `random_state=42` → 3,616 train, 905 test (`test_data.csv`) |

**Features**

- **Numeric:** `age`, `balance` (avg. yearly balance, €), `day` (last contact day of month), `campaign` (contacts in this campaign), `pdays` (days since last contact in a previous campaign, -1 = never), `previous` (contacts before this campaign)
- **Categorical:** `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome` (outcome of previous campaign)

**Dropped:** `duration` (length of the last call). It is only known *after* the call has
ended, so it leaks the outcome. UCI recommends discarding it for a realistic model.

**Preprocessing** (inside each saved scikit-learn `Pipeline`): `StandardScaler` on numeric
columns, `OneHotEncoder(handle_unknown="ignore")` on categorical columns. Because of the
class imbalance, `class_weight="balanced"` is used for Logistic Regression, Decision Tree
and SVM (KNN and Gaussian NB do not support it).

## c. Models used

All five models are trained on the same training split and evaluated on the same held-out
test split (905 rows).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7050 | 0.7350 | 0.2237 | 0.6346 | 0.3308 | 0.2373 |
| Decision Tree | 0.7657 | 0.7024 | 0.2404 | 0.4808 | 0.3205 | 0.2149 |
| KNN | **0.8829** | 0.6450 | **0.4167** | 0.0481 | 0.0862 | 0.1097 |
| Naive Bayes | 0.8099 | 0.7166 | 0.2763 | 0.4038 | 0.3281 | 0.2274 |
| Support Vector Machine | 0.7901 | **0.7502** | 0.2913 | 0.5769 | **0.3871** | **0.3002** |

Hyper-parameters: LR `max_iter=2000`; DT `max_depth=6, min_samples_leaf=20`; KNN `k=7`;
Gaussian NB defaults; SVM RBF kernel, `C=1.0`, `probability=True`.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest recall (0.63): with balanced class weights it catches most subscribers, but at the cost of many false positives (precision 0.22) and the lowest accuracy (0.71). Good AUC (0.735) shows the features carry a mostly linear signal, the strongest positive coefficients are `poutcome=success` and contacts in Oct/Mar, while Jan/Nov contacts and `contact=unknown` push towards "no". |
| Decision Tree | Depth-limited (6) to avoid overfitting. Middling recall (0.48) and the second-lowest AUC (0.70): its step-wise probabilities rank clients less smoothly than the other models. Easy to interpret but weaker overall than LR or SVM. |
| KNN | Highest accuracy (0.88) and precision, but this is misleading: recall is only 0.05, so it predicts "no" for almost everyone, which is close to the 88.5 % majority-class baseline. Lowest AUC, F1 and MCC. Distance-based voting with no class weighting is swamped by the majority class and by many one-hot dimensions. |
| Naive Bayes | Fast and reasonably balanced (accuracy 0.81, F1 0.33, AUC 0.72) despite the independence assumption being violated (e.g. `pdays`, `previous` and `poutcome` are strongly correlated). Gaussian likelihoods on one-hot columns are crude, which caps its performance. |
| Support Vector Machine | Best overall: highest AUC (0.75), F1 (0.39) and MCC (0.30). The RBF kernel captures non-linear interactions, and class weighting keeps recall high (0.58) while precision stays the best among the high-recall models. Slowest to train, but fine at this data size. |
| **Overall winner** | **Support Vector Machine.** On an imbalanced dataset accuracy is not informative (KNN "wins" accuracy by mostly predicting "no"). On the balanced metrics (MCC, F1) and on ranking quality (AUC), SVM is best. Logistic Regression is a close, more interpretable second if catching subscribers (recall) matters most. |

## Streamlit app features

- CSV upload for test data (plus a bundled sample and a download button for `test_data.csv`)
- Model selection dropdown (5 models)
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion matrix heat-map and classification report
- Comparison tab: all five models on the uploaded data, with a bar chart per metric
- Download predictions as CSV

The app only **loads the saved models** from `model/`. It never retrains on uploaded data.

## Project structure

```
├── app.py               # Streamlit app
├── train_model.py       # training + evaluation script
├── requirements.txt
├── README.md
├── test_data.csv        # held-out test split (features + target)
├── data/bank.csv        # full dataset (UCI)
└── model/               # saved Pipelines (.joblib) + metrics.csv
```

## Run locally

```bash
pip install -r requirements.txt
python train_model.py     # optional: re-creates model/ and test_data.csv
streamlit run app.py
```
