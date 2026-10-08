# Alert Escalation Scoring Model for Bank Transaction Monitoring

Machine learning pipeline that ranks bank transaction-monitoring alerts by their probability of escalation, so analysts can review the riskiest cases first.

**Finalist solution** (one of 20 teams, FinTech / AI in Finance track) at the **WIUT Inter-University Hackathon 2026 "Intelligence in Use"**, by team **DebugMaster** (Customs Institute).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1K_MGhSoFG5ZecJJcev85nXrjVRA_-7QU)
&nbsp; **[Project website (EDA and modelling report)](https://enchanting-sunflower-7fe95c.netlify.app)**

## Results

| Model | Details | CV ROC-AUC (15 folds) |
|---|---|---|
| Logistic regression | 65 features, L2, C chosen by inner CV | 0.6453 ± 0.0116 |
| LightGBM | depth 3, strong regularisation, early stopping | 0.6400 ± 0.0118 |
| CatBoost | depth 3, strong regularisation, early stopping | 0.6391 ± 0.0108 |
| **Ensemble (LR + LightGBM + CatBoost)** | rank-average 50 / 25 / 25, weights fixed in advance | **0.6478 ± 0.0105** |

The signal in this dataset is weak and close to linear-additive. A small, carefully validated model captured almost all of it, so we put our effort into reliable validation rather than model complexity.

## The problem

Each alert comes with about 180 days of the customer's transaction history. The task is to estimate the probability that an analyst escalates the alert. Submissions are scored by ROC-AUC.

| | Train | Test |
|---|---|---|
| Alerts | 14,000 | 6,000 |
| Transactions | 6,987,663 | 3,027,575 |
| Escalation rate | 17.18% | hidden |

Each transaction has a timestamp, a direction (inflow / outflow), a type (card, bank transfer, cash, international) and a standardised amount index. The data is synthetic and was provided by the organisers.

## Key findings

- **The signal sits in amount levels by transaction type and direction.** Higher bank-transfer amounts go with fewer escalations; higher card outflows and cash inflows go with more.
- **The pre-alert burst does not discriminate.** Activity in the last day before an alert is 13.6× the usual rate for escalated and non-escalated alerts alike. It is what triggers the alert, not what decides escalation.
- **No single strong feature.** The best single aggregate reaches only AUC 0.548.
- **"Floor" bank inflows.** 5,117 transactions sit at the minimum amount, and all of them are bank-transfer inflows. Alerts containing one escalate at 10.0% versus 18.2% for the rest.
- **Train and test are exchangeable.** A model trying to tell them apart scores AUC 0.499, so stratified cross-validation is a trustworthy estimate.

![Daily activity before the alert, by class](https://enchanting-sunflower-7fe95c.netlify.app/assets/08_recent_activity.png)

![Univariate AUC of signal-level aggregates](https://enchanting-sunflower-7fe95c.netlify.app/assets/09_univariate_auc.png)

## Approach

1. **Anchor rule.** Each alert's history is measured back from `max(alert date 00:00, end of the last transaction's day)`. No transaction exists after the anchor, which rules out leakage from the future.
2. **Features.** One function builds 65 features for both train and test, with no use of the target:

   | Group | Count | Content |
   |---|---|---|
   | General aggregates | 27 | volume, type and direction shares, amount statistics, history length |
   | Amount by type | 32 | per-type min, max, std, median, percentiles; mean per type × direction |
   | Floor | 6 | transactions at the minimum amount |

3. **Validation.** Repeated stratified K-fold (5 folds × 3 repeats, seed 42), the same 15 folds for every model. Changes are compared fold by fold and accepted only if they gain at least 0.003 AUC, win in at least 12 of 15 folds and keep the same sign in all 3 repeats.
4. **Models.** Logistic regression, LightGBM and CatBoost on the same folds.
5. **Ensemble.** In-fold rank-average with weights fixed in advance (no weight tuning), then quantile-mapped onto the logistic-regression prediction distribution to keep probabilities calibrated.

## A leak we caught in our own pipeline

Selecting 65 features with L1 regularisation on the full training set and then cross-validating gave **0.6548** and "won" in 15 of 15 folds. Repeating the selection inside each fold gave **0.6441**. The +0.0107 was pure selection optimism. We discarded that feature list and made in-fold selection mandatory.

## What we tried and did not keep

Each idea was added on top of the 65 features and compared fold by fold. None passed the acceptance rule.

| Idea | Extra features | Δ AUC vs baseline | Folds better |
|---|---|---|---|
| Type × direction statistics | +126 | −0.0006 | 6 / 15 |
| Per-type and per-direction statistics | +102 | −0.0014 | 3 / 15 |
| Long history: daily series, trends | +48 | −0.0030 | 4 / 15 |
| Burst features (last ~3 minutes) | +5 | +0.0007 | 11 / 15 |
| Amount histograms, 12 bins | +336 | −0.0055 | 3 / 15 |
| Splines on the top-20 features | +140 | −0.0010 | 9 / 15 |
| Pairwise products of the top-10 features | +45 | −0.0009 | 7 / 15 |

In total, 380+ candidate features were built and tested.

## Repository structure

```
notebooks/team_2CC9AA36.ipynb   Standalone notebook: raw data → features → CV → ensemble → predictions
site/                           Source of the project website (static HTML, 10 charts)
requirements.txt                Python dependencies
```

## How to reproduce

The notebook is self-contained and runs top to bottom in about 10–20 minutes on a Google Colab CPU.

1. Open the notebook in Colab (badge above) or locally after `pip install -r requirements.txt`.
2. Place the competition files under `data/raw/` next to the notebook:
   `train_signals.csv`, `test_signals.csv`, `sample_submission.csv`, `train_transactions.parquet`, `test_transactions.parquet`.
3. Run all cells. Expected cross-validation scores: LR 0.6453, LightGBM 0.6400, CatBoost 0.6391, ensemble 0.6478.

The notebook was re-run in a clean Colab environment from the raw files; it reproduced these scores and produced a file identical to our submitted predictions.

**Data.** The dataset belongs to the hackathon organisers and is not included in this repository.

**Language.** Column names in the data and comments in the notebook are in Uzbek. The website and this README are in English.

## Limitations

- With 6,000 test alerts, the standard error of a test AUC near 0.65 is about 0.010. The ensemble's gain over logistic regression (+0.0025) is inside that noise, although it is positive in all three CV repeats.
- A sequence model on the raw transactions was prepared but not trained.
- Ensemble weights were fixed in advance rather than tuned with nested cross-validation.

## Team DebugMaster

- Muhammadali Yuldoshov
- Oyatillo Abdulazizov 
- Behro'z Sa'dullayev

## Tech stack

Python, pandas, NumPy, SciPy, scikit-learn, LightGBM, CatBoost, Matplotlib, Google Colab.
