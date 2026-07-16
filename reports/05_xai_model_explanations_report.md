# Post-hoc XAI Model Explanation Report

## Scope

This report applies SHAP and LIME to the frozen tuned XGBoost fraud model. It does not retrain the model, change the 0.2757 validation-selected threshold, or reuse explanations for model selection.

## Global SHAP findings

- Fixed training background: 100 rows
- Fixed test reporting sample: 1,000 rows
- SHAP output: uncalibrated fraud risk-score units
- Top global feature by mean absolute SHAP: `TransactionDT`
- SHAP/native-gain Spearman rank correlation: 0.621
- Maximum SHAP reconstruction error: 3.82e-07

| SHAP rank | Feature | Mean absolute SHAP | Normalized Gain |
|---|---|---|---|
| 1 | TransactionDT | 0.03664 | 0.20% |
| 2 | C13 | 0.02640 | 0.54% |
| 3 | C1 | 0.02375 | 0.55% |
| 4 | TransactionAmt | 0.01434 | 0.21% |
| 5 | CardID_Frequency | 0.01184 | 0.22% |
| 6 | card6_credit | 0.01147 | 0.73% |
| 7 | P_emaildomain_Frequency | 0.00999 | 0.23% |
| 8 | card1 | 0.00937 | 0.19% |
| 9 | V69 | 0.00932 | 4.73% |
| 10 | C5 | 0.00931 | 0.67% |

## Local SHAP and LIME comparison

| TransactionID | Outcome | Overlapping features | Top-feature Jaccard | LIME local fidelity R2 |
|---|---|---|---|---|
| 3519397 | True positive | 2 | 11.1% | 0.242 |
| 3524909 | False positive | 2 | 11.1% | 0.105 |
| 3551357 | False negative | 2 | 11.1% | 0.224 |
| 3541077 | True negative | 3 | 17.6% | 0.182 |

Mean LIME local fidelity R² was 0.188; median top-10 SHAP/LIME feature Jaccard overlap was 11.1%. LIME weights must be interpreted in light of case-level fidelity rather than treated as direct XGBoost contributions.

## Required interpretation boundary

SHAP and LIME describe model behavior under different assumptions. Neither method establishes causality or proves fraud. Anonymous and processed features limit semantic interpretation. A human investigator must verify all supporting evidence and follow approved policy.
