# Fraud Investigation and AI Assistant Report

## Scope

This report interprets the frozen champion fraud model at a global level, examines later-period test errors, retrieves descriptively similar labeled validation reference cases, and produces evidence-grounded investigation summaries. The assistant supports human review and does not make autonomous decisions.

## Model context

- Champion model: XGBoost (tuned)
- Frozen risk-score threshold: 0.2757
- Score semantics: uncalibrated ranking score, not a fraud probability
- Test average precision: 0.5213
- Test ROC-AUC: 0.8996
- Test precision: 0.3258
- Test recall: 0.6160

## Top native model features by normalized gain

| Gain Rank | Feature | Normalized Gain |
|---|---|---|
| 1 | V258 | 19.79% |
| 2 | V69 | 4.73% |
| 3 | V318 | 3.10% |
| 4 | V201 | 2.48% |
| 5 | V90 | 1.65% |
| 6 | V187 | 1.62% |
| 7 | V29 | 1.41% |
| 8 | V283 | 1.05% |
| 9 | ProductCD_W | 0.85% |
| 10 | V102 | 0.81% |

These values describe training-time split improvements across the ensemble. They do not establish causality or explain one transaction.

## Test prediction outcomes

| Outcome | Transactions | Share |
|---|---|---|
| True positive | 1,899 | 2.14% |
| False positive | 3,930 | 4.44% |
| False negative | 1,184 | 1.34% |
| True negative | 81,568 | 92.08% |

## Cohort warning

The lowest product-level recall is W at 36.3%. The lowest device-level recall is missing at 37.2%. These gaps require investigation before any cohort-specific operating policy is considered.

## Similar-case retrieval

The TF-IDF reference library contains 88,581 labeled validation transactions represented by 12,000 text features. It is a reference set, not a database of completed investigations. Labels and model outputs were excluded from retrieval text.
On a balanced retrospective sample of 500 test queries, fraud queries retrieved 15.8% fraud-labeled neighbors versus a 3.4% reference prevalence (4.59x lift).

## Assistant controls

- Uses a restricted evidence packet.
- Does not receive retrospective query ground truth by default.
- Describes model output as an uncalibrated risk score.
- Cannot declare fraud or make approval, rejection, or enforcement decisions.
- Must distinguish global importance from individual model reasoning.
- Requires human verification.

## Case 3519397 (Deterministic)

### Investigation summary: Transaction 3519397

**Transaction overview**
- Amount: $262.40 (89.7% percentile in the available dataset).
- Product: C; card network/type: visa / debit.
- Device: desktop; operating system: missing; browser: chrome 66.0.

**Model signal**
- Fraud risk score: 88.80%.
- Alert threshold: 27.57%; the score is at or above the threshold.
- The score ranks fraud risk and is not a calibrated probability.
- Global feature importance describes the model overall and does not identify the cause of this transaction's score.

**Similar reference cases**
- 1 of the 5 retrieved reference cases were labelled as fraud.
- Similarity ranged from 0.489 to 0.608.
- Reference-case outcomes are contextual evidence, not a probability estimate.

**Suggested human review checks**
- Verify cardholder authorization and account activity near the transaction time.
- Review device, browser, email-domain, and product consistency against trusted history.
- Compare the transaction with the retrieved cases and document material similarities and differences.
- Escalate only according to the institution's approved investigation policy.

**Limitations**
- The model and retrieval index may reproduce historical data limitations.
- Missing or encoded attributes reduce semantic interpretation.
- This output supports human review and is not proof of fraud or an autonomous decision.
## Case 3524909 (Deterministic)

### Investigation summary: Transaction 3524909

**Transaction overview**
- Amount: $87.00 (55.6% percentile in the available dataset).
- Product: W; card network/type: mastercard / debit.
- Device: missing; operating system: missing; browser: missing.

**Model signal**
- Fraud risk score: 41.69%.
- Alert threshold: 27.57%; the score is at or above the threshold.
- The score ranks fraud risk and is not a calibrated probability.
- Global feature importance describes the model overall and does not identify the cause of this transaction's score.

**Similar reference cases**
- 0 of the 5 retrieved reference cases were labelled as fraud.
- Similarity ranged from 0.686 to 0.801.
- Reference-case outcomes are contextual evidence, not a probability estimate.

**Suggested human review checks**
- Verify cardholder authorization and account activity near the transaction time.
- Review device, browser, email-domain, and product consistency against trusted history.
- Compare the transaction with the retrieved cases and document material similarities and differences.
- Escalate only according to the institution's approved investigation policy.

**Limitations**
- The model and retrieval index may reproduce historical data limitations.
- Missing or encoded attributes reduce semantic interpretation.
- This output supports human review and is not proof of fraud or an autonomous decision.
## Case 3551357 (Deterministic)

### Investigation summary: Transaction 3551357

**Transaction overview**
- Amount: $390.00 (94.1% percentile in the available dataset).
- Product: W; card network/type: visa / debit.
- Device: missing; operating system: missing; browser: missing.

**Model signal**
- Fraud risk score: 9.93%.
- Alert threshold: 27.57%; the score is below the threshold.
- The score ranks fraud risk and is not a calibrated probability.
- Global feature importance describes the model overall and does not identify the cause of this transaction's score.

**Similar reference cases**
- 0 of the 5 retrieved reference cases were labelled as fraud.
- Similarity ranged from 0.719 to 0.733.
- Reference-case outcomes are contextual evidence, not a probability estimate.

**Suggested human review checks**
- Verify cardholder authorization and account activity near the transaction time.
- Review device, browser, email-domain, and product consistency against trusted history.
- Compare the transaction with the retrieved cases and document material similarities and differences.
- Escalate only according to the institution's approved investigation policy.

**Limitations**
- The model and retrieval index may reproduce historical data limitations.
- Missing or encoded attributes reduce semantic interpretation.
- This output supports human review and is not proof of fraud or an autonomous decision.
## Case 3541077 (Deterministic)

### Investigation summary: Transaction 3541077

**Transaction overview**
- Amount: $57.95 (39.6% percentile in the available dataset).
- Product: W; card network/type: visa / debit.
- Device: missing; operating system: missing; browser: missing.

**Model signal**
- Fraud risk score: 1.92%.
- Alert threshold: 27.57%; the score is below the threshold.
- The score ranks fraud risk and is not a calibrated probability.
- Global feature importance describes the model overall and does not identify the cause of this transaction's score.

**Similar reference cases**
- 0 of the 5 retrieved reference cases were labelled as fraud.
- Similarity ranged from 0.652 to 0.689.
- Reference-case outcomes are contextual evidence, not a probability estimate.

**Suggested human review checks**
- Verify cardholder authorization and account activity near the transaction time.
- Review device, browser, email-domain, and product consistency against trusted history.
- Compare the transaction with the retrieved cases and document material similarities and differences.
- Escalate only according to the institution's approved investigation policy.

**Limitations**
- The model and retrieval index may reproduce historical data limitations.
- Missing or encoded attributes reduce semantic interpretation.
- This output supports human review and is not proof of fraud or an autonomous decision.

## Limitations

- IEEE-CIS contains structured fields rather than authentic investigation narratives.
- Similarity reflects chosen tokens and does not estimate fraud probability.
- Native feature importance can favor frequently split or high-cardinality features.
- Anonymous and encoded variables limit semantic interpretation.
- Test-set analysis is reporting-only and must not drive further model selection.
- Automated grounding checks do not replace manual review of generated text.