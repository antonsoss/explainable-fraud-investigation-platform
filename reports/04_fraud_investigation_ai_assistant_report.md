# Fraud Investigation and AI Assistant Report

## Scope

This report interprets the frozen XGBoost fraud model at a global level, examines test-set errors, retrieves descriptively similar validation cases, and produces evidence-grounded investigation summaries. The assistant supports human review and does not make autonomous decisions.

## Model context

- Champion model: XGBoost
- Frozen alert threshold: 0.7034
- Test average precision: 0.7119
- Test ROC-AUC: 0.9511
- Test precision: 0.5512
- Test recall: 0.7160

## Top native XGBoost features by normalized gain

| Gain Rank | Feature | Normalized Gain |
|---|---|---|
| 1 | V258 | 13.70% |
| 2 | V201 | 8.39% |
| 3 | V264 | 5.21% |
| 4 | V69 | 4.17% |
| 5 | ProductCD_C | 3.61% |
| 6 | V90 | 2.48% |
| 7 | V294 | 1.94% |
| 8 | ProductCD_W | 1.89% |
| 9 | V187 | 1.44% |
| 10 | card6_credit | 1.02% |

These values describe training-time split improvements across the ensemble. They do not establish causality or explain one transaction.

## Test prediction outcomes

| Outcome | Transactions | Share |
|---|---|---|
| True positive | 2,219 | 2.51% |
| False positive | 1,807 | 2.04% |
| False negative | 880 | 0.99% |
| True negative | 83,675 | 94.46% |

## Similar-case retrieval

The TF-IDF reference library contains 88,581 validation transactions represented by 12,000 text features. Labels and model outputs were excluded from the retrieval text.

## Assistant controls

- Uses a restricted evidence packet.
- Does not receive retrospective ground truth by default.
- Cannot declare fraud or make approval, rejection, or enforcement decisions.
- Must distinguish global importance from individual model reasoning.
- Requires human verification.

## Case 3299850 (Deterministic)

### Investigation summary: Transaction 3299850

**Transaction overview**
- Amount: $126.88 (75.4% percentile in the available dataset).
- Product: C; card network/type: mastercard / credit.
- Device: desktop; operating system: missing; browser: chrome 63.0.

**Model signal**
- Fraud probability: 100.0%.
- Alert threshold: 70.3%; the score is above the threshold.
- Global feature importance describes the model overall and does not identify the cause of this transaction's score.

**Similar historical cases**
- 3 of the 5 retrieved cases were labelled as fraud.
- Similarity ranged from 0.695 to 0.816.
- Similar-case outcomes are contextual evidence, not a probability estimate.

**Suggested human review checks**
- Verify cardholder authorization and account activity near the transaction time.
- Review device, browser, email-domain, and product consistency against trusted history.
- Compare the transaction with the retrieved cases and document material similarities and differences.
- Escalate only according to the institution's approved investigation policy.

**Limitations**
- The model and retrieval index may reproduce historical data limitations.
- Missing or encoded attributes reduce semantic interpretation.
- This output supports human review and is not proof of fraud or an autonomous decision.
## Case 3397044 (Deterministic)

### Investigation summary: Transaction 3397044

**Transaction overview**
- Amount: $114.04 (69.5% percentile in the available dataset).
- Product: C; card network/type: mastercard / debit.
- Device: desktop; operating system: missing; browser: chrome 65.0.

**Model signal**
- Fraud probability: 99.9%.
- Alert threshold: 70.3%; the score is above the threshold.
- Global feature importance describes the model overall and does not identify the cause of this transaction's score.

**Similar historical cases**
- 0 of the 5 retrieved cases were labelled as fraud.
- Similarity ranged from 0.672 to 0.852.
- Similar-case outcomes are contextual evidence, not a probability estimate.

**Suggested human review checks**
- Verify cardholder authorization and account activity near the transaction time.
- Review device, browser, email-domain, and product consistency against trusted history.
- Compare the transaction with the retrieved cases and document material similarities and differences.
- Escalate only according to the institution's approved investigation policy.

**Limitations**
- The model and retrieval index may reproduce historical data limitations.
- Missing or encoded attributes reduce semantic interpretation.
- This output supports human review and is not proof of fraud or an autonomous decision.
## Case 3379834 (Deterministic)

### Investigation summary: Transaction 3379834

**Transaction overview**
- Amount: $54.50 (36.5% percentile in the available dataset).
- Product: W; card network/type: visa / debit.
- Device: missing; operating system: missing; browser: missing.

**Model signal**
- Fraud probability: 1.4%.
- Alert threshold: 70.3%; the score is below the threshold.
- Global feature importance describes the model overall and does not identify the cause of this transaction's score.

**Similar historical cases**
- 0 of the 5 retrieved cases were labelled as fraud.
- Similarity ranged from 0.556 to 0.644.
- Similar-case outcomes are contextual evidence, not a probability estimate.

**Suggested human review checks**
- Verify cardholder authorization and account activity near the transaction time.
- Review device, browser, email-domain, and product consistency against trusted history.
- Compare the transaction with the retrieved cases and document material similarities and differences.
- Escalate only according to the institution's approved investigation policy.

**Limitations**
- The model and retrieval index may reproduce historical data limitations.
- Missing or encoded attributes reduce semantic interpretation.
- This output supports human review and is not proof of fraud or an autonomous decision.
## Case 3427680 (Deterministic)

### Investigation summary: Transaction 3427680

**Transaction overview**
- Amount: $25.21 (8.6% percentile in the available dataset).
- Product: C; card network/type: visa / credit.
- Device: mobile; operating system: missing; browser: chrome 65.0 for android.

**Model signal**
- Fraud probability: 70.3%.
- Alert threshold: 70.3%; the score is below the threshold.
- Global feature importance describes the model overall and does not identify the cause of this transaction's score.

**Similar historical cases**
- 0 of the 5 retrieved cases were labelled as fraud.
- Similarity ranged from 0.562 to 0.624.
- Similar-case outcomes are contextual evidence, not a probability estimate.

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
- Similarity reflects the chosen tokens and does not estimate fraud probability.
- Native feature importance can favor frequently split or high-cardinality features.
- Anonymous and encoded variables limit semantic interpretation.
- Automated grounding checks do not replace manual review of generated text.