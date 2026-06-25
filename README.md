# Bank Marketing Campaign — Classifier Comparison

Jupiter notebook: https://github.com/afauci/bank-term-deposits/blob/main/prompt_III.ipynb

## Overview
Analysis of a Portuguese bank's direct marketing campaign data to predict whether a client will subscribe to a term deposit. The goal is to identify which clients are most likely to subscribe, allowing the bank to direct marketing resources more efficiently.

---

## Business Understanding
The bank runs telephone marketing campaigns to sell term deposit products. The business objective is to build a model that predicts which clients are likely to subscribe, so agents can prioritize outreach and reduce wasted contacts while maintaining conversion rates.

The dataset represents 17 campaigns run between May 2008 and November 2010, covering ~41,000 contacts from a Portuguese bank.

---

## Data Preparation
The dataset required several preprocessing steps:

- Dropped `duration`: call duration is only known after a call ends, making it unusable for pre-call predictions (data leakage)
- Dropped `default`: only 3 "yes" values out of 41,000 rows — no usable signal
- `housing` and `loan`: ~2.4% unknown values collapsed to "no", then binary encoded
- `pdays`: 95% of values were 999 (no prior contact). Converted to 
  a binary `has_been_contacted` flag
- `education`: ordinal encoded (illiterate=0 through university=6)
- `contact`: binary encoded (cellular=1, telephone=0)
- `marital`, `job`, `poutcome`, `day_of_week`, `month`: one-hot 
  encoded; rows with "unknown" in marital/job/education dropped (<1% each)
- Features scaled with `StandardScaler` before modeling

Final dataset: 39,191 rows × 40 features

---

## Key Challenge: Class Imbalance
The target variable is heavily imbalanced: ~89% "no", ~11% "yes". A naive model that always predicts "no" achieves 89% accuracy while being completely useless. For this reason, ROC-AUC, recall, and F1 on the "yes" class were important evaluation metrics, since we want to optimize for true positives, and if we have some false positives that's better than missing potential subscribers.

---

## Models Compared
Four classifiers were evaluated with default settings:
- Logistic Regression
- K Nearest Neighbor
- Decision Tree
- Support Vector Machine

Key observations:
- Decision Tree (99% train, 84% test) showed clear overfitting in default state
- SVM achieved the highest accuracy but was significantly slower than the otherost
- KNN trains instantly but scoring takes 22 seconds due to distance calculations at predict time
- Logistic Regression matched SVM accuracy at a fraction of the cost
- No models acheived higher than 91% accuracy

---

## Hyperparameter Tuning
Grid search (5-fold CV, scoring on ROC-AUC) was applied to LR, KNN, and Decision Tree. SVM was excluded due to training time constraints - LinearSVC was evaluated as a faster alternative.

Important finding: adding `class_weight='balanced'` to Logistic Regression dropped overall accuracy from 90% to 82%, but improved recall on the "yes" class from 23% to 63% and ROC-AUC from 0.60 to 0.74. For the business objective of finding subscribers, this is a significantly more useful model.

---

## Feature Importance

Both models agree that macroeconomic conditions matter more than individual client characteristics:

Top positive predictors (increase subscription likelihood):
- `cons.price.idx` — higher consumer prices correlate with more subscriptions
- `contact` (cellular) — cellular outreach is more effective than telephone
- `has_been_contacted` — prior campaign contact increases likelihood
- `poutcome_success` — previous campaign success is a strong signal
- `month_mar` — March contacts convert well
- `job_retired` — retired clients are more likely to subscribe

Top negative predictors (decrease subscription likelihood):
- `emp.var.rate` — strongest negative signal; economic instability reduces subscriptions
- `nr.employed` — dominated the Decision Tree (64% of splits); high employment correlates with lower subscription rates
- `euribor3m` — higher interest rates elsewhere make it less likely people will use term deposits
- `month_may` — worst converting month despite being most common contact month
- `campaign` — more contacts in this campaign reduces likelihood, so repeated calling is counterproductive

---

## Findings & Recommendations

### For a non-technical audience:
The model identifies clients most likely to open a term deposit account. Rather than calling everyone, the bank can focus on clients who match the high-likelihood profile, potentially achieving the same number of subscriptions with significantly fewer calls.

### Actionable recommendations:
1. Time campaigns to economic conditions — the model's strongest signals are macroeconomic. Campaigns should run when employment variation is low and consumer confidence is high to see higher conversion rates
2. Avoid May, prioritize March — May is the most common contact month but one of the worst converting. March, June, September, and December perform better
3. Prefer cellular contact — cellular outreach consistently outperforms telephone contact
4. Limit repeated contacts — more calls to the same client in a single campaign reduces conversion likelihood, so prioritize new contacts
5. Target retired clients — this segment shows consistently higher subscription rates

Use the balanced model in production — the `class_weight='balanced'` Logistic Regression finds 63% of actual subscribers vs 23% for the default model, at the cost of some additional false positives. 

### Next Steps:
- Investigate the economic indicator features more deeply — if campaigns can be timed to favorable economic windows, this may be more impactful than client targeting
- Consider adjusting the classification threshold (currently 0.5) to further tune the precision/recall tradeoff based on the relative cost of a missed subscriber vs a wasted call

