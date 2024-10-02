# Loan Approval Prediction - Kaggle Playground Series S04E10

## Download Data


```bash
kaggle competitions download -c playground-series-s4e10
unzip playground-series-s4e10.zip -d data
rm playground-series-s4e10.zip
```

## Data Exploration


- Identified key features: 4 categorical columns (e.g., 'person_home_ownership', 'loan_intent') and 7 numerical columns (e.g., 'person_age', 'loan_amnt') that are likely to influence the target variable 'loan_status'.
- Established 'loan_status' as our target column for prediction, indicating this is a binary classification problem (loan approval or rejection).

NOTE: 0 in the target column, loan_status indicates that the loan is approved. [SOURCE](https://www.kaggle.com/competitions/playground-series-s4e10/discussion/536981)

## Performance Tracker

| Submission File | Date & Time | CV ROC_AUC | Public Score |
|-----------------|-------------|------------|--------------|
| [Autogluon v1](predictions/submission_autogluon_2024-10-02_06-21-21.csv) | 2024-10-02 06:21:21 | 0.98411 | 0.96161 |
| [Undersampling 1:1](predictions/submission_autogluon_2024-10-02_08-56-07.csv) | 2024-10-02 08:56:07 | 0.98477 | 0.96101 |
| [Undersampling 2:1](predictions/submission_autogluon_2024-10-02_08-57-59.csv) | 2024-10-02 08:57:59 | 0.98408 | - |
