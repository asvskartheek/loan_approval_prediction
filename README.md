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