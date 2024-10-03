# Handling Class Imbalance

## Hypothesis
**Handling Class Imbalance Improves Model Robustness**

## Reasoning
![Loan status distribution](../viz/loan_status_distribution.png)
Looking at the loan status distribution, we can see that the data is skewed towards the '0' class (loan approved). There are a lot more 0s than 1s. Just 14% of the loans are not approved.

## Baseline
ROC/AUC score: 0.98411
```
Classification Report:
                precision    recall  f1-score   support

Approved Loans       0.96      0.99      0.98     50295
Rejected Loans       0.95      0.77      0.85      8350

      accuracy                           0.96     58645
     macro avg       0.96      0.88      0.92     58645
  weighted avg       0.96      0.96      0.96     58645


Conclusion:
The model is performing better on Class 0 (Approved Loans).
```

### Experiment Design:
- **Analyze Class Distribution:** Examine the distribution of the `loan_status` target variable to identify any imbalance.
- **Apply Resampling Techniques:** Implement oversampling (e.g., SMOTE) or undersampling methods to balance the classes.
- **Integrate into Preprocessing:** Modify the `data_preprocessing` function to include resampling steps.
- **Evaluate Performance:** Assess whether balancing the classes leads to improved ROC/AUC scores.

## Sanity check and expected increase in performance
For class imbalance being the reason for bad performance, the trained model should perform better on training data points with label 0 than label 1. Sanity check confirmed that the model is performing better on the training data points with label 0 than label 1. 

## Experiment - Undersampling:
We will undersample the training data points with label 0 in different ratios and check the performance. We will define new hyper-parameter sampling_ratio = len(df_majority)/len(df_minority).

### Ratio of 1:1
ROC_AUC: 0.98477
```
Classification report:
               precision    recall  f1-score   support

           0       0.99      0.93      0.96     50295
           1       0.70      0.94      0.80      8350

    accuracy                           0.93     58645
   macro avg       0.85      0.94      0.88     58645
weighted avg       0.95      0.93      0.94     58645
```
marginal increase in ROC_AUC score, but the classification report suggests otherwise.

### Ratio of 2:1
ROC_AUC: 0.98408
```
Classification report:
               precision    recall  f1-score   support

           0       0.98      0.97      0.98     50295
           1       0.84      0.86      0.85      8350

    accuracy                           0.96     58645
   macro avg       0.91      0.92      0.91     58645
weighted avg       0.96      0.96      0.96     58645
```

## Observations
Undersampling the majority class (0) does not seem to be a good idea. It is not only leading to a decrease in ROC_AUC score (very lil increase when 1:1 ratio sampled), but also the classification report suggests that the model is performing worse on the training data points with label 0 than label 1. Maybe let us try oversampling the minority class (1) instead.

## Experiment - Oversampling:
We will oversample the training data points with label 1 in different ratios and check the performance. We will define new hyper-parameter sampling_ratio = len(df_minority)/len(df_majority).

### Ratio of 1:1
ROC_AUC score: 1.00000
```
Classification report:
               precision    recall  f1-score   support
           0       1.00      1.00      1.00     50295
           1       0.99      1.00      0.99      8350
    accuracy                           1.00     58645
   macro avg       0.99      1.00      1.00     58645
weighted avg       1.00      1.00      1.00     58645
```

The results are 100% everywhere this is fishy, maybe the model is overfitting to the training dataset.