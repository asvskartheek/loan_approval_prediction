# Handling Class Imbalance

## Hypothesis
**Handling Class Imbalance Improves Model Robustness**

## Reasoning
![Loan status distribution](../viz/loan_status_distribution.png)
Looking at the loan status distribution, we can see that the data is skewed towards the '0' class (loan approved). There are a lot more 0s than 1s. Just 14% of the loans are not approved.

### Experiment Design:
- **Analyze Class Distribution:** Examine the distribution of the `loan_status` target variable to identify any imbalance.
- **Apply Resampling Techniques:** Implement oversampling (e.g., SMOTE) or undersampling methods to balance the classes.
- **Integrate into Preprocessing:** Modify the `data_preprocessing` function to include resampling steps.
- **Evaluate Performance:** Assess whether balancing the classes leads to improved ROC/AUC scores.

## Sanity check and expected increase in performance
For class imbalance being the reason for bad performance, the trained model should perform better on training data points with label 0 than label 1. Classification report:
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

Sanity check confirmed that the model is performing better on the training data points with label 0 than label 1. 

## Experiment 1:

We will undersample the training data points with label 0 to balance the classes.