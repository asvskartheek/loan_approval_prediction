"""
This script trains an AutoGluon model for loan approval prediction.

It performs the following main steps:
1. Data preprocessing (including class balancing)
2. Model training using AutoGluon's TabularPredictor
3. Model evaluation
4. Generating predictions for submission

The script uses predefined constants for data paths, column names, and model parameters.
Logging is implemented to capture all console output in a timestamped log file.
"""

from datetime import datetime
import pandas as pd
from autogluon.tabular import TabularPredictor
import logging
from sklearn.utils import resample

# Constants
DATA_DIR = '/Users/asvs/kartheek_hobby_projects/loan_approval_prediction/data'

CATEGORICAL_COLS = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
NUMERICAL_COLS = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
TARGET_COL = 'loan_status'

EVAL_METRIC = 'roc_auc'
TIME_LIMIT = 10*60 # 10 minutes
TIME_STAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
LOG_FILE = f"logs/train_autogluon_{TIME_STAMP}.log"

# Set up logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def data_preprocessing(df, test=False):
    """
    Preprocess the input DataFrame for AutoGluon training or prediction.

    Args:
        df (pandas.DataFrame): The input DataFrame to preprocess.
        test (bool): Flag indicating whether this is test data (default: False).

    Returns:
        pandas.DataFrame: The preprocessed DataFrame.

    This function performs the following preprocessing steps:
    1. Creates a copy of the input DataFrame to avoid modifying the original.
    2. Drops the 'id' column as it's not required for training or prediction.
    3. Converts categorical columns to 'category' dtype for AutoGluon compatibility.
    4. For training data (test=False), converts the target column to integer type.
    5. For training data, applies undersampling to balance the classes.
    """
    df = df.copy()
    df = df.drop(columns=['id']) # dropping id column as it is not required for training
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype('category') # type casting categorical columns to category, necessary for AutoGluon
    if not test:
        df[TARGET_COL] = df[TARGET_COL].astype('int') # type casting target column to int.
        
        # Undersampling the majority class (0)
        sampling_ratio = 2.0 # 2:1
        df_majority = df[df[TARGET_COL] == 0]
        df_minority = df[df[TARGET_COL] == 1]
        df_majority_downsampled = resample(df_majority, 
                                           replace=False,
                                           n_samples=int(len(df_minority)*sampling_ratio),
                                           random_state=42)
        df = pd.concat([df_majority_downsampled, df_minority])
        
        logger.info(f"Class distribution after balancing: {df[TARGET_COL].value_counts()}")

    return df

def train_model(df_train):
    """
    Train an AutoGluon TabularPredictor model on the preprocessed training data.

    Args:
        df_train (pandas.DataFrame): The preprocessed training DataFrame.

    Returns:
        autogluon.tabular.TabularPredictor: The trained AutoGluon model.

    This function initializes and trains an AutoGluon TabularPredictor with the following parameters:
    - Label column: Specified by TARGET_COL constant
    - Evaluation metric: Specified by EVAL_METRIC constant
    - Time limit: Specified by TIME_LIMIT constant
    - Verbosity: Set to 2 for detailed output
    """
    predictor = TabularPredictor(label=TARGET_COL, eval_metric=EVAL_METRIC, log_to_file=True, log_file_path=LOG_FILE).fit(
        df_train,
        time_limit=TIME_LIMIT,
        verbosity=2
    )

    return predictor

def evaluate_predictor(predictor, df_train):
    """
    Evaluate the trained AutoGluon model, log performance metrics, and save predictions.

    Args:
        predictor (autogluon.tabular.TabularPredictor): The trained AutoGluon model.
        df_train (pandas.DataFrame): The training DataFrame used to train the model.

    This function performs the following:
    1. Calculates and logs the model's performance metrics on the training data.
    2. Calculates and logs feature importance scores.
    3. Generates and saves probability predictions for the positive class.
    """
    train_metrics = predictor.evaluate(df_train)
    feature_importance = predictor.feature_importance(df_train)
    logger.info(f"Train metrics:\n {train_metrics}")
    logger.info(f"Feature importance:\n {feature_importance}")

    # Save probability predictions on train dataset to a CSV file.
    train_preds = predictor.predict_proba(df_train)
    positive_class_preds = train_preds[1]
    pred_df = df_train.copy()
    pred_df['predicted_proba'] = positive_class_preds
    pred_df.to_csv(f"predictions/train_preds_autogluon_{TIME_STAMP}.csv", index=False)
    logger.info(f"Train predictions saved to predictions/train_preds_autogluon_{TIME_STAMP}.csv")

    # ROC_AUC score
    roc_auc = train_metrics[EVAL_METRIC]
    logger.info(f"ROC_AUC score: {roc_auc:.5f}")

    # Log the classification report with threshold 0.5
    from sklearn.metrics import classification_report
    y_true = df_train[TARGET_COL]
    y_pred = (positive_class_preds > 0.5).astype(int)
    report = classification_report(y_true, y_pred)
    logger.info(f"Classification report:\n {report}")

def save_submission_file(predictor, df_test, df_sub):
    """
    Generate predictions for the test set and save them in a submission file.

    Args:
        predictor (autogluon.tabular.TabularPredictor): The trained AutoGluon model.
        df_test (pandas.DataFrame): The preprocessed test DataFrame.
        df_sub (pandas.DataFrame): The sample submission DataFrame with the correct format.

    This function performs the following steps:
    1. Creates copies of the input DataFrames to avoid modifying the originals.
    2. Uses the predictor to generate probability predictions for the positive class.
    3. Adds the predictions to the submission DataFrame.
    4. Saves the submission DataFrame as a CSV file with a timestamped filename.
    """
    df_test = df_test.copy()
    df_sub = df_sub.copy()
    df_sub[TARGET_COL] = predictor.predict_proba(df_test)[1] # predicting the probability of the positive class, index 1 is for the positive class.
    df_sub.to_csv(f"predictions/submission_autogluon_{TIME_STAMP}.csv", index=False)
    logger.info(f"Submission file saved to predictions/submission_autogluon_{TIME_STAMP}.csv")

if __name__ == "__main__":
    """
    Main execution block of the script.

    This block performs the following steps:
    1. Sets up logging to capture all console output in a timestamped log file.
    2. Loads the training, test, and sample submission data from CSV files.
    3. Preprocesses the training and test data (including class balancing for training data).
    4. Trains an AutoGluon model on the preprocessed training data.
    5. Evaluates the trained model's performance on the training data.
    6. Generates predictions for the test data and saves them in a submission file.

    All console output is redirected to the log file specified by LOG_FILE constant.
    """
    try:
        df_train = pd.read_csv(f'{DATA_DIR}/train.csv')
        df_test  = pd.read_csv(f'{DATA_DIR}/test.csv')
        df_sub = pd.read_csv(f'{DATA_DIR}/sample_submission.csv')

        logger.info(f"Original class distribution: {df_train['loan_status'].value_counts()}")
        mod_df_train = data_preprocessing(df_train)
        mod_df_test = data_preprocessing(df_test, test=True)

        predictor = train_model(mod_df_train)
        evaluate_predictor(predictor, df_train) # we need to pass the original df_train to the evaluate_predictor function, because the model has been trained on the downsampled df_train.
        save_submission_file(predictor, mod_df_test, df_sub)
    except Exception as e:
        logger.exception("An error occurred during script execution:")
    finally:
        logging.shutdown()