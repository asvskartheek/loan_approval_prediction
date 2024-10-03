"""
This script trains an AutoGluon model for loan approval prediction using 5-fold cross-validation.

It performs the following main steps:
1. Data preprocessing
2. 5-fold cross-validation with AutoGluon's TabularPredictor
3. Model evaluation
4. Generating out-of-fold (OOF) predictions for train and test sets
5. Generating predictions for submission

The script uses predefined constants for data paths, column names, and model parameters.
Logging is implemented to capture all console output in a timestamped log file.
"""

from collections import defaultdict
from datetime import datetime
from typing import List
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import logging
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# Constants
DATA_DIR = "/Users/asvs/kartheek_hobby_projects/loan_approval_prediction/data"

CATEGORICAL_COLS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]
NUMERICAL_COLS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
TARGET_COL = "loan_status"

EVAL_METRIC = "roc_auc"
TIME_LIMIT = 60 * 60  # 60 minutes
TIME_STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
EXPERIMENT_NAME = "cv_5fold"
LOG_FILE = f"logs/train_autogluon_{EXPERIMENT_NAME}_{TIME_STAMP}.log"
N_FOLDS = 5
RANDOM_STATE = 42

# Set up logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()


def data_preprocessing(df, test=False):
    """
    Preprocess the input DataFrame for AutoGluon training or prediction.

    Args:
        df (pandas.DataFrame): The input DataFrame to preprocess.
        test (bool): Flag indicating whether this is test data (default: False).

    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
    """
    df = df.copy()
    df = df.drop(
        columns=["id"]
    )  # dropping id column as it is not required for training
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype(
        "category"
    )  # type casting categorical columns to category, necessary for AutoGluon
    if not test:
        df[TARGET_COL] = df[TARGET_COL].astype(
            "int"
        )  # type casting target column to int.

    return df


def cross_validation(df_train, df_test) -> List[pd.DataFrame]:
    """
    Perform 5-fold cross-validation on the training data and return the predictions for each fold.

    Args:
        df_train (pandas.DataFrame): The training DataFrame.
        df_test (pandas.DataFrame): The test DataFrame.

    Returns:
        oof_train_preds (pandas.DataFrame): The out-of-fold predictions for the training data.
        oof_test_preds (pandas.DataFrame): The out-of-fold predictions for the test data.
    """
    oof_train_preds = defaultdict(list)
    oof_test_preds = defaultdict(list)
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_index, val_index) in tqdm(
        enumerate(kf.split(df_train, df_train[TARGET_COL])),
        total=N_FOLDS,
        desc="Cross-validation",
    ):
        train_data = df_train.iloc[train_index]
        val_data = df_train.iloc[val_index]

        predictor = TabularPredictor(
            label=TARGET_COL,
            eval_metric=EVAL_METRIC,
            log_to_file=True,
            log_file_path=LOG_FILE,
        ).fit(
            train_data,
            time_limit=TIME_LIMIT,
            verbosity=2,
            presets="best_quality"
        )

        # Evaluate the model on the validation data for this fold
        val_metrics = predictor.evaluate(val_data)
        val_roc_auc = val_metrics[EVAL_METRIC]
        logger.info(f"Fold {fold} validation {EVAL_METRIC}: {val_roc_auc:.5f}")

        # Evaluate predictor
        train_preds = predictor.predict_proba(df_train)[1]
        evaluate_predictor(predictor, df_train, train_preds)

        oof_train_preds[f"fold_{fold}"] = train_preds
        oof_test_preds[f"fold_{fold}"] = predictor.predict_proba(df_test)[1]

    return pd.DataFrame(oof_train_preds), pd.DataFrame(oof_test_preds)


def evaluate_predictor(predictor, df_train, train_preds):
    """
    Evaluate the trained AutoGluon model, log performance metrics, and save predictions.

    Args:
        predictor (autogluon.tabular.TabularPredictor): The trained AutoGluon model.
        df_train (pandas.DataFrame): The training DataFrame used to train the model.
        train_preds (numpy.ndarray): The predictions on the training data.

    This function performs the following:
    1. Calculates and logs the model's performance metrics on the training data.
    2. Calculates and logs feature importance scores.
    3. Generates and saves probability predictions for the positive class.
    """
    train_metrics = predictor.evaluate(df_train)
    feature_importance = predictor.feature_importance(df_train)
    logger.info(f"Train metrics:\n {train_metrics}")
    logger.info(f"Feature importance:\n {feature_importance}")

    # ROC_AUC score
    roc_auc = train_metrics[EVAL_METRIC]
    logger.info(f"ROC_AUC score: {roc_auc:.5f}")

    # Log the classification report with threshold 0.5
    from sklearn.metrics import classification_report

    y_true = df_train[TARGET_COL]
    y_pred = (train_preds > 0.5).astype(int)
    report = classification_report(y_true, y_pred)
    logger.info(f"Classification report:\n {report}")


def save_submission_file(oof_train_preds, oof_test_preds, df_sub):
    """
    Generate predictions for the test set and save them in a submission file.

    Args:
        oof_train_preds (pandas.DataFrame): The out-of-fold predictions for the training data.
        oof_test_preds (pandas.DataFrame): The out-of-fold predictions for the test data.
        df_sub (pandas.DataFrame): The sample submission DataFrame with the correct format.

    This function performs the following steps:
    1. Creates copies of the input DataFrames to avoid modifying the originals.
    2. Uses the predictor to generate probability predictions for the positive class.
    3. Adds the predictions to the submission DataFrame.
    4. Saves the submission DataFrame as a CSV file with a timestamped filename.
    """
    df_sub = df_sub.copy()
    df_sub[TARGET_COL] = oof_test_preds.mean(
        axis=1
    )  # predicting the probability of the positive class, index 1 is for the positive class.
    df_sub.to_csv(
        f"predictions/submission_autogluon_{EXPERIMENT_NAME}_{TIME_STAMP}.csv",
        index=False,
    )
    oof_train_preds.to_csv(
        f"predictions/oof_train_preds_{EXPERIMENT_NAME}_{TIME_STAMP}.csv", index=False
    )
    oof_test_preds.to_csv(
        f"predictions/oof_test_preds_{EXPERIMENT_NAME}_{TIME_STAMP}.csv", index=False
    )
    logger.info(f"Files saved:")
    logger.info(
        f"- Submission: predictions/submission_autogluon_{EXPERIMENT_NAME}_{TIME_STAMP}.csv"
    )
    logger.info(
        f"- OOF train predictions: predictions/oof_train_preds_{EXPERIMENT_NAME}_{TIME_STAMP}.csv"
    )
    logger.info(
        f"- OOF test predictions: predictions/oof_test_preds_{EXPERIMENT_NAME}_{TIME_STAMP}.csv"
    )


if __name__ == "__main__":
    """
    Main execution block of the script.

    This block performs the following steps:
    1. Sets up logging to capture all console output in a timestamped log file.
    2. Loads the training, test, and sample submission data from CSV files.
    3. Preprocesses the training and test data.
    4. Trains an AutoGluon model on the preprocessed training data.
    5. Evaluates the trained model's performance on the training data.
    6. Generates predictions for the test data and saves them in a submission file.

    All console output is redirected to the log file specified by LOG_FILE constant.
    """
    try:
        df_train = pd.read_csv(f"{DATA_DIR}/train.csv")
        df_test = pd.read_csv(f"{DATA_DIR}/test.csv")
        df_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

        logger.info(
            f"Original class distribution: {df_train['loan_status'].value_counts()}"
        )
        mod_df_train = data_preprocessing(df_train)
        mod_df_test = data_preprocessing(df_test, test=True)

        oof_train_preds, oof_test_preds = cross_validation(mod_df_train, mod_df_test)
        save_submission_file(oof_train_preds, oof_test_preds, df_sub)
    except Exception as e:
        logger.exception("An error occurred during script execution:")
    finally:
        logging.shutdown()
