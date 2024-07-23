import logging
from typing import Tuple
import numpy as np
import pandas
import pandas as pd
from sklearn import ensemble
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from vambn.utils.helpers import encode_numerical_columns, handle_nan_values

logger = logging.getLogger(__name__)


def get_auc(
    real: pandas.DataFrame, synthetic: pandas.DataFrame, n_folds: int = 10
) -> Tuple[float, float, int]:
    """
    Calculate the AUC score of a dataset using a Random Forest Classifier.

    Args:
        real (pandas.DataFrame): Real dataset.
        synthetic (pandas.DataFrame): Either decoded or synthetic dataset.
        n_folds (int, optional): Number of folds for cross-validation. Defaults to 10.

    Returns:
        Tuple[float, float, int]: Partial AUC, AUC, Number of samples.

    Raises:
        ValueError: If "VISIT" or "SUBJID" columns are present in the dataset.
    """
    for col in ["VISIT", "SUBJID"]:
        if col in real.columns:
            raise ValueError(f"Column {col} is present in the dataset")
        if col in synthetic.columns:
            raise ValueError(f"Column {col} is present in the dataset")

    logger.info(
        f"Calculating AUC with Random Forest Classifier ({n_folds} folds)"
    )
    real, synthetic = handle_nan_values(real, synthetic)
    real_enc = encode_numerical_columns(real)
    synthetic_enc = encode_numerical_columns(synthetic)

    x = pd.concat([real_enc, synthetic_enc]).values

    y = np.concatenate(
        (
            np.zeros(real_enc.shape[0], dtype=int),
            np.ones(synthetic_enc.shape[0], dtype=int),
        ),
        axis=None,
    )

    rfc = ensemble.RandomForestClassifier(random_state=42, n_estimators=100)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    auc_scores = []
    partial_auc_scores = []

    for i, (train_index, test_index) in enumerate(cv.split(x, y)):
        x_train, x_test = x[train_index, :], x[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        rfc.fit(x_train, y_train)
        y_pred_proba = rfc.predict_proba(x_test)[:, 1]

        auc = roc_auc_score(y_test, y_pred_proba)
        auc_scores.append(auc)

        partial_auc = roc_auc_score(y_test, y_pred_proba, max_fpr=0.2)
        partial_auc_scores.append(partial_auc)

        # If partial auc > 0.8, log the feature importances
        if partial_auc > 0.8:
            logger.warning(
                f"Partial AUC: {partial_auc} at fold {i} with {x_train.shape[0]} samples"
            )
            feature_importances = rfc.feature_importances_
            indices = np.argsort(feature_importances)[::-1]
            logger.warning("Feature ranking:")
            colnames = real.columns
            for f in range(x_train.shape[1]):
                logger.warning(
                    f"{f + 1}. feature {colnames[indices[f]]} ({feature_importances[indices[f]]})"
                )
        else:
            logger.debug(
                f"AUC: {auc}, partial AUC: {partial_auc} at fold {i} with {x_train.shape[0]} samples"
            )

    average_auc = np.mean(auc_scores)
    average_partial_auc = np.mean(partial_auc_scores)

    logger.info(f"Average AUC: {average_auc}")
    logger.info(f"Average partial AUC: {average_partial_auc}")

    return average_partial_auc, average_auc, real.shape[0]
