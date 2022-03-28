import numpy as np
from sklearn.metrics import mean_squared_error


def RMSE(y_true: np.array, y_pred: np.array):
    """
    :param y_true: tensor with true values. Dimensions: batch_size
    :param y_pred: tensor with sample predictions from the model. Dimensions: batch_size x n_samples
    :return: Root mean-squared error metric
    """
    y_pred_mean = y_pred.mean(axis=1)
    mse = mean_squared_error(y_true, y_pred_mean)
    return np.sqrt(mse)


def PCIP(y_true: np.array, y_pred: np.array, percentile: float = 50.0):
    """
    :param y_true: tensor with true values. Dimensions: batch_size
    :param y_pred: tensor with sample predictions from the model. Dimensions: batch_size x n_samples
    :param percentile: alpha (significance level) parameter in percents
    :return: Prediction interval coverage probability metric
    """
    y_l, y_u = _calculate_confidence_interval(percentile, y_pred)
    pcip = np.mean((y_l < y_true) & (y_true < y_u))
    return pcip


def MPIW(y_true: np.array, y_pred: np.array, percentile: float = 50.0):
    """
    :param y_true: tensor with true values. Dimensions: batch_size
    :param y_pred: tensor with sample predictions from the model. Dimensions: batch_size x n_samples
    :param percentile: alpha (significance level) parameter in percents
    :return: Mean prediction interval width metric
    """
    y_l, y_u = _calculate_confidence_interval(percentile, y_pred)
    mpiw = np.mean(y_u - y_l)
    return mpiw


def _calculate_confidence_interval(percentile: float, y_pred: np.array):
    assert 0.0 < percentile < 100.0, "percentile must be between 0 and 100"
    marigin = (100 - percentile) / 2.0
    y_l = np.percentile(y_pred, marigin, axis=1)
    y_u = np.percentile(y_pred, 100 - marigin, axis=1)
    return y_l, y_u