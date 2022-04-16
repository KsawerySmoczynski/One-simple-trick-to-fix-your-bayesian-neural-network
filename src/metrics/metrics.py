import sys
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import mean_squared_error

EPS = sys.float_info.min * sys.float_info.epsilon


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


def MPIW(y_pred: np.array, percentile: float = 50.0):
    """
    :param y_pred: tensor with sample predictions from the model. Dimensions: batch_size x n_samples
    :param percentile: alpha (significance level) parameter in percents
    :return: Mean prediction interval width metric
    """
    y_l, y_u = _calculate_confidence_interval(percentile, y_pred)
    mpiw = np.mean(y_u - y_l)
    return mpiw


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor):
    """

    :param y_true: tensor with true values. Dimensions: batch_size
    :param y_pred: tensor with sample predictions from the model. Dimensions: batch_size x n_samples
    :return: Accuracy of prediction (given as mode of sampled distribution)
    """

    mode_pred = torch.mode(y_pred, 1)[0]
    accuracy = (mode_pred == y_true).sum() / y_true.shape[0]
    return accuracy


def bin_metric(y_true, y_pred):
    """

    :param y_true: tensor with true values. Dimensions: batch_size
    :param y_pred: tensor with sample predictions from the model. Dimensions: batch_size x n_samples
    :return: For percantages in intervals [0,5], [5,15], [15,25], ..., [85,95], [95, 100]
            good_prob_bins and bad_prob_bins are model's good and bad predictions given confidence in that interval
    """

    n_samples = y_pred.shape[1]

    class_probs = np.apply_along_axis(lambda x: np.bincount(x, minlength=10), axis=1, arr=y_pred) / n_samples
    class_bins = np.rint(class_probs * 10) + 1

    one_hot = np.zeros((y_true.shape[0], 10))
    one_hot[np.arange(y_true.shape[0]), y_true] = 1

    good_mask = one_hot
    bad_mask = 1 - good_mask

    good_bins = class_bins * good_mask
    bad_bins = class_bins * bad_mask

    good_prob_bins = np.apply_along_axis(lambda x: np.bincount(x, minlength=12), axis=1, arr=good_bins.astype(int)).sum(
        axis=0
    )
    bad_prob_bins = np.apply_along_axis(lambda x: np.bincount(x, minlength=12), axis=1, arr=bad_bins.astype(int)).sum(
        axis=0
    )

    return good_prob_bins, bad_prob_bins


def _calculate_confidence_interval(percentile: float, y_pred: np.array):
    assert 0.0 < percentile < 100.0, "percentile must be between 0 and 100"
    marigin = (100 - percentile) / 2.0
    y_l = np.percentile(y_pred, marigin, axis=1)
    y_u = np.percentile(y_pred, 100 - marigin, axis=1)
    return y_l, y_u


def get_bin_cardinalities_acc_conf(
    pred: np.ndarray, target: np.ndarray, n_bins=10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred, target = np.array(pred), np.array(target)
    pred_class = np.argmax(pred, axis=1)
    conf = np.amax(pred, axis=1)
    bins = ((conf - EPS) * n_bins).astype(int)
    Bn = np.bincount(bins, minlength=n_bins)  # Bin cardinalities
    TPs = (pred_class == target).astype(int)
    bin_sum_acc = np.bincount(bins, weights=TPs, minlength=n_bins)
    bin_sum_conf = np.bincount(bins, weights=conf, minlength=n_bins)
    return Bn, bin_sum_conf, bin_sum_acc


def get_ece(Bn: np.ndarray, bin_sum_acc: np.ndarray, bin_sum_conf: np.ndarray):
    return np.abs((bin_sum_acc - bin_sum_conf)).sum() / Bn.sum()


def ece_score(py, y_test, n_bins=10):
    """code based on
    https://github.com/sirius8050/Expected-Calibration-Error/blob/master/ECE.py"""
    py = np.array(py)
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)


def chi_sqare(class_probabilities: np.array, labels: np.array, n_bins: int = 5, verbose: bool = True):
    bins = np.linspace(0, 1.001, n_bins + 1)
    chi_sq = 0
    for class_ in range(class_probabilities.shape[1]):
        x = []
        y = []
        el = 0
        for lower, upper in zip(bins[:-1], bins[1:]):
            p = class_probabilities[:, class_]
            selected = (p >= lower) & (p < upper)
            expected = p[selected].sum()
            observed = (labels[selected] == class_).sum()
            if expected >= 30.0:
                el += (observed - expected) ** 2 / expected
            elif (expected > 0.0) and verbose:
                print(
                    f"""Expected counts equals {expected:.1f},
which is lower than 30 and violates Chi^2 assumption.
class={class_}, lower_end={lower}, upper_end={upper}
"""
                )
                el += (observed - expected) ** 2 / expected
            else:
                pass
        chi_sq += el
        return chi_sq
