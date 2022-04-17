import numpy as np
import torch

# TODO implement metrics with torchmetrics.Metric design, vectorize chi_squared metric and others


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


def _calculate_confidence_interval(percentile: float, y_pred: np.array):
    assert 0.0 < percentile < 100.0, "percentile must be between 0 and 100"
    marigin = (100 - percentile) / 2.0
    y_l = np.percentile(y_pred, marigin, axis=1)
    y_u = np.percentile(y_pred, 100 - marigin, axis=1)
    return y_l, y_u


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
