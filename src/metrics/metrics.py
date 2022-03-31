import numpy as np
import torch
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


def MPIW(y_pred: np.array, percentile: float = 50.0):
    """
    :param y_pred: tensor with sample predictions from the model. Dimensions: batch_size x n_samples
    :param percentile: alpha (significance level) parameter in percents
    :return: Mean prediction interval width metric
    """
    y_l, y_u = _calculate_confidence_interval(percentile, y_pred)
    mpiw = np.mean(y_u - y_l)
    return mpiw

def accuracy (y_true: torch.Tensor, y_pred: torch.Tensor):
  """

  :param y_true: tensor with true values. Dimensions: batch_size
  :param y_pred: tensor with sample predictions from the model. Dimensions: batch_size x n_samples
  :return: Accuracy of prediction (given as mode of sampled distribution)
  """

  mode_pred = torch.mode(y_pred, 1)[0]
  accuracy = (mode_pred == y_true).sum() / y_true.shape[0]
  return accuracy

def bin_metric (y_true, y_pred):
  """

  :param y_true: tensor with true values. Dimensions: batch_size
  :param y_pred: tensor with sample predictions from the model. Dimensions: batch_size x n_samples
  :return: For percantages in intervals [0,5], [5,15], [15,25], ..., [85,95], [95, 100] 
          good_prob_bins and bad_prob_bins are model's good and bad predictions given confidence in that interval
  """

  n_samples = y_pred.shape[1]

  class_probs =  np.apply_along_axis(lambda x: np.bincount(x, minlength=10), axis=1, arr=y_pred) / n_samples
  class_bins = np.rint(class_probs * 10) + 1
  
  one_hot = np.zeros((y_true.shape[0], 10)) 
  one_hot[np.arange(y_true.shape[0]), y_true] = 1

  good_mask = one_hot
  bad_mask = 1 - good_mask

  good_bins = class_bins * good_mask
  bad_bins = class_bins * bad_mask

  good_prob_bins = np.apply_along_axis(lambda x: np.bincount(x, minlength=12), axis=1, arr=good_bins.astype(int)).sum(axis=0)
  bad_prob_bins = np.apply_along_axis(lambda x: np.bincount(x, minlength=12), axis=1, arr=bad_bins.astype(int)).sum(axis=0)

  return good_prob_bins, bad_prob_bins

def _calculate_confidence_interval(percentile: float, y_pred: np.array):
    assert 0.0 < percentile < 100.0, "percentile must be between 0 and 100"
    marigin = (100 - percentile) / 2.0
    y_l = np.percentile(y_pred, marigin, axis=1)
    y_u = np.percentile(y_pred, 100 - marigin, axis=1)
    return y_l, y_u