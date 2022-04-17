from typing import Tuple

import torch
from torchmetrics import Metric

from src.metrics.reduction import ClassificationReductionMixin

EPS = torch.tensor(torch.finfo(float).eps)


class ExpectedCalibrationError(ClassificationReductionMixin, Metric):
    def __init__(self, num_classes: int, input_type="samples", num_bins: int = 10, *args, **kwargs):
        """
        Args:
            num_classes: number of classes
            input_type: "samples" or "probabilities", when used with samples reduction and probability calculation will be applied
            num_bins: number of equal intervals to which probability range will be split
        """
        super().__init__(*args, **kwargs)
        self.classes = torch.tensor(range(num_classes))
        self.reduction = self._get_reduction(input_type)
        self.num_bins = num_bins
        self.add_state("bin_cardinalities", default=torch.zeros(self.num_bins), dist_reduce_fx="sum")
        self.add_state("bin_sum_acc", default=torch.zeros(self.num_bins), dist_reduce_fx="sum")
        self.add_state("bin_sum_conf", default=torch.zeros(self.num_bins), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = self.reduction(preds)
        bin_cardinalities, bin_sum_acc, bin_sum_conf = self._get_bin_cardinalities_acc_conf(preds, target)
        self.bin_cardinalities += bin_cardinalities
        self.bin_sum_acc += bin_sum_acc
        self.bin_sum_conf = bin_sum_conf

    def compute(self):
        return (self.bin_sum_acc - self.bin_sum_conf).abs().sum() / self.bin_cardinalities.sum()

    def _get_bin_cardinalities_acc_conf(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_class = preds.argmax(dim=1)
        conf = preds.amax(dim=1)
        bins = (
            (conf.double() - EPS) * self.num_bins
        ).int()  # Cast for double due to numerical rounding issues look here: https://stackoverflow.com/questions/63818676/what-is-the-machine-precision-in-pytorch-and-when-should-one-use-doubles
        bin_cardinalities = torch.bincount(bins, minlength=self.num_bins)
        TPs = (pred_class == target).int()
        bin_sum_acc = torch.bincount(bins, weights=TPs, minlength=self.num_bins)
        bin_sum_conf = torch.bincount(bins, weights=conf, minlength=self.num_bins)
        return bin_cardinalities, bin_sum_acc, bin_sum_conf
