import torch
import numpy as np
from src.metrics.reduction import RegressionReductionMixin
from src.metrics.metrics import confidence_interval
from torchmetrics import Metric


class PCIP(RegressionReductionMixin, Metric):
    def __init__(self, percentile, input_type: str = "none", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reduction = self._get_reduction(input_type)
        self.percentile = percentile
        self.in_interval = 0
        self.total = 0

    def set_device(self, device):
        return

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = self.reduction(preds)
        lower, upper = confidence_interval(self.percentile, preds)
        self.in_interval += ((target > lower) & (target < upper)).sum()
        self.total += target.shape[0]


    def compute(self) -> torch.Tensor:
        return 100 * self.in_interval / self.total 

