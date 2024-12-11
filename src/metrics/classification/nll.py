from typing import Tuple

import torch
import torch.nn.functional as F
from torchmetrics import Metric

from src.metrics.reduction import ClassificationReductionMixin


class NegativeLogLikelihood(ClassificationReductionMixin, Metric):
    def __init__(self, num_classes: int, input_type="probabilities", *args, **kwargs):
        """
        Args:
            num_classes: number of classes
            input_type: "samples" or "probabilities", when used with samples reduction and probability calculation will be applied
        """
        super().__init__(*args, **kwargs)
        self.classes = torch.tensor(range(num_classes))
        self.reduction = self._get_reduction(input_type)
        self.add_state("nll", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def _get_reduction(self, input_type: str):
        reduction = super()._get_reduction(input_type)
        return lambda x: reduction(x).log()

    def set_device(self, device: torch.DeviceObjType):
        self.classes = self.classes.to(device)
        self.nll = self.nll.to(device)
        self.to(device)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = self.reduction(preds)
        self.nll += F.nll_loss(preds, target, reduction="sum")

    def compute(self):
        return self.nll
