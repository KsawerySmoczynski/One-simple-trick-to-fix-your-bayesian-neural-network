from typing import Optional

import torch
from torchmetrics.classification import Accuracy as Acc

from src.metrics.reduction import ClassificationReductionMixin


class Accuracy(ClassificationReductionMixin, Acc):
    def __init__(
        self,
        input_type: str = "samples",
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        average: str = "micro",
        mdmc_average: Optional[str] = "global",
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        multiclass: Optional[bool] = None,
        subset_accuracy: bool = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            threshold,
            num_classes,
            average,
            mdmc_average,
            ignore_index,
            top_k,
            multiclass,
            subset_accuracy,
            *args,
            **kwargs
        )
        self.classes = torch.tensor(range(num_classes))
        self.reduction = self._get_reduction(input_type)
        # self.num_classes = num_classes

    def set_device(self, device):
        self.classes = self.classes.to(device)
        self.to(device)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = self.reduction(preds)
        # print(preds.shape, target.shape, self.num_classes)
        return super().update(preds, target)
