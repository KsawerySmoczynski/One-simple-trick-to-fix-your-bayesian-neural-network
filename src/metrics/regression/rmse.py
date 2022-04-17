import torch
from torchmetrics.regression import MeanSquaredError

from src.metrics.reduction import RegressionReductionMixin


class RootMeanSquaredError(RegressionReductionMixin, MeanSquaredError):
    def __init__(self, input_type: str = "samples", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reduction = self._get_reduction(input_type)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = self.reduction(preds)
        super().update(preds, target)

    def compute(self) -> torch.Tensor:
        return super().compute().sqrt()
