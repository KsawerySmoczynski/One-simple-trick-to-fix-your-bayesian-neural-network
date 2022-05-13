import abc

import torch


class ReductionMixin(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "load_data_source")
            and callable(subclass.load_data_source)
            and hasattr(subclass, "extract_text")
            and callable(subclass.extract_text)
            or NotImplemented
        )

    @abc.abstractmethod
    def _get_reduction(self, input_type: str):
        """Load in the data set"""
        raise NotImplementedError

    @abc.abstractmethod
    def set_device(self, device):
        raise NotImplementedError


class ClassificationReductionMixin(ReductionMixin):
    def _get_reduction(self, input_type: str):
        def reduce_samples(preds: torch.Tensor):
            summed_classes = (preds[:, :, None] == self.classes[None, None, :]).sum(1)  # batch_size x n_samples x class
            return summed_classes / preds.shape[1]

        if input_type == "samples":
            return reduce_samples
        elif input_type == "probabilities":
            return lambda x: x


class RegressionReductionMixin(ReductionMixin):
    def _get_reduction(self, input_type: str):
        if input_type == "samples":
            return lambda x: x.mean(dim=1)  # batch_size x n_samples
        elif input_type == "probabilities":
            return lambda x: x
