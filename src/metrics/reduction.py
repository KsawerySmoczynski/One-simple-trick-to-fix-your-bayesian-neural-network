import abc

import torch


class ReductionMixin(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "set_device")
            and callable(subclass.set_device)
            and hasattr(subclass, "_get_reduction")
            and callable(subclass._get_reduction)
            or NotImplemented
        )

    @abc.abstractmethod
    def _get_reduction(self, input_type: str):
        raise NotImplementedError

    @abc.abstractmethod
    def set_device(self, device: torch.DeviceObjType):
        raise NotImplementedError


class ClassificationReductionMixin(ReductionMixin):
    def _get_reduction(self, input_type: str):
        assert hasattr(self, "classes"), "Object has to have classes vector assigned"

        def reduce_samples(preds: torch.Tensor):
            summed_classes = (preds[:, :, None] == self.classes[None, None, :]).sum(1)  # batch_size x n_samples x class
            return summed_classes / preds.shape[1]

        if input_type == "samples":
            return reduce_samples
        elif input_type == "none":
            return lambda x: x

    def set_device(self, device: torch.DeviceObjType):
        self.to(device)
        self.classes = self.classes.to(device)


class RegressionReductionMixin(ReductionMixin):
    def _get_reduction(self, input_type: str):
        if input_type == "samples":
            return lambda x: x.mean(dim=1)  # batch_size x n_samples
        elif input_type == "none":
            return lambda x: x

    def set_device(self, device: torch.DeviceObjType):
        self.to(device)
