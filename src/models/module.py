from torch import nn


class Module(nn.Module):
    def __init__(self, activation: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = activation

    def __len__(self):
        return len(nn.utils.parameters_to_vector(self.parameters()))

    def print_parameter_size(self):
        print(f"Net: {self.__class__.__name__} has: {len(self)} parameters")
