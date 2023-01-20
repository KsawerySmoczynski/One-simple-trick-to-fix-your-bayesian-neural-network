from torch import nn
from typing import List

class Module(nn.Module):
    def __init__(self, activations: List[nn.Module], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activations = activations

    def __len__(self):
        return len(nn.utils.parameters_to_vector(self.parameters()))

    def print_parameter_size(self):
        print(f"{self.__class__.__name__} has: {len(self)} parameters")
