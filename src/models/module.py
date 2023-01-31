from torch import nn


class Module(nn.Module):
    def __init__(self, activation: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = activation
        self.scale_function = None

    def _get_scale_function(self):
        # partial from activations
        raise NotImplementedError
        return None

    def __len__(self):
        return len(nn.utils.parameters_to_vector(self.parameters()))

    def print_parameter_size(self):
        print(f"{self.__class__.__name__} has: {len(self)} parameters")
