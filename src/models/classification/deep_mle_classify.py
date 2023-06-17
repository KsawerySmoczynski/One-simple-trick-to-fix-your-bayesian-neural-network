import torch.nn.functional as F
from torch import nn

from src.models.module import Module


class DeepMLEClassify(Module):
    def __init__(self, activation: nn.Module, in_size: int, hidden_size: int, out_size: int, n_hidden: int = 2):
        super().__init__(activation)

        # self.bnorm1 = nn.BatchNorm1d(hidden_size)
        # self.bnorm2 = nn.BatchNorm1d(hidden_size)
        # self.bnorm3 = nn.BatchNorm1d(hidden_size)
        # self.bnorm4 = nn.BatchNorm1d(out_size)

        self.layer1 = nn.Linear(in_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, out_size)
        print(activation)
        self.print_parameter_size()

    def forward(self, x):
        x = self.activation(self.layer1(x.view(x.shape[0], -1)))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.layer4(x)
        # print(x.shape)
        # print(x[0])
        return F.log_softmax(x, dim=1)

# import torch.nn.functional as F
# from torch import nn

# from src.models.module import Module


# class DeepMLEClassify(Module):
#     def __init__(self, activation: nn.Module, in_size: int, hidden_size: int, out_size: int, n_hidden: int = 2):
#         super().__init__(activation)

#         self.layer1 = nn.Linear(in_size, hidden_size)
#         self.hidden = (
#             nn.Sequential(*[nn.Sequential(nn.Linear(hidden_size, hidden_size), activation) for i in range(n_hidden)])
#             if n_hidden > 0
#             else nn.Identity()
#         )
#         self.layer4 = nn.Linear(hidden_size, out_size)
#         self.print_parameter_size()

#     def forward(self, x):
#         x = self.activation(self.layer1(x.view(x.shape[0], -1)))
#         x = self.hidden(x)
#         x = self.layer4(x)
#         return F.log_softmax(x, dim=1)