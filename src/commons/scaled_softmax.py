import torch.nn.functional as F

def scaled_log_softmax(x, scale, dim=None):
    return F.log_softmax(x / scale, dim=dim)

def scaled_softmax(x, scale, dim=None):
    return F.softmax(x / scale, dim=dim)