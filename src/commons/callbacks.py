import numpy as np

from src.commons.io import save_param_store


class EarlyStopping:
    # Source: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, monitor_metric: str, mode: str = "max", patience=3, delta=0, path="checkpoint.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 3
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.monitor_metric = monitor_metric
        self.patience = patience
        self.mode = self._get_mode(mode)
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def _get_mode(self, mode: str):
        if mode == "max":
            return 1
        elif mode == "min":
            return -1
        else:
            raise NotImplementedError("Available modes for early stopping are 'min' or  'max'")

    def __call__(self, metrics):
        score = self.mode * metrics[self.monitor_metric].compute().cpu().item()
        if self.best_score is None:
            self.best_score = score
            save_param_store(self.path)
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.counter = 0
            save_param_store(self.path)
            return True

    def improved(self, metrics):
        return self(metrics)
