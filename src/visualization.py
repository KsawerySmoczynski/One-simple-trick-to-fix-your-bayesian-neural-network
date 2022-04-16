import matplotlib.pyplot as plt
import numpy as np


def plot_expercted_vs_observed(class_probabilities: np.array, labels: np.array, n_bins: int = 5, title: str = None):
    bins = np.linspace(0, 1.001, n_bins + 1)
    for class_ in range(class_probabilities.shape[1]):
        x = []
        y = []
        for lower, upper in zip(bins[:-1], bins[1:]):
            p = class_probabilities[:, class_]
            selected = (p >= lower) & (p < upper)
            expected = p[selected].sum()
            observed = (labels[selected] == class_).sum()
            if selected.sum() > 0:
                x.append(expected / selected.sum())
                y.append(observed / selected.sum())
            elif selected.sum() == 0:
                x.append(None)
                y.append(None)
    plt.scatter(x, y)
    plt.plot([0, 1], [0, 1], c="r")
    plt.title(title)
    plt.show()
