import matplotlib.pyplot as plt
import numpy as np

from src.commons.utils import fit_N, fit_sigma


def plot_1d(df, val, i, train_limit, save_path: str = None):
    good = df[:, 2].astype("float") / train_limit
    logp = df[:, 1]
    p = np.exp(logp - np.max(logp))
    plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 1, 1)
    ax.plot(df[:, 0], p, c="b")
    # p_N = fit_N(df[:, 0], p)
    p_N = fit_sigma(df[:, 0], p)
    # ax.plot(df[:, 0], p_N, alpha=0.7, c="y")
    # ax.plot(df[:, 0], good, alpha=0.7, c="g")
    # ax.axvline(x=val, c="r")
    plt.ylim(0.0, 1.05)
    # plt.legend(["likelihood", "normal fitted", "accuracy", "parameter"])
    plt.title('LeakyReLU')
    plt.xlabel('weight')
    plt.ylabel('Likelihood')
    if save_path:
        plt.savefig(save_path, dpi=1000)
    else:
        plt.show()


def plot_2d(df, val1, i1, val2, i2, reference_ll, save_path: str = None):
    logp = df[:, 2]
    p = np.exp(logp - reference_ll)
    length = np.sqrt(len(p))
    assert int(length) == length
    length = int(length)
    X = df[:, 1].reshape(length, length)
    Y = df[:, 0].reshape(length, length)
    plt.title(f"i1:{i1}, val: {val1:.4f}" + "\nX\n" + f"i2:{i2}, val: {val2:.4f}")
    plt.axhline(y=val1, c="orchid", alpha=0.95)
    plt.axvline(x=val2, c="orchid", alpha=0.95)
    plt.contourf(X, Y, p.reshape(length, length), levels=np.linspace(0, p.max(), 25), cmap="terrain")
    plt.colorbar()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", transparent=True)
    else:
        plt.show()
    plt.close()
