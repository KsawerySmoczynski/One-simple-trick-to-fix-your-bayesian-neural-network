import matplotlib.pyplot as plt
import numpy as np

from src.commons.utils import fit_N


def plot_1d(df, val, i, train_limit, save_path: str = None):
    good = df[:, 2].astype("float") / train_limit
    logp = df[:, 1]
    p = np.exp(logp - np.max(logp))
    plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 1, 1)
    ax.plot(df[:, 0], p)
    ax.axvline(x=val, c="r")
    p_N = fit_N(df[:, 0], p)
    ax.plot(df[:, 0], p_N, alpha=0.7)
    ax.plot(df[:, 0], good, alpha=0.7)
    plt.ylim(0.0, 1.05)
    plt.title(f"i:{i}, val: {val}")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_2d(df, val1, i1, val2, i2, window, rate, train_limit, save_path: str = None):
    good = df[:, 3].astype("float") / train_limit
    logp = df[:, 2]
    p = np.exp(logp - np.max(logp))
    length = np.sqrt(len(p))
    assert int(length) == length
    length = int(length)
    X, Y = np.meshgrid(
        np.linspace(val1 - window // 2, val1 + window // 2, rate),
        np.linspace(val2 - window // 2, val2 + window // 2, rate),
    )
    plt.contourf(X, Y, p.reshape(length, length), levels=np.linspace(0, p.max(), 20))
    plt.colorbar()
    plt.title(f"i1:{i1}, val: {val1:.4f} X i2:{i2}, val: {val2:.4f}")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
