import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


def normalize_data(data):
    """
    normalizes data between 0 and 1
    arguments:
        data: data to be normalized
    return:
        normalized data
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def corrcoef2D(A, B):
    """
    from https://stackoverflow.com/questions/30143417/computing-the-correlation-coefficient-between-two-multi-dimensional-arrays
    """
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))
    """
    example:
    a = np.array(([1, 3, 7], [1, 2, 7]))
    b = np.array(([1, 3, 5], [1, 3, 7]))

    print(f"a:\n{a}\nb:\n{b}")

    print(f"row-wise corr:\n{corrcoef2D(a, b)}")
    """


def ttest_greater(a, b, context):
    tick_labels, y_label, title = context
    means = (np.mean(a), np.mean(b))
    stds = (np.std(a), np.std(b))
    plt.bar(
        (1, 2),
        means,
        yerr=stds,
        capsize=10,
        tick_label=tick_labels,
    )
    plt.ylabel(y_label)
    plt.title(title)
    plt.plot()
    plt.show()
    results = ttest_ind(a, b, alternative="greater")
    return results
