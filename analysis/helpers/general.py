import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs
from scipy.stats import ttest_ind


def normalize_data_minmax(data, axis=None):
    """
    normalizes data between 0 and 1
    arguments:
        data: data to be normalized
        axis: normalization along axis
    return:
        normalized data
    """
    if axis is None:
        norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    else:
        norm_data = (data - np.min(data, axis=axis)) / (
            np.max(data, axis=axis) - np.min(data, axis=axis)
        )

    return norm_data


def normalize_data_sum(data, axis=None):
    """
    normalizes data by subtracting the min if there are negative values in the dataset and then normalizing the sum of the data to 1
    arguments:
        data: data to be normalized
        axis: normalization along axis
    return:
        normalized data
    """
    if axis is None:
        data_min = np.min(data)
        if data_min < 0:
            data = data - data_min
        norm_data = data / np.sum(data)
    else:
        data_min = np.min(data, axis=axis)
        # if data_min < 0:
        #    data = data - data_min
        data = np.where(data_min < 0, data - data_min, data)
        norm_data = data / np.sum(data, axis=axis)
    return norm_data


def corrcoef2D(A, B):
    """
    from https://stackoverflow.com/questions/30143417/computing-the-correlation-coefficient-between-two-multi-dimensional-arrays
    calculate correlation between all rows of 2 arrays
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


# make this a function with mean(fisher_transf(abs(correlation_data))) instead?
def Fisher_transf(data):
    """
    Returns Fisher transformed data.
    arguments:
        data
    returns
        transf_data: Fisher transformed data
    """
    transf_data = np.emath.arctanh(data)
    return transf_data


def ttest_greater(a, b, context):
    """
    Performs an independent t-test and plots the result in a barplot.
    arguments:
        a: group A
        b: group B
        context: x tick labels, y label and title
    returns:
        results: ttest results
    """
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
    plt.savefig(f"../results/hypotheses/ttest_{title}.png")
    results = ttest_ind(a, b, alternative="greater")
    return results


def check_graph(G, weights):
    """
    Prints ionformation about graph:
        - weight symmetry
        - nodes & edges
        - connected/directed graph
        - weights as expected
    """
    print(f"weights symmetric: {np.allclose(weights, weights.T)}")
    print(weights)

    # create graph for first participant & check properties
    G = graphs.Graph(weights)
    print("{} nodes, {} edges".format(G.N, G.Ne))

    print(f"connected: {G.is_connected()}")
    print(f"directed: {G.is_directed()}")  # why is it directed?

    print(G.check_weights())
    print(f"weights as initialized: {(G.W == weights).all()}")
