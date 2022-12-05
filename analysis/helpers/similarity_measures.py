import numpy as np


def normalize_adjacency(W):
    """
    -> adapted from https://www.programcreek.com/python/?CodeExample=normalize+adjacency, Project: graph-neural-networks Author: alelab-upenn
    Computes the degree-normalized adjacency matrix

    arguments:
        W (np.array): adjacency matrix
    returns:
        A (np.array): degree-normalized adjacency matrix
    """
    n, m = W.shape
    # Check that the matrix is square
    assert n == m
    # Compute the degree vector
    d = np.sum(W, axis=1)
    # Invert the square root of the degree
    d = 1 / np.sqrt(d)
    # And build the square root inverse degree matrix
    D = np.zeros((n, m))
    for i in np.arange(n):  # np.diag did not work bc d cannot be squeezed/flattened
        D[i, i] = d[i, 0]
    # Return the Normalized Adjacency
    return D @ W @ D


def TV(G, signal):
    """
    -> based on Bay-Ahmed et al., 2017
    Calculates the Total Variation of a signal on a graph normalized by the number of nodes.
    Degree-normalized adjacency matrix and L2 norm are used.
    arguments:
        G: graph (pygsp object)
        signal: data matrix (nodes x timepoints)
    returns:
        TV: total variation
    """
    # normalize adjacency matrix
    A = normalize_adjacency(G.W)
    TV = np.linalg.norm(signal - A @ signal) / G.N
    return TV


def simi_TV(G1, G2, signal1, signal2):
    """
    -> based on Bay-Ahmed et al., 2017
    Calculates the similarity of two graphs & sognals based on their total variations.
    The similarity measure is based on the interaction between node values (signal) and graph structure.
    arguments:
        G1: first graph (pygsp object)
        G2: second graph (pygsp object)
        signal1: data matrix for G1 (nodes x timepoints)
        signal2: data matrix for G2 (nodes x timepoints)
        q: used for q-norm
    returns:
        TVG: similarity measure based on total variation
    """
    TVG = np.abs(TV(G1, signal1) - TV(G2, signal2))
    return TVG


def LE(G):
    """
    -> based on Bay-Ahmed et al., 2017
    Calculates Laplacian graph energy.
    arguments:
        G: graph (pygsp object)
    returns:
        TV: total variation
    """
    LE = np.sum(np.abs(G.e - 2 * G.Ne / G.N))
    return LE


def simi_GE(G1, G2):
    """
    -> based on Bay-Ahmed et al., 2017
    Calculates the similarity of two graphs based on their Laplacian graph energy.
    This similarity measure is based only on the graph (complexity).
    arguments:
        G1: first graph (pygsp object)
        G2: second graph (pygsp object)
    returns:
        TVG: similarity measure based on total variation
    """
    GE = np.sum(np.abs(LE(G1) - LE(G2)))
    return GE


def simi_JET(G1, G2, signal1, signal2, gamma=0.5):
    """
    -> based on Bay-Ahmed et al., 2017
    Calculates the similarity of two signals and their graphs through the joint difference of
    the similarity emasure based on the Total Variation and the Laplacian graph energy similarity measure.
    This similarity measure is based on the interaction between node values (signal) and graph structure, and graph complexity.
        arguments:
        G1: first graph (pygsp object)
        G2: second graph (pygsp object)
        signal1: data matrix for G1 (nodes x timepoints)
        signal2: data matrix for G2 (nodes x timepoints)
        gamma: defines weights of TVG and GE (from [0,1], larger -> more influence of GE)
    returns:
        JET: similarity measure
    """

    TVG = simi_TV(G1, G2, signal1, signal2)
    GE = simi_GE(G1, G2)

    JET = gamma * GE - (1 - gamma) * TVG

    return JET
