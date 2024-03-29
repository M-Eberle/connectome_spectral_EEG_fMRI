import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from helpers.methods.general import *
from helpers.methods.overview_plots import ex_EEG_fMRI_corr


def vertex_vs_graph(
    EEG_timeseries, fMRI_timeseries_interp, trans_EEG_timeseries, trans_fMRI_timeseries
):
    """
    Compares correlation between fMRI and EEG in vertex (regions) and graph (harmonics) domain.
    Plots comparison of means and uses an independent t-test.
    arguments:
        EEG_timeseries: EEG activity timeseries, vertex domain
        fMRI_timeseries: fMRI activity timeseries (same length as EEG timeseries), vertex domain
        trans_EEG_timeseries: EEG gft weights timeseries, graph domain
        trans_fMRI_timeseries: fMRI gft weights timeseries (same length as EEG timeseries), graph domain
    returns:
        regions_all_corrs: all correlations within regions for all participants
        harmonic_all_corrs: all correlations within harmonics for all participants
        mean_regions_corrs
        mean_harmonics_corrs
        ttest_results

    """
    N = EEG_timeseries.shape[-1]
    N_reg, _ = EEG_timeseries[:, :, 0].shape
    mean_regions_corrs = np.empty((N))
    mean_harmonics_corrs = np.empty((N))
    regions_all_corrs = np.empty((N_reg, N))
    harmonics_all_corrs = np.empty((N_reg, N))

    for participant in np.arange(N):
        regions_corr = ex_EEG_fMRI_corr(
            EEG_timeseries, fMRI_timeseries_interp, participant, "region"
        )
        harmonics_corr = ex_EEG_fMRI_corr(
            trans_EEG_timeseries, trans_fMRI_timeseries, participant, "harmonic"
        )
        # only consider same regions/ harmonics
        regions_all_corrs[:, participant] = np.diag(regions_corr)
        harmonics_all_corrs[:, participant] = np.diag(harmonics_corr)
        # abs before mean?
        mean_reg = mean_corr(np.abs(regions_all_corrs[:, participant]))
        mean_harmonic = mean_corr(np.abs(harmonics_all_corrs[:, participant]))

        print(
            f"participant {participant+1} regions diag: {np.round(mean_reg, 4)}, harmonics diag: {np.round(mean_harmonic, 4)}"
        )
        mean_regions_corrs[participant] = mean_reg
        mean_harmonics_corrs[participant] = mean_harmonic

    # t-tets?
    ttest_results = ttest_greater(
        mean_regions_corrs,
        mean_harmonics_corrs,
        [
            ["regions\n(vertex)", "harmonics\n(graph)"],
            "correlation",
            "correlation between EEG and fMRI in vertex vs graph domain",
        ],
    )

    # ? absolute correlation between regions higher than harmonics
    return (
        regions_all_corrs,
        harmonics_all_corrs,
        mean_regions_corrs,
        mean_harmonics_corrs,
        ttest_results,
    )


# %%
# similarity measures
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
    The signal is also normalized (not mentioned in Bay-Ahmed et al.).
    arguments:
        G: graph (pygsp object)
        signal: data matrix (nodes x timepoints, for one participant)
    returns:
        TV: total variation
    """
    # normalize adjacency matrix
    A = normalize_adjacency(G.W)

    # normalize each timestep within each participant
    # which normalization???
    signal = normalize_data_minmax(signal, axis=0)
    # signal = normalize_data_sum(signal, axis=0)

    TV = np.linalg.norm(signal - A @ signal) / G.N
    return TV


def simi_TVG(G1, G2, signal1, signal2):
    """
    -> based on Bay-Ahmed et al., 2017
    Calculates the similarity of two graphs & sognals based on their total variations.
    The similarity measure is based on the interaction between node values (signal) and graph structure.
    arguments:
        G1: first graph (pygsp object)
        G2: second graph (pygsp object)
        signal1: data matrix for G1 (nodes x timepoints)
        signal2: data matrix for G2 (nodes x timepoints)
    returns:
        TVG: similarity measure based on total variation
    """
    TVG = np.abs(TV(G1, signal1) - TV(G2, signal2))
    return TVG


def LE(G):
    """
    -> based on Bay-Ahmed et al., 2017
    Calculates Laplacian graph energy.
    Here, the normalized Laplacian is used (not specified in Bay-Ahmed et al.).
    arguments:
        G: graph (pygsp object)
    returns:
        TV: total variation
    """
    # LE = np.sum(np.abs(G.e - 2 * G.Ne / G.N)) use normalized Laplacian for eigenvalues instead
    A = normalize_adjacency(G.W)
    norm_L = np.eye(A.shape[0]) - A  # normalized L = I - normalized A
    evals, evecs = np.linalg.eigh(norm_L)  # use eigh because norm_L is symmetric?
    LE = np.sum(np.abs(evals - 2 * G.Ne / G.N))

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


def simi_JET(G1, G2, signal1, signal2, gamma=0.001):
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

    TVG = simi_TVG(G1, G2, signal1, signal2)
    GE = simi_GE(G1, G2)
    print(f"gamma = {gamma}")
    JET = gamma * GE - (1 - gamma) * TVG

    return JET


def simi_betw_participants(
    Gs, simi_measure, measure_name, N, mode=None, timeseries=None
):
    """
    Calculates a similarity measure between all participants.
    arguments:
        Gs: list of graphs for participants
        simi_measure: function for similarity measure
        measure_name: name of similarity measure
        N: number ofp participants
        mode: data mode, e.g. 'EEG', 'fMRI' or 'random'
        timeseries: signal timeseries
    returns:
        simi: matrix with similarity measure
    """
    simi = np.zeros((N, N))
    for participant1 in np.arange(N):
        for participant2 in np.arange(N):
            if participant1 <= participant2:
                if timeseries is not None:
                    simi[participant1, participant2] = simi_measure(
                        Gs[participant1],
                        Gs[participant2],
                        timeseries[:, :, participant1],
                        timeseries[:, :, participant2],
                    )
                else:
                    simi[participant1, participant2] = simi_measure(
                        Gs[participant1],
                        Gs[participant2],
                    )
            else:
                simi[participant1, participant2] = simi[participant2, participant1]

    sns.heatmap(simi)
    plt.xlabel("participants")
    plt.ylabel("participants")
    if mode != None:
        plt.title(f"{mode}: {measure_name} between participants")
        file = f"../results/hypotheses/similarity_measures/{measure_name}_{mode}.png"
    else:
        plt.title(f"{measure_name} between participants")
        file = f"../results/hypotheses/similarity_measures/{measure_name}.png"
    plt.savefig(file)
    plt.show()


def TVG_random_signal(Gs, N_regions, timesteps, N):
    """
    Calculates the TVG between all participants for random weights on given graphs.
    argumnets:
        Gs: graphs for participants
        N_regions: number of nodes
        timesteps: length of signal timeseries
        N: number of participants
    returns:
        TVG: TVG between partiicpant's signal & graphs
    """
    random_timeseries = np.random.uniform(0, 1, (N_regions, timesteps, N))

    TVG = simi_betw_participants(
        Gs, simi_TVG, "TVG", N, "random signal", random_timeseries
    )
    return TVG


def TVG_random_signal_and_weights(coords, N_regions, timesteps, N):
    """
    Calculates the TVG between all participants for random weights and signals on graphs.
    argumnets:
        coords: graph coordinates
        N_regions: number of nodes
        timesteps: length of signal timeseries
        N: number of participants
    returns:
        TVG: TVG between partiicpant's signal & graphs
    """
    random_timeseries = np.random.uniform(0, 1, (N_regions, timesteps, N))

    random_graphs = []

    for participant in np.arange(N):
        random_weights = np.random.uniform(0, 1, (N_regions, N_regions))
        np.fill_diagonal(random_weights, 0)
        random_weights = random_weights + random_weights.T

        random_graph = graphs.Graph(
            random_weights,
            lap_type="normalized",
            coords=coords,
        )
        random_graphs.append(random_graph)

    TVG = simi_betw_participants(
        random_graphs,
        simi_TVG,
        "TVG",
        N,
        "random signal and weights",
        random_timeseries,
    )
    return TVG


def TVG_betw_evecs(Gs, N_regions, ex_participant):
    """
    Calculates the TVG between all harmonics.
    arguments:
        Gs: list of graphs for participants
        measure_name: name of similarity measure
        N_regions: number of haronics
        ex_participant: example participant index
    returns:
        TVG: matrix with TVGs
    """
    G = Gs[ex_participant]
    TVG = np.zeros((N_regions, N_regions))
    for region1 in np.arange(N_regions):
        for region2 in np.arange(N_regions):
            if region1 <= region2:
                TVG[region1, region2] = simi_TVG(
                    G,
                    G,
                    G.U[:, region1][:, np.newaxis],
                    G.U[:, region2][:, np.newaxis],
                )
            else:
                TVG[region1, region2] = TVG[region2, region1]

    sns.heatmap(TVG)
    plt.xlabel("regions")
    plt.ylabel("regions")
    plt.title(f"similarity of harmonics for participant {ex_participant +1}")
    plt.savefig(
        f"../results/hypotheses/similarity_measures/TVG_evecs_participant_{ex_participant+1}.png"
    )
    plt.show()
    return TVG
