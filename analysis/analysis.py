# %%
from scipy import io as sio
from pygsp import graphs, filters, plotting, utils
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.interpolate as interp
import seaborn as sns
from matplotlib.lines import Line2D


# %%

# %% [markdown]

### last steps
# - load data
# - transform data onto graph
# - check that participant's indices in SC, fMRI, and EEG align
# - symmetry of SCs? --> use +transpose for now
### next steps
# - compare fMRI & EEG signal with individual SCs
#   - plot power over smoothness per participant
#   - nr. of harmonics needed to recreate fMRI/EEG signal --> cumulative power from Glomb et al. (2020) Fig. 2
#   - compare patterns between participants (correlation matrices?)
# - compare fMRI & EEG signal with average SC
#   - averaged correlation matrix ?
#   - ? plot power over smoothness per participant


# %%
def get_data(
    SC_path,
    EEG_data_path,
    fMRI_data_path,
):
    """
    SC matrices, EEG and fMRI timeseries from ‘Hybrid Brain Model data’ in data folder
    are extracted for all participants.
    The fMRI data is interpolated (stretched) to fit the length of the EEG data.
    The EEG and interpolated fMRI timeseries are transformed by the SC graph's
    Laplacian's eigenvectors.
    arguments:
        SC_path: path to SC matrices file
        EEG_data_path: path to EEG data file
        fMRI_data_path: path to fMRI data file
    returns:
        SC_weights: list with SC matrix for each participant
        EEG_timeseries: list with EEG activity for each participant
        trans_EEG_timeseries: list with EEG GFT weights for each participant
        fMRI_timeseries: list with fMRI activity for each participant
        fMRI_timeseries_interp: list with interpolated fMRI activity for each participant
        trans_fMRI_timeseries: list with fMRI GFT weights for each participant
    """
    # load SC data
    unflattened_SC_data = sio.loadmat(SC_path)["SC"]
    SC_data = np.ndarray.flatten(unflattened_SC_data)
    N = SC_data.size

    # load fMRI data
    unflattened_fMRI_data = sio.loadmat(fMRI_data_path)["fMRI"]
    fMRI_data = np.ndarray.flatten(unflattened_fMRI_data)

    # load EEG data
    EEG_data_file = h5py.File(EEG_data_path, "r")

    # list of SC matrices
    SC_weights = []
    # lists of signal timeseries
    # EEG: 15x (68x259184) for participants x(RegionsxTime)
    EEG_timeseries = []
    trans_EEG_timeseries = []
    # fMRI: before interpolation: 15x (68x661) for participants x(RegionsxTime)
    # after interpolation as in EEG: 15x (68x259184) for participants x(RegionsxTime)
    fMRI_timeseries = []
    fMRI_timeseries_interp = []
    trans_fMRI_timeseries = []

    ID_count = 0
    for participant in np.arange(N):

        SC_weights_pre, SC_participant_ID = SC_data[participant]
        # extract weights from nested object
        # make weights symmetric to keep going for now
        SC_weights_pre[0, 0][0] = SC_weights_pre[0, 0][0] + SC_weights_pre[0, 0][0].T
        SC_weights.append(SC_weights_pre[0, 0][0])

        # compute participant's graph
        # save all Graphs?
        G = graphs.Graph(SC_weights[participant], lap_type="normalized")
        G.compute_fourier_basis()  # harmonics in G.U, eigenvalues in G.e

        # get participant's EEG data
        EEG_ID_store_curr = EEG_data_file["source_activity/sub_id"][participant][0]
        EEG_participant_ID = EEG_data_file[EEG_ID_store_curr][:]
        EEG_participant_ID = (
            "['" + "".join(chr(c[0]) for c in EEG_participant_ID) + "']"
        )

        EEG_data_store_curr = EEG_data_file["source_activity/ts"][participant][0]
        EEG_timeseries_curr = EEG_data_file[EEG_data_store_curr]
        EEG_timeseries.append(EEG_timeseries_curr[:, :].T)

        # get participant's fMRI data
        fMRI_timeseries_curr, fMRI_participant_ID = fMRI_data[participant]
        fMRI_timeseries.append(fMRI_timeseries_curr.T)

        N_regions, fMRI_timesteps = fMRI_timeseries[-1].shape
        N_regions, EEG_timesteps = EEG_timeseries[-1].shape

        # stretch fMRI data over time to EEG sequence length
        fMRI_interp_curr = np.empty((N_regions, EEG_timesteps))

        for region in np.arange(N_regions):

            fMRI_interp = interp.interp1d(
                np.arange(fMRI_timesteps), fMRI_timeseries[-1][region, :]
            )
            fMRI_interp_curr[region, :] = fMRI_interp(
                np.linspace(0, fMRI_timesteps - 1, EEG_timesteps)
            )
        fMRI_timeseries_interp.append(fMRI_interp_curr)

        # spectral representation of signal
        # the Fourier transform is simply calling the .gft() method
        # e.g. G.gft(signal)
        trans_EEG_timeseries.append(G.gft(EEG_timeseries[-1]))
        trans_fMRI_timeseries.append(G.gft(fMRI_timeseries_interp[-1]))

        if (str(SC_participant_ID) == str(EEG_participant_ID)) and (
            str(SC_participant_ID) == str(fMRI_participant_ID)
        ):
            ID_count += 1
            if ID_count == N:
                print(
                    "all participant IDs are represented by the same indices in SC matrix, fMRI, and EEG data"
                )
    return (
        SC_weights,
        EEG_timeseries,
        trans_EEG_timeseries,
        fMRI_timeseries,
        fMRI_timeseries_interp,
        trans_fMRI_timeseries,
        N,
        N_regions,
        EEG_timesteps,
    )


def plot_ex_interp(timeseries, timeseries_interp, ex_participant, ex_harmonic):
    """
    plots exemplary timeseries and interpolated timeseries for comparison (for 1 participant, 1 harmonic)
    arguments:
        timeseries: array before interpolation
        timeseries_interp: array after interpolation
        ex_participant: example participant index
        ex_harmonic: example harmonic index
    """
    plt.subplot(211)
    plt.plot(timeseries[ex_participant][ex_harmonic, :])
    plt.xlabel("time")
    plt.ylabel("signal")
    plt.title("original fMRI signal")
    plt.subplot(212)
    plt.plot(timeseries_interp[ex_participant][ex_harmonic, :])
    plt.xlabel("time")
    plt.ylabel("signal")
    plt.title("stretched fMRI signal")
    plt.tight_layout()
    plt.show()


def plot_ex_GFT_weights(
    trans_EEG_timeseries, trans_fMRI_timeseries, N_regions, ex_participant
):
    """
    plots exemplary GFT weights of EEG and interpolated fMRI activity over time (for 1 participant, 5 harmonics)
    arguments:
        trans_EEG_timeseries: GFT weights for EEG
        trans_fMRI_timeseries: GFT weights for fMRI
        N_regions: number of brain regions
        ex_participant: example participant index
    """
    # normalize each line?
    plt.subplot(211)
    for i in np.linspace(0, N_regions - 1, 5).astype(int):
        plt.plot(trans_fMRI_timeseries[ex_participant][i, :], label=f"harmonic {i+1}")
    plt.xlabel("time")
    plt.ylabel("signal")
    plt.title("examplary fMRI activities for one participant")
    plt.legend()
    plt.subplot(212)
    for i in np.linspace(0, N_regions - 1, 5).astype(int):
        plt.plot(trans_EEG_timeseries[ex_participant][i, :], label=f"harmonic {i+1}")
    plt.xlabel("time")
    plt.ylabel("signal")
    plt.title("examplary EEG activities for one participant")
    plt.legend()
    plt.tight_layout()
    plt.show()


# %%
SC_path = "../data/empirical_structural_connectomes/SCs.mat"
fMRI_data_path = "../data/empirical_fMRI/empirical_fMRI.mat"
EEG_data_path = "../data/empirical_source_activity/source_activity.mat"

ex_participant = 1
ex_harmonic = 5

(
    SC_weights,
    EEG_timeseries,
    trans_EEG_timeseries,
    fMRI_timeseries,
    fMRI_timeseries_interp,
    trans_fMRI_timeseries,
    N,
    N_regions,
    EEG_timesteps,
) = get_data(SC_path, EEG_data_path, fMRI_data_path)
# %%
plot_ex_interp(fMRI_timeseries, fMRI_timeseries_interp, ex_participant, ex_harmonic)

# %%
plot_ex_GFT_weights(
    trans_EEG_timeseries, trans_fMRI_timeseries, N_regions, ex_participant
)

# %%
# 10**5 used in paper for same plots
factor = 10**5
EEG_activity_scaled = EEG_timeseries[ex_participant] * factor
EEG_weights_scaled = trans_EEG_timeseries[ex_participant] * factor

# EEG activity in original domain
map = sns.heatmap(
    EEG_activity_scaled / np.max(EEG_activity_scaled),
    cbar_kws={"label": "activity $* 10^5$ [a.u.]"},
)
map.set_xlabel("time", fontsize=10)
map.set_ylabel("brain region", fontsize=10)
plt.title(
    f"EEG activity for all brain areas over time\n for participant {ex_participant+1}"
)
plt.show()
# EEG activity in graph frequency/spectral domain
map = sns.heatmap(
    EEG_weights_scaled / np.max(EEG_weights_scaled),
    cbar_kws={"label": "GFT weights $* 10^5$ [a.u.]"},
)
map.set_xlabel("time", fontsize=10)
map.set_ylabel("network harmonic", fontsize=10)
plt.title(
    f"EEG activity for all network harmonics over time\n for participant {ex_participant+1}"
)
plt.show()
print(
    "for lower plot, there should be a difference between top and bottom network harmonic activations"
)

# %%
EEG_power = trans_EEG_timeseries[ex_participant] ** 2
EEG_power_norm = EEG_power / np.sum(EEG_power)
# normalize power in every timestep
EEG_power_norm = EEG_power_norm / np.sum(EEG_power_norm, 0)

fMRI_power = trans_fMRI_timeseries[ex_participant] ** 2
fMRI_power_norm = fMRI_power / np.sum(fMRI_power)
# normalize power in every timestep
fMRI_power_norm = fMRI_power_norm / np.sum(fMRI_power_norm, 0)

# activity in original domain
map = sns.heatmap(
    EEG_power_norm,
    cbar_kws={"label": "EEG L2$^2$"},
)
map.set_xlabel("time", fontsize=10)
map.set_ylabel("brain region", fontsize=10)
plt.title(
    f"EEG power for all network harmonics over time\n for participant {ex_participant+1}"
)
plt.show()
# activity in graph frequency/spectral domain
map = sns.heatmap(
    fMRI_power_norm,
    cbar_kws={"label": "fMRI L2$^2$"},
)
map.set_xlabel("time", fontsize=10)
map.set_ylabel("network harmonic", fontsize=10)
plt.title(
    f"fMRI power for all network harmonics over time\n for participant {ex_participant+1}"
)
plt.show()
print(
    "for lower plot, there should be a difference between top and bottom network harmonic activations"
)

map = sns.heatmap(corrcoef2D(EEG_power_norm, fMRI_power_norm))
map.set_xlabel("EEG ?", fontsize=10)
map.set_ylabel("fMRI ?", fontsize=10)
plt.title(
    f"correlation of harmonics in fMRI and EEG for participant {ex_participant+1}"
)
plt.show()

# %%
# exemplary comparison EEG and fMRI timeseries data
plt.plot(
    fMRI_timeseries_interp[ex_participant][ex_harmonic, :]
    / np.max(np.abs(fMRI_timeseries_interp[ex_participant][ex_harmonic, :])),
    label="fMRI",
    alpha=0.7,
)
plt.plot(
    trans_EEG_timeseries[ex_participant][ex_harmonic, 100 : EEG_timesteps - 101]
    / np.max(
        np.abs(
            trans_EEG_timeseries[ex_participant][ex_harmonic, 100 : EEG_timesteps - 101]
        )
    ),
    label="EEG",
    alpha=0.7,
)
plt.legend()
plt.xlabel("time")
plt.ylabel("scaled signal")
plt.title(
    f"exemplary comparison of harmonic {ex_harmonic} in EEG and fMRI\n for participant {ex_participant + 1}"
)
plt.show()


# %%
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


a = np.array(([1, 3, 7], [1, 2, 7]))
b = np.array(([1, 3, 5], [1, 3, 7]))

print(f"a:\n{a}\nb:\n{b}")

print(f"row-wise corr:\n{corrcoef2D(a, b)}")

# %%
# look at fMRI-EEg timeseries correlation between harmonics
for participant in np.arange(N):
    fMRI_EEG_corr = corrcoef2D(
        fMRI_timeseries_interp[participant], trans_EEG_timeseries[participant]
    )
    map = sns.heatmap(fMRI_EEG_corr)
    map.set_xlabel("EEG ?", fontsize=10)
    map.set_ylabel("fMRI ?", fontsize=10)
    plt.title(
        f"correlation of harmonics in fMRI and EEG for participant {participant+1}"
    )
    plt.show()

# %%
ex_participant = 10
# mean power (L2 norm) over time
# or sqrt of L2 norm??
# square in right place?

# does this plot make sense with a mean over time? -> analogous to EEG/fMRI power plots above, otherwise timesteps instead of harmonics are important

# normalize power vector to 1 --> normalize power to 1 at every point in time????

# normalize power at every time point? and then also divide by number of regions?
temp = trans_EEG_timeseries[ex_participant] ** 2
temp = np.mean(temp / np.sum(temp, 0)[np.newaxis, :], 1)
plt.stem(temp)
plt.xlabel("harmonic")
plt.ylabel("signal strength")
plt.title(
    f"normalized graph frequency domain for participant {ex_participant + 1}\n (mean over time, normalized also per timepoint)"
)
plt.show()
for i in np.arange(50):
    curr_random = np.random.uniform(0, 1, N_regions)
    plt.plot(np.cumsum(curr_random) / np.sum(curr_random), color="grey", alpha=0.1)
plt.plot(np.cumsum(temp), label="SC graph")
plt.xlabel("harmonic")
plt.ylabel("cumulative power ?")
plt.title(
    f"Power captured cumulatively for participant {ex_participant + 1}\n (mean over time, normalized also per timepoint)"
)
handles, labels = plt.gca().get_legend_handles_labels()
line = Line2D([0], [0], label="random graphs", color="grey", alpha=0.1)
handles.extend(
    [
        line,
    ]
)
plt.legend(handles=handles)
plt.show()
"""
# normalize power only after taking the mean, not per time point?
t_mean_freq = np.mean(trans_EEG_timeseries[ex_participant] ** 2, 1)
t_mean_freq_norm = t_mean_freq / np.sum(t_mean_freq)
plt.stem(t_mean_freq_norm)
plt.xlabel("harmonic")
plt.ylabel("signal strength")
plt.title(
    f"normalized graph frequency domain for participant {ex_participant + 1} (mean over time)"
)
plt.show()
for i in np.arange(50):
    curr_random = np.random.uniform(0, 1, N_regions)
    plt.plot(np.cumsum(curr_random) / np.sum(curr_random), color="grey", alpha=0.1)
plt.plot(np.cumsum(t_mean_freq_norm), label="SC graph")
plt.xlabel("harmonic")
plt.ylabel("cumulative power ?")
plt.title(
    f"Power captured cumulatively for participant {ex_participant + 1} (mean over time)"
)
handles, labels = plt.gca().get_legend_handles_labels()
line = Line2D([0], [0], label="random graphs", color="grey", alpha=0.1)
handles.extend(
    [
        line,
    ]
)
plt.legend(handles=handles)
plt.show()
"""
# %%
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


print(check_symmetric(SC_weights[0]))
print(SC_weights[0])

# create graph for first participant & check properties
G1 = graphs.Graph(SC_weights[0])
print("{} nodes, {} edges".format(G1.N, G1.Ne))

print(f"connected: {G1.is_connected()}")
print(f"directed: {G1.is_directed()}")  # why is it directed?

print(G1.check_weights())
print(f"weights as initialized: {(G1.W == SC_weights[0]).all()}")
