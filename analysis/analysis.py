# %%
from helpers import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import io as sio

# %% [markdown]

### last steps
# - load data
# - transform data onto graph
# - check that participant's indices in SC, fMRI, and EEG align
# - symmetry of SCs? --> use +transpose for now
# - interpolate fMRI
# - look at data
#   - activity, harmonics, power
#   - correlations between fMRI & EEG
# - sanity checks for EEG data
#   - EEG power, mean over participants
#   - EEG frequency band
#   - EEG-fMRI -> alpha regressor, alpha power band vs fMRI timeseries
# - compare graph vs vertex domain: corr EEG-fMRI
# - plot signal on graph
### next steps
# - compare timeseries EEG & fMRI (e.g. lower vs higehr half of harmonics)
# - compare patterns between participants
#### other ToDos
# - fix time axis in all plots
# - save plots
# - data in numpy arrays instead of lists
# - compare scipy.interpolate.interp1d & scipy.signal.resample

# %%
SC_path = "../data/empirical_structural_connectomes/SCs.mat"
fMRI_data_path = "../data/empirical_fMRI/empirical_fMRI.mat"
EEG_data_path = "../data/empirical_source_activity/source_activity.mat"

regionsMap = sio.loadmat("../data/empirical_source_activity/regionsMap.mat")[
    "regionsMap"
][:, 0]
HRF_resEEG = sio.loadmat("../data/HRF_200Hz.mat")[
    "HRF_resEEG"
]  # hemodynamic response function

coords_all = sio.loadmat("../data/ROI_coords.mat")["ROI_coords"]
coords_labels = sio.loadmat("../data/ROI_coords.mat")["all_labels"].flatten()
roi_IDs = np.hstack(
    sio.loadmat("../data/empirical_fMRI/empirical_fMRI.mat")["freesurfer_roi_IDs"][
        0
    ].flatten()
).astype("int32")
idx_rois = np.empty((len(roi_IDs)), dtype="int32")
for idx, roi in enumerate(roi_IDs):
    idx_rois[idx] = np.argwhere(coords_labels == roi)
coords = coords_all[idx_rois]

ex_participant = 1
ex_harmonic = 5
ex_region = 5
sampling_freq = 200  # Hz, sampling frequency for EEG
mode = "ind"  # ind or mean

if mode == "ind":
    (
        Gs,
        SC_weights,
        EEG_timeseries,
        trans_EEG_timeseries,
        fMRI_timeseries,
        fMRI_timeseries_interp,
        trans_fMRI_timeseries,
        N,
        N_regions,
        EEG_timesteps,
    ) = get_data_ind_SCs(SC_path, EEG_data_path, fMRI_data_path, coords)
else:
    (
        G,
        mean_SC_weights,
        EEG_timeseries,
        trans_EEG_timeseries,
        fMRI_timeseries,
        fMRI_timeseries_interp,
        trans_fMRI_timeseries,
        N,
        N_regions,
        EEG_timesteps,
    ) = get_data_mean_SC(SC_path, EEG_data_path, fMRI_data_path, coords)

N = len(trans_EEG_timeseries)
plot_ex_interp(fMRI_timeseries, fMRI_timeseries_interp, ex_participant, ex_harmonic)


# %%
plot_ex_signal_EEG_fMRI(
    EEG_timeseries, fMRI_timeseries_interp, ex_participant, "region"
)
plot_ex_signal_EEG_fMRI(
    trans_EEG_timeseries, trans_fMRI_timeseries, ex_participant, "harmonic"
)
# %%
plot_ex_regions_harmonics(EEG_timeseries, trans_EEG_timeseries, ex_participant, "EEG")
plot_ex_regions_harmonics(
    fMRI_timeseries_interp, trans_fMRI_timeseries, ex_participant, "fMRI"
)
# %%
plot_ex_signal_fMRI_EEG_one(
    EEG_timeseries, fMRI_timeseries_interp, ex_participant, ex_region, "region"
)
plot_ex_signal_fMRI_EEG_one(
    trans_EEG_timeseries, trans_fMRI_timeseries, ex_participant, ex_harmonic, "harmonic"
)
# %%
plot_ex_graphs_3D(Gs, trans_EEG_timeseries, ex_participant, "EEG")
plot_ex_graphs_3D(Gs, trans_fMRI_timeseries, ex_participant, "fMRI")

# %%
EEG_power_norm = power_norm(trans_EEG_timeseries, ex_participant)
fMRI_power_norm = power_norm(trans_fMRI_timeseries, ex_participant)
plot_ex_power_EEG_fMRI(EEG_power_norm, fMRI_power_norm, ex_participant)
plot_power_corr(EEG_power_norm, fMRI_power_norm, ex_participant)

# %%
regions_corr = ex_EEG_fMRI_corr(
    EEG_timeseries, fMRI_timeseries_interp, ex_participant, "region"
)
harmonics_corr = ex_EEG_fMRI_corr(
    trans_EEG_timeseries, trans_fMRI_timeseries, ex_participant, "harmonic"
)
print(
    f"regions: {np.round(np.mean(np.abs(regions_corr)), 4)}, harmonics: {np.round(np.mean(np.abs(harmonics_corr)), 4)}"
)
print(
    f"regions diag: {np.round(np.mean(np.abs(np.diag(regions_corr))), 4)}, harmonics diag: {np.round(np.mean(np.abs(np.diag(harmonics_corr))), 4)}"
)

# ? test for significance somehow?
# ? absolute correlation between regions higher than harmonics - even on diagonal

# %%
# restructure data as numpy array
EEG_power = np.empty((68, N))
fMRI_power = np.empty((68, N))
for participant in np.arange(N):
    EEG_power[:, participant] = power_mean(trans_EEG_timeseries, participant, "EEG")
    fMRI_power[:, participant] = power_mean(trans_fMRI_timeseries, participant, "fMRI")

# %%
# cumulative power for example participant
plot_cum_power(EEG_power[:, ex_participant], ex_participant, "EEG")
plot_cum_power(fMRI_power[:, ex_participant], ex_participant, "fMRI")

# %%
# EEG sanity check 1: EEG power, mean over participants
# ________________________
# find mean power over all participants ( & over time)
power_mean_EEG = np.mean(EEG_power, 1)
power_mean_fMRI = np.mean(fMRI_power, 1)
# cumulative power, mean over participants (plot titles not correct)
plot_cum_power(power_mean_EEG, ex_participant, "EEG")
plot_cum_power(power_mean_fMRI, ex_participant, "fMRI")

# %%
# EEG sanity check 2: EEG frequency band
# ________________________
# NO means (separate for each region and participant)

freqs, psd = plot_fft_welch(EEG_timeseries[ex_participant][ex_region, :], sampling_freq)

# alpha-activity very prominent

# %%
# EEG sanity check 3: EEG-fMRI -> see helpers/sanity_check.py
(
    alpha_reg,
    alpha_reg_filt,
    alpha_power,
    alpha_power_filt,
    mean_reg_all,
    mean_power_all,
    shifts_reg,
    shifts_power,
) = alpha_mean_corrs(fMRI_timeseries, EEG_timeseries, regionsMap, HRF_resEEG)
# very low correlation? -> also, not necessarily negative correlation? sometimes more positive
# filtered has second filter applied (low?)
# %%
plot_compare_alpha(mean_reg_all, mean_power_all, shifts_reg, shifts_power)
# __________________

# %% [markdown]
# measures

# %%
# look at total variation and laplacian graph energy
for participant in np.arange(N):
    if mode == "ind":
        print(
            f"fMRI: TV: {np.round(TV(Gs[participant], trans_fMRI_timeseries[participant]), 4)}, LE: {np.round(LE(Gs[participant]), 4)}"
        )
        print(
            f"EEG: TV: {np.round(TV(Gs[participant], trans_EEG_timeseries[participant]), 4)}, LE: {np.round(LE(Gs[participant]), 4)}"
        )
    else:
        print(
            f"fMRI: TV: {np.round(TV(G, trans_fMRI_timeseries[participant]), 4)}, LE: {np.round(LE(G), 4)}"
        )
        print(
            f"fMRI: TV: {np.round(TV(G, trans_EEG_timeseries[participant]), 4)}, LE: {np.round(LE(G), 4)}"
        )

# ? TV only 0 for EEG data?

# %%
# for fMRI: compare TV between all p[articipants]
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


fMRI_TVGs = np.empty((N, N))
for participant1 in np.arange(N):
    for participant2 in np.arange(N):
        if participant1 <= participant2:
            fMRI_TVGs[participant1, participant2] = simi_TV(
                Gs[participant1],
                Gs[participant2],
                trans_fMRI_timeseries[participant1],
                trans_fMRI_timeseries[participant2],
            )
        else:
            fMRI_TVGs[participant1, participant2] = fMRI_TVGs[
                participant2, participant1
            ]

sns.heatmap(fMRI_TVGs)

# use unnormalized adjacency to compare EEG & fMRI for now?
