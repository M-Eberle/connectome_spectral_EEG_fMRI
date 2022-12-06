# %%
from helpers import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import io as sio
from scipy.stats import ttest_ind

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
# - compare patterns between participants
### next steps
# - compare timeseries EEG & fMRI (e.g. lower vs higehr half of harmonics)
#### other ToDos
# - fix time axis in all plots
# - save plots
# - data in numpy arrays instead of lists
# - compare scipy.interpolate.interp1d & scipy.signal.resample
# - general refactoring

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

# freqs, psd = plot_fft_welch(EEG_timeseries[ex_participant][ex_region, :], sampling_freq)
# use all EEG data instead
freqs, psd = plot_fft_welch(np.array(EEG_timeseries).flatten(), sampling_freq)
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
# test hypotheses
# %%
# compare vertex vs graph domain: correlation between EEG & fMRI
mean_regions_corrs, mean_harmonics_corrs, ttest_results = vertex_vs_graph(
    EEG_timeseries, fMRI_timeseries_interp, trans_EEG_timeseries, trans_fMRI_timeseries
)

# %%
# compare similarity measures between all p[articipants]
simi_betw_participants(Gs, simi_TVG, "TVG", "fMRI", trans_fMRI_timeseries)
simi_betw_participants(Gs, simi_TVG, "TVG", "EEG", trans_EEG_timeseries)
# scale for EEG a lot smaller than for fMRI

simi_betw_participants(Gs, simi_GE, "GE")

simi_betw_participants(Gs, simi_JET, "JET", "fMRI", trans_fMRI_timeseries)
simi_betw_participants(Gs, simi_JET, "JET", "EEG", trans_EEG_timeseries)

# %%
# compare lower vs upper half of harmonics
N_regions = len(trans_fMRI_timeseries[0])
half = int(N_regions / 2)
# instead of GFT with all harmonics


def power_mean(
    signal,
    ex_participant,
    mode,
    low_harm=0,
    high_harm=68,
):
    """
    plots mean power over time (1 participant, all harmonics)
    arguments:
        signal: GFT weights
        ex_participant: example participant index
        mode: string, should be 'EEG' or 'fMRI'
    return:
        power: power per harmonic
    """
    # mean power (L2 norm) over time
    # or sqrt of L2 norm??
    # square in right place?
    # does this make sense with a mean over time? -> analogous to EEG/fMRI power plots above, otherwise timesteps instead of harmonics are important
    # normalize power vector to 1 --> normalize power to 1 at every point in time????
    # normalize power at every time point? and then also divide by number of regions?
    power = signal[ex_participant][low_harm:high_harm, :] ** 2
    power = np.mean(power / np.sum(power, 0)[np.newaxis, :], 1)
    print("mean")
    print(power)
    plt.stem(power)
    plt.xlabel("harmonic")
    plt.ylabel("signal strength")
    plt.title(
        f"{mode} normalized graph frequency domain for participant {ex_participant + 1}\n (mean over time, normalized also per timepoint)"
    )
    plt.show()
    return power


fMRI_power_low = power_mean(trans_fMRI_timeseries, ex_participant, "fMRI", 0, half)
fMRI_power_high = power_mean(
    trans_fMRI_timeseries, ex_participant, "fMRI", half, N_regions
)


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
    results = ttest_ind(a, b, alternative="greater")
    return results


ttest_greater(
    fMRI_power_low,
    fMRI_power_high,
    [["low", "high"], "power", "power between lower and higher harmonics"],
)
