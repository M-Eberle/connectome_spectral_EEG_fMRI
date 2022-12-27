# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.interpolate as interp

from scipy import io as sio
from scipy.stats import ttest_ind
from pygsp import graphs

from helpers import *

# %% [markdown]

### last steps
# - load data
# - sort data (participants, regions)
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


class data:
    """
    class for all data and methods for generation of overview plots and analysis
    """

    def __init__(self, mode, loop_participants=True):

        self.mode = mode
        self.loop_participants = loop_participants

        # paths
        # data
        self.SC_path = "../data/empirical_structural_connectomes/SCs.mat"
        self.fMRI_data_path = "../data/empirical_fMRI/empirical_fMRI.mat"
        self.EEG_data_path = "../data/empirical_source_activity/source_activity.mat"
        # EEG regions
        self.EEG_regions_path = "../data/empirical_source_activity/regionsMap.mat"
        # region coordinates
        self.coords_path = "../data/ROI_coords.mat"
        # HRF
        self.HRF_path = "../data/HRF_200Hz.mat"

        # known variables about data
        self.N = 15
        self.N_regions = 68
        self.fMRI_timesteps = 661
        self.EEG_timesteps = 259184
        self.sampling_freq = 200  # Hz, sampling frequency for EEG

        # other variable settings
        self.ex_participant = 1
        self.ex_harmonic = 5
        self.ex_region = 5

        # initialize variables for data and IDs
        # structural connectivity matrices  graphs
        self.G = None
        self.Gs = []
        self.SC_weights = np.empty((self.N_regions, self.N_regions, self.N))
        self.mean_SC_weights = None
        self.SC_participant_IDs = []
        self.SC_region_IDs = []
        # EEG: (68x259184)x15 for (RegionsxTime)xparticipants
        self.EEG_timeseries = np.empty((self.N_regions, self.EEG_timesteps, self.N))
        self.trans_EEG_timeseries = np.empty(
            (self.N_regions, self.EEG_timesteps, self.N)
        )
        self.EEG_participant_IDs = []
        self.EEG_region_IDs = []
        # fMRI: before interpolation: (68x661)x15 for (RegionsxTime)xparticipants
        # after interpolation as in EEG: (68x259184)x15 for (RegionsxTime)xparticipants
        self.fMRI_timeseries = np.empty((self.N_regions, self.fMRI_timesteps, self.N))
        self.fMRI_timeseries_interp = np.empty(
            (self.N_regions, self.EEG_timesteps, self.N)
        )
        self.trans_fMRI_timeseries = np.empty(
            (self.N_regions, self.EEG_timesteps, self.N)
        )
        self.fMRI_participant_IDs = []
        self.fMRI_region_IDs = []
        # other
        self.coords = None
        self.HRF_resEEG = sio.loadmat(self.HRF_path)["HRF_resEEG"]

        # get data
        self._get_coords()
        self._get_SC_matrices()
        self._get_functional_data()

    def _loop_participants(func):
        def wrapper(*args):
            if args.loop_participants:
                for participant in np.arange(args.N):
                    args.ex_participant = participant
                    func(*args)
            else:
                func(*args)

    def _get_coords(self):

        coords_all = sio.loadmat(self.coords_path)["ROI_coords"]
        coords_labels = sio.loadmat("../data/ROI_coords.mat")["all_labels"].flatten()
        self.roi_IDs = np.hstack(
            sio.loadmat("../data/empirical_fMRI/empirical_fMRI.mat")[
                "freesurfer_roi_IDs"
            ][0].flatten()
        ).astype("int32")
        idx_rois = np.empty((len(self.roi_IDs)), dtype="int32")
        for idx, roi in enumerate(self.roi_IDs):
            idx_rois[idx] = np.argwhere(coords_labels == roi)
        self.coords = coords_all[idx_rois]

    def _get_SC_matrices(self):
        unflattened_SC_data = sio.loadmat(self.SC_path)["SC"]
        SC_data = np.ndarray.flatten(unflattened_SC_data)

        for participant in np.arange(self.N):
            SC_weights_pre, SC_participant_ID_curr = SC_data[participant]
            # extract weights from nested object
            # make weights symmetric t keep going for now
            SC_weights_pre[0, 0][0] = (
                SC_weights_pre[0, 0][0] + SC_weights_pre[0, 0][0].T
            )
            self.SC_weights[:, :, participant] = SC_weights_pre[0, 0][0]
            # add participant ID
            self.SC_participant_IDs.append(SC_participant_ID_curr)

    def _get_functional_data(self):

        self.__get_raw_functional_data()
        self.__sort_EEG_data()

        # transform data
        if self.mode == "ind":
            self.__transform_w_ind_SCs()
        else:
            self.__transform_w_mean_SC()

        self.__check_participant_IDs()
        self.__check_region_IDs()

    def __get_raw_functional_data(self):

        # load fMRI data
        unflattened_fMRI_data = sio.loadmat(self.fMRI_data_path)["fMRI"]
        fMRI_data = np.ndarray.flatten(unflattened_fMRI_data)

        # load EEG data
        EEG_data_file = h5py.File(self.EEG_data_path, "r")

        for participant in np.arange(self.N):

            # get EEG participant IDs
            EEG_ID_store_curr = EEG_data_file["source_activity/sub_id"][participant][0]
            EEG_participant_ID = EEG_data_file[EEG_ID_store_curr][:]
            EEG_participant_ID = (
                "['" + "".join(chr(c[0]) for c in EEG_participant_ID) + "']"
            )
            self.EEG_participant_IDs.append(EEG_participant_ID)
            # get participant's EEG data
            EEG_data_store_curr = EEG_data_file["source_activity/ts"][participant][0]
            EEG_timeseries_curr = EEG_data_file[EEG_data_store_curr]
            # subtract mean over regions ?
            EEG_timeseries_curr = EEG_timeseries_curr - np.mean(EEG_timeseries_curr, 0)
            self.EEG_timeseries[:, :, participant] = EEG_timeseries_curr[:, :].T

            # get participant's fMRI data
            fMRI_timeseries_curr, fMRI_participant_ID = fMRI_data[participant]
            self.fMRI_timeseries[:, :, participant] = fMRI_timeseries_curr.T
            self.fMRI_participant_IDs.append(fMRI_participant_ID)
            # stretch fMRI data over time to EEG sequence length
            for region in np.arange(self.N_regions):
                fMRI_interp = interp.interp1d(
                    np.arange(self.fMRI_timesteps),
                    self.fMRI_timeseries[region, :, participant],
                )
                self.fMRI_timeseries_interp[region, :, participant] = fMRI_interp(
                    np.linspace(0, self.fMRI_timesteps - 1, self.EEG_timesteps)
                )
        # plot interpolated fMRI timeseries
        plot_ex_interp(
            self.fMRI_timeseries,
            self.fMRI_timeseries_interp,
            self.ex_participant,
            self.ex_harmonic,
        )

    def __sort_EEG_data(self):

        """
        EEG data is known to have different region sorting than (f)MRI data
        """
        self.SC_region_IDs = np.hstack(
            sio.loadmat(self.SC_path)["freesurfer_roi_IDs"][0].flatten()
        ).astype("uint16")

        self.EEG_region_IDs = sio.loadmat(self.EEG_regions_path)["regionsMap"][:, 0]

        self.fMRI_region_IDs = np.hstack(
            sio.loadmat(self.fMRI_data_path)["freesurfer_roi_IDs"][0].flatten()
        ).astype("uint16")

        region_sort_EEG_after_SC = np.empty((self.N_regions), dtype=int)
        for region in np.arange(self.N_regions):
            region_sort_EEG_after_SC[region] = np.argwhere(
                self.EEG_region_IDs == self.SC_region_IDs[region]
            )

        self.EEG_timeseries = self.EEG_timeseries[region_sort_EEG_after_SC, :, :]
        self.EEG_region_IDs = self.EEG_region_IDs[region_sort_EEG_after_SC]

    def __transform_w_mean_SC(self):

        self.mean_SC_weights = np.mean(self.SC_weights, 2)

        # compute one graph for all participants
        self.G = graphs.Graph(
            self.mean_SC_weights, lap_type="normalized", coords=self.coords
        )
        self.G.compute_fourier_basis()  # harmonics in G.U, eigenvalues in G.e

        # get spectral representation of signal
        for participant in np.arange(self.N):
            self.trans_EEG_timeseries[:, :, participant] = self.G.gft(
                self.EEG_timeseries[:, :, participant]
            )
            self.trans_fMRI_timeseries[:, :, participant] = self.G.gft(
                self.fMRI_timeseries_interp[:, :, participant]
            )

    def __transform_w_ind_SCs(self):

        for participant in np.arange(self.N):

            # compute individual graph for each participant
            G = graphs.Graph(
                self.SC_weights[:, :, participant],
                lap_type="normalized",
                coords=self.coords,
            )
            G.compute_fourier_basis()  # harmonics in G.U, eigenvalues in G.e
            self.Gs.append(G)

            # get spectral representation of signal
            self.trans_EEG_timeseries[:, :, participant] = self.Gs[participant].gft(
                self.EEG_timeseries[:, :, participant]
            )
            self.trans_fMRI_timeseries[:, :, participant] = self.Gs[participant].gft(
                self.fMRI_timeseries_interp[:, :, participant]
            )

    def __check_participant_IDs(self):
        ID_count = 0
        for participant in np.arange(self.N):
            if (
                str(self.SC_participant_IDs[participant])
                == str(self.EEG_participant_IDs[participant])
            ) and (
                str(self.SC_participant_IDs[participant])
                == str(self.fMRI_participant_IDs[participant])
            ):
                ID_count += 1

        if ID_count == self.N:
            print(
                "all participant IDs are represented by the same indices in SC matrix, fMRI, and EEG data"
            )

    def __check_region_IDs(self):
        ID_count = 0
        for region in np.arange(self.N_regions):
            if (
                str(self.SC_region_IDs[region]) == str(self.EEG_region_IDs[region])
            ) and (
                str(self.SC_region_IDs[region]) == str(self.fMRI_region_IDs[region])
            ):
                ID_count += 1

        if ID_count == self.N_regions:
            print(
                "all region IDs are represented by the same indices in SC matrix, fMRI, and EEG data"
            )

    @_loop_participants
    def plot_ex_signal(self):
        plot_ex_signal_EEG_fMRI(
            self.EEG_timeseries,
            self.fMRI_timeseries_interp,
            self.ex_participant,
            "region",
        )


# %%
mode = "mean"  # 'mean' or 'ind'
# data_mean = data('mean')
data_ind = data("ind")

# %%
data_ind.plot_ex_signal()
# %%
#


for ex_participant in np.arange(N):
    plot_ex_signal_EEG_fMRI(
        EEG_timeseries, fMRI_timeseries_interp, ex_participant, "region"
    )
# %%
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
freqs, psd = plot_fft_welch(np.moveaxis(EEG_timeseries, -1, 0).flatten(), sampling_freq)

# alpha-activity very prominent

# %%
# EEG sanity check 3: EEG-fMRI -> see helpers/sanity_check.py
(
    all_corrs_reg,
    all_corrs_power,
    mean_reg_all,
    mean_power_all,
    shifts_reg,
    shifts_power,
) = alpha_mean_corrs(fMRI_timeseries, EEG_timeseries, regionsMap, HRF_resEEG)

# very low correlation? -> also, not necessarily negative correlation? sometimes more positive
# filtered has second filter applied (low?)
# %%
import seaborn as sns

min = np.min((np.min(all_corrs_reg), np.min(all_corrs_power)))
max = np.max((np.max(all_corrs_reg), np.max(all_corrs_power)))
for s in np.arange(10):  # 10 is max shift?
    sns.heatmap(all_corrs_reg[:, :, s], vmin=min, vmax=max)
    plt.xlabel("participant idx")
    plt.ylabel("region")
    plt.title(f"EEG-fMRI over regions and subjects for shift {s+1}\nalpha regressor")
    plt.show()
    sns.heatmap(all_corrs_power[:, :, s], vmin=min, vmax=max)
    plt.xlabel("participant idx")
    plt.ylabel("region")
    plt.title(f"EEG-fMRI over regions and subjects for shift {s+1}\nalpha powerband")
    plt.show()

# %%
# compare to MATLAB regressor

ex_participant = 0
alpha_reg, alpha_reg_filt = compute_alpha_regressor(
    EEG_timeseries, ex_participant, regionsMap, HRF_resEEG, sampling_freq
)

print(alpha_reg.T[0, 44])
print(alpha_reg.T[15, 3])
print(alpha_reg.T[386, 16])
plt.plot(alpha_reg.T)
plt.title("Python: alpha regressor for participant 1")
plt.xlabel("time")
plt.ylabel("value")
# %%
plot_compare_alpha(mean_reg_all, mean_power_all, shifts_reg, shifts_power)
# __________________

# %% [markdown]
# test hypotheses
# %%
# compare vertex vs graph domain: correlation between EEG & fMRI
(
    regions_all_corrs,
    harmonics_all_corrs,
    mean_regions_corrs,
    mean_harmonics_corrs,
    ttest_results,
) = vertex_vs_graph(
    EEG_timeseries, fMRI_timeseries_interp, trans_EEG_timeseries, trans_fMRI_timeseries
)

# %%
min = np.min((np.min(regions_all_corrs), np.min(harmonics_all_corrs)))
max = np.max((np.max(regions_all_corrs), np.max(harmonics_all_corrs)))
sns.heatmap(regions_all_corrs, vmin=min, vmax=max)
plt.xlabel("participant idx")
plt.ylabel("region")
plt.title("EEG-fMRI correlation within regions")
plt.show()
sns.heatmap(harmonics_all_corrs, vmin=min, vmax=max)
plt.xlabel("participant idx")
plt.ylabel("harmonic")
plt.title("EEG-fMRI correlation within harmonics")

# %%
# compare similarity measures between all participants
simi_betw_participants(Gs, simi_TVG, "TVG", "fMRI", trans_fMRI_timeseries)
simi_betw_participants(Gs, simi_TVG, "TVG", "EEG", trans_EEG_timeseries)

# scale for EEG a lot smaller than for fMRI

simi_betw_participants(Gs, simi_GE, "GE")

simi_betw_participants(Gs, simi_JET, "JET", "fMRI", trans_fMRI_timeseries)
simi_betw_participants(Gs, simi_JET, "JET", "EEG", trans_EEG_timeseries)
# %%
# cut off start & end of EEG data due to weird spikes
"""
N_reg, N_time = EEG_timeseries[0].shape
cut = 5000
EEG_power_all = np.empty((N_reg, N))
for participant in np.arange(N):
    signal = EEG_timeseries[participant][:, cut:N_time-cut]
    plt.plot(signal.T)
    plt.title(f"EEG for participant {participant+1}")
    plt.show()
    power = signal ** 2
    power = np.mean(power / np.sum(power, 0)[np.newaxis, :], 1)
    EEG_power_all[:,participant] = power
    print("mean")
    print(np.mean(power))
    plt.stem(power)
    plt.xlabel("harmonic")
    plt.ylabel("signal strength")
    plt.title(
        f"{mode} normalized graph frequency domain for participant {participant + 1}\n (mean over time, normalized also per timepoint)"
    )
    plt.show()

plt.stem(np.mean(EEG_power_all, 1))
#plt.axhline(np.mean(EEG_power_all))
plt.plot(np.cumsum(np.mean(EEG_power_all, 1)))
for i in np.arange(50):
    curr_random = np.random.uniform(0, 1, N_regions)
    plt.plot(np.cumsum(curr_random) / np.sum(curr_random), color="grey", alpha=0.1)
"""

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

# %%
# plot eigenvectors on graphs

N_plots = 3

fig, axes = plt.subplots(1, N_plots, figsize=(10, 3), subplot_kw=dict(projection="3d"))
for t, ax in enumerate(axes):
    G.plot_signal(
        G.U[:, t],
        vertex_size=30,
        show_edges=True,
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"harmonic {t + 1}")
    ax.axis("off")
    plt.suptitle(f"first {N_plots} harmonics")
fig.tight_layout()
plt.show()

fig, axes = plt.subplots(1, N_plots, figsize=(10, 3), subplot_kw=dict(projection="3d"))
for t, ax in enumerate(axes):
    G.plot_signal(
        G.U[:, -(t + 1)],
        vertex_size=30,
        show_edges=True,
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"harmonic {68 -t}")
    ax.axis("off")
    plt.suptitle(f"last {N_plots} harmonics")
fig.tight_layout()
plt.show()


# %%
fig, axes = plt.subplots(1, N_plots, figsize=(10, 3), subplot_kw=dict(projection="3d"))
for t, ax in enumerate(axes):
    G.plot_signal(
        G.U[:, t + 30],
        vertex_size=30,
        show_edges=True,
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"harmonic {t+31}")
    ax.axis("off")
    plt.suptitle(f"last {N_plots} harmonics")
fig.tight_layout()
plt.show()
